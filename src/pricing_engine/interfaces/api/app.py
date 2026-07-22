"""FastAPI application factory and operational endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from hashlib import sha256
from secrets import compare_digest
from time import perf_counter
from typing import cast
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from pricing_engine.config import Settings, get_settings
from pricing_engine.domain.exceptions import (
    ModelUnavailableError,
    PricingEngineError,
)
from pricing_engine.domain.policies import PricingPolicy
from pricing_engine.interfaces.api.container import ApplicationContainer
from pricing_engine.interfaces.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PricingRecommendationRequest,
    PricingRecommendationResponse,
)

REQUEST_COUNTER = Counter(
    "pricing_api_requests_total",
    "Number of API requests by endpoint and response status.",
    ["path", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "pricing_api_request_duration_seconds",
    "API request duration by path.",
    ["path"],
)
logger = structlog.get_logger(__name__)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
RECOMMENDATION_PATH = "/v1/pricing/recommendations"


class RequestBodyLimitMiddleware:
    """Bound the expensive JSON endpoint before FastAPI parses its body."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        maximum_bytes: int,
        read_timeout_seconds: float,
        path: str,
        api_key_digest: bytes | None,
    ) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes
        self.read_timeout_seconds = read_timeout_seconds
        self.path = path
        self.api_key_digest = api_key_digest

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self.path
        ):
            await self.app(scope, receive, send)
            return

        if self.api_key_digest is not None and not self._has_valid_api_key(scope):
            await self._reject_unauthorized(scope, receive, send)
            return

        content_length = self._content_length(scope)
        if content_length is not None and content_length > self.maximum_bytes:
            await self._reject(scope, receive, send)
            return

        messages: list[Message] = []
        received_bytes = 0
        try:
            async with asyncio.timeout(self.read_timeout_seconds):
                while True:
                    message = await receive()
                    messages.append(message)
                    if message["type"] == "http.disconnect":
                        break
                    received_bytes += len(message.get("body", b""))
                    if received_bytes > self.maximum_bytes:
                        await self._reject(scope, receive, send)
                        return
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await self._reject_timeout(scope, receive, send)
            return

        message_index = 0

        async def replay_receive() -> Message:
            nonlocal message_index
            if message_index < len(messages):
                message = messages[message_index]
                message_index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() == b"content-length":
                try:
                    return int(raw_value)
                except ValueError:
                    return None
        return None

    def _has_valid_api_key(self, scope: Scope) -> bool:
        expected = self.api_key_digest
        if expected is None:
            return True
        supplied_values = [
            value for name, value in scope.get("headers", []) if name.lower() == b"x-api-key"
        ]
        return any(compare_digest(sha256(value).digest(), expected) for value in supplied_values)

    @staticmethod
    async def _reject_unauthorized(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "HTTPException", "detail": "Invalid API key."},
            headers={"WWW-Authenticate": "ApiKey"},
        )
        await response(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "error": "RequestTooLarge",
                "detail": "Request body exceeds the configured limit.",
            },
        )
        await response(scope, receive, send)

    @staticmethod
    async def _reject_timeout(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            content={
                "error": "RequestTimeout",
                "detail": "Request body was not received before the configured deadline.",
            },
        )
        await response(scope, receive, send)


def _safe_request_id(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID")
    if (
        supplied
        and len(supplied) <= 128
        and all(character.isalnum() or character in "._:-" for character in supplied)
    ):
        return supplied
    return str(uuid4())


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return template if isinstance(template, str) else "unmatched"


def _configure_logging(log_level: str) -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level.upper()),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
    )


def create_app(
    settings: Settings | None = None,
    *,
    container: ApplicationContainer | None = None,
) -> FastAPI:
    """Create an independently testable FastAPI application."""

    resolved_settings = settings or get_settings()
    _configure_logging(resolved_settings.log_level)
    resolved_container = container or ApplicationContainer(resolved_settings)
    is_production = resolved_settings.environment == "production"
    recommendation_slots = asyncio.Semaphore(resolved_settings.recommendation_max_concurrency)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        resolved_container.load_configured_model()
        yield

    app = FastAPI(
        title=resolved_settings.api_title,
        version=resolved_settings.api_version,
        lifespan=lifespan,
        debug=False,
        docs_url=None if is_production else "/docs",
        redoc_url=None,
        openapi_url=None if is_production else "/openapi.json",
    )
    app.state.container = resolved_container
    if resolved_settings.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=resolved_settings.trusted_hosts,
        )
    allowed_origins = [str(origin).rstrip("/") for origin in resolved_settings.cors_origins]
    if allowed_origins:
        allowed_headers = ["Content-Type", "X-API-Key", "X-Request-ID"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=allowed_headers,
        )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        maximum_bytes=resolved_settings.max_request_body_bytes,
        read_timeout_seconds=resolved_settings.request_body_timeout_seconds,
        path=RECOMMENDATION_PATH,
        api_key_digest=(
            sha256(resolved_settings.api_key.get_secret_value().encode("utf-8")).digest()
            if resolved_settings.api_key is not None
            else None
        ),
    )

    @app.middleware("http")
    async def telemetry(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = _safe_request_id(request)
        request.state.request_id = request_id
        started = perf_counter()
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        failure_type: str | None = None
        try:
            response: Response = await call_next(request)
            response_status = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = (
                "camera=(), microphone=(), geolocation=(), payment=()"
            )
            response.headers["Cache-Control"] = "no-store"
            if is_production:
                response.headers["Content-Security-Policy"] = (
                    "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
                )
            return response
        except Exception as error:
            failure_type = type(error).__name__
            raise
        finally:
            elapsed = perf_counter() - started
            route_label = _route_label(request)
            REQUEST_COUNTER.labels(
                path=route_label,
                method=request.method,
                status=str(response_status),
            ).inc()
            REQUEST_LATENCY.labels(path=route_label).observe(elapsed)
            logger.info(
                "http_request_completed",
                route=route_label,
                method=request.method,
                status=response_status,
                duration_ms=round(elapsed * 1000, 2),
                request_id=request_id,
                failure_type=failure_type,
            )

    @app.exception_handler(PricingEngineError)
    async def pricing_error_handler(_: Request, error: PricingEngineError) -> JSONResponse:
        response_status = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if isinstance(error, ModelUnavailableError)
            else status.HTTP_422_UNPROCESSABLE_CONTENT
        )
        detail = (
            "Pricing model is temporarily unavailable."
            if isinstance(error, ModelUnavailableError)
            else str(error)
        )
        return JSONResponse(
            status_code=response_status,
            content={"error": error.__class__.__name__, "detail": detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, error: RequestValidationError) -> JSONResponse:
        sanitized_errors = [
            {key: value for key, value in issue.items() if key in {"type", "loc", "msg"}}
            for issue in error.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "RequestValidationError",
                "detail": sanitized_errors,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_: Request, error: HTTPException) -> JSONResponse:
        detail = error.detail if isinstance(error.detail, str) else "Request rejected."
        return JSONResponse(
            status_code=error.status_code,
            content={"error": "HTTPException", "detail": detail},
            headers=error.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, error: Exception) -> JSONResponse:
        """Return one safe, stable envelope without exposing internal exception details."""

        request_id = getattr(request.state, "request_id", str(uuid4()))
        logger.exception(
            "unhandled_http_error",
            request_id=request_id,
            failure_type=type(error).__name__,
        )
        headers = {
            "X-Request-ID": request_id,
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
            "Cache-Control": "no-store",
        }
        if is_production:
            headers["Content-Security-Policy"] = (
                "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
            )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "detail": "An unexpected internal error occurred.",
            },
            headers=headers,
        )

    def authenticate_api_key(
        x_api_key: str | None = Security(API_KEY_HEADER),
    ) -> None:
        """Authenticate the caller without exposing or logging the credential."""

        configured = resolved_settings.api_key
        if configured is None:
            return
        if x_api_key is None or not compare_digest(
            x_api_key.encode("utf-8"),
            configured.get_secret_value().encode("utf-8"),
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    def resolve_authorized_tenant(
        request: Request,
        _: None = Depends(authenticate_api_key),
    ) -> str | None:
        """Resolve a tenant only from a configured key binding or trusted gateway."""

        if resolved_settings.api_key_tenant_id is not None:
            return resolved_settings.api_key_tenant_id
        if not resolved_settings.trust_proxy_identity:
            return None

        header_name = resolved_settings.trusted_tenant_header
        if header_name is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tenant identity configuration is invalid.",
            )
        supplied_tenant = request.headers.get(header_name)
        if supplied_tenant is None or not supplied_tenant.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted tenant identity is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        normalized_tenant = supplied_tenant.strip()
        if len(normalized_tenant) > 100:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Trusted tenant identity is invalid.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if (
            resolved_settings.serving_tenant_id is not None
            and normalized_tenant != resolved_settings.serving_tenant_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Trusted tenant was routed to the wrong model deployment.",
            )
        return normalized_tenant

    def authorize_pricing_request(
        payload: PricingRecommendationRequest,
        authorized_tenant: str | None,
    ) -> None:
        """Enforce object-level tenant authorization before inference."""

        if authorized_tenant is not None and payload.tenant_id != authorized_tenant:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Caller is not authorized for the requested tenant.",
            )

    def get_container(request: Request) -> ApplicationContainer:
        return cast(ApplicationContainer, request.app.state.container)

    @app.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["operations"],
    )
    async def liveness(
        container_dependency: ApplicationContainer = Depends(get_container),
    ) -> HealthResponse:
        return HealthResponse(
            status="alive",
            model_loaded=container_dependency.model_loaded,
            model_version=container_dependency.model_version,
        )

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
        tags=["operations"],
    )
    async def readiness(
        container_dependency: ApplicationContainer = Depends(get_container),
    ) -> HealthResponse:
        if not container_dependency.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded."
            )
        return HealthResponse(
            status="ready",
            model_loaded=True,
            model_version=container_dependency.model_version,
        )

    @app.get(
        "/metrics",
        response_class=PlainTextResponse,
        include_in_schema=False,
        dependencies=[Depends(authenticate_api_key)],
    )
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        RECOMMENDATION_PATH,
        response_model=PricingRecommendationResponse,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
            status.HTTP_408_REQUEST_TIMEOUT: {"model": ErrorResponse},
            status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
            status.HTTP_504_GATEWAY_TIMEOUT: {"model": ErrorResponse},
        },
        tags=["pricing"],
    )
    async def recommend_price(
        request: Request,
        payload: PricingRecommendationRequest,
        authorized_tenant: str | None = Depends(resolve_authorized_tenant),
        container_dependency: ApplicationContainer = Depends(get_container),
    ) -> PricingRecommendationResponse:
        authorize_pricing_request(payload, authorized_tenant)

        def execute_recommendation() -> PricingRecommendationResponse:
            context = payload.to_domain()
            service = container_dependency.recommendation_service
            recommendation = service.execute(context)
            candidate_count = len(
                PricingPolicy().feasible_prices(
                    current_price=context.current_price,
                    constraints=context.constraints,
                    maximum_candidates=(
                        container_dependency.settings.recommendation_max_candidates
                    ),
                )
            )
            return PricingRecommendationResponse.from_domain(
                recommendation,
                candidate_count=candidate_count,
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + resolved_settings.request_timeout_seconds
        try:
            await asyncio.wait_for(
                recommendation_slots.acquire(),
                timeout=resolved_settings.request_timeout_seconds,
            )
        except TimeoutError as error:
            logger.warning(
                "pricing_recommendation_capacity_timed_out",
                request_id=request.state.request_id,
                tenant_id=payload.tenant_id,
                property_id=payload.property_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pricing recommendation capacity is temporarily exhausted.",
                headers={"Retry-After": "1"},
            ) from error

        async def execute_with_slot() -> PricingRecommendationResponse:
            try:
                return await run_in_threadpool(execute_recommendation)
            finally:
                # The permit follows the real synchronous work, even if the
                # HTTP deadline expires and its response has already returned.
                recommendation_slots.release()

        recommendation_task = asyncio.create_task(execute_with_slot())

        def observe_late_completion(task: asyncio.Task[PricingRecommendationResponse]) -> None:
            try:
                task.result()
            except Exception as error:
                logger.warning(
                    "pricing_recommendation_late_task_failed",
                    failure_type=type(error).__name__,
                )

        remaining_seconds = max(0.0, deadline - loop.time())
        try:
            response = await asyncio.wait_for(
                asyncio.shield(recommendation_task),
                timeout=remaining_seconds,
            )
        except TimeoutError as error:
            recommendation_task.add_done_callback(observe_late_completion)
            logger.warning(
                "pricing_recommendation_timed_out",
                request_id=request.state.request_id,
                tenant_id=payload.tenant_id,
                property_id=payload.property_id,
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Pricing recommendation exceeded the serving deadline.",
            ) from error
        except asyncio.CancelledError:
            recommendation_task.add_done_callback(observe_late_completion)
            raise

        logger.info(
            "pricing_recommendation_issued",
            request_id=request.state.request_id,
            tenant_id=payload.tenant_id,
            property_id=payload.property_id,
            stay_date=payload.stay_date.isoformat(),
            model_version=response.model_version,
            recommended_price=str(response.recommended_price),
            confidence_score=response.confidence_score,
        )
        return response

    return app
