"""FastAPI application factory and operational endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from secrets import compare_digest
from time import perf_counter
from typing import cast
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from pricing_engine.config import Settings, get_settings
from pricing_engine.domain.exceptions import (
    ModelUnavailableError,
    PricingEngineError,
)
from pricing_engine.domain.policies import PricingPolicy
from pricing_engine.interfaces.api.container import ApplicationContainer
from pricing_engine.interfaces.api.schemas import (
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

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        resolved_container.load_configured_model()
        yield

    app = FastAPI(
        title=resolved_settings.api_title,
        version=resolved_settings.api_version,
        lifespan=lifespan,
        docs_url="/docs" if resolved_settings.environment != "production" else None,
        redoc_url=None,
    )
    app.state.container = resolved_container
    allowed_origins = [str(origin).rstrip("/") for origin in resolved_settings.cors_origins]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )

    @app.middleware("http")
    async def telemetry(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        started = perf_counter()
        response: Response = await call_next(request)
        elapsed = perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        REQUEST_COUNTER.labels(
            path=request.url.path,
            method=request.method,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(path=request.url.path).observe(elapsed)
        logger.info(
            "http_request_completed",
            path=request.url.path,
            method=request.method,
            status=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
            request_id=request_id,
        )
        return response

    @app.exception_handler(PricingEngineError)
    async def pricing_error_handler(_: Request, error: PricingEngineError) -> JSONResponse:
        response_status = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if isinstance(error, ModelUnavailableError)
            else status.HTTP_422_UNPROCESSABLE_CONTENT
        )
        return JSONResponse(
            status_code=response_status,
            content={"error": error.__class__.__name__, "detail": str(error)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "RequestValidationError",
                "detail": jsonable_encoder(error.errors()),
            },
        )

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        configured = resolved_settings.api_key
        if configured is None:
            return
        if x_api_key is None or not compare_digest(x_api_key, configured.get_secret_value()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    def get_container(request: Request) -> ApplicationContainer:
        return cast(ApplicationContainer, request.app.state.container)

    @app.get("/health/live", response_model=HealthResponse, tags=["operations"])
    async def liveness(
        container_dependency: ApplicationContainer = Depends(get_container),
    ) -> HealthResponse:
        return HealthResponse(
            status="alive",
            model_loaded=container_dependency.model_loaded,
            model_version=container_dependency.model_version,
        )

    @app.get("/health/ready", response_model=HealthResponse, tags=["operations"])
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

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/pricing/recommendations",
        response_model=PricingRecommendationResponse,
        tags=["pricing"],
        dependencies=[Depends(require_api_key)],
    )
    async def recommend_price(
        payload: PricingRecommendationRequest,
        container_dependency: ApplicationContainer = Depends(get_container),
    ) -> PricingRecommendationResponse:
        context = payload.to_domain()
        service = container_dependency.recommendation_service
        recommendation = service.execute(context)
        candidate_count = len(
            PricingPolicy().feasible_prices(
                current_price=context.current_price,
                constraints=context.constraints,
                maximum_candidates=container_dependency.settings.recommendation_max_candidates,
            )
        )
        return PricingRecommendationResponse.from_domain(
            recommendation,
            candidate_count=candidate_count,
        )

    return app
