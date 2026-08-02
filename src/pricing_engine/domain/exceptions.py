"""Domain-specific exceptions mapped by inbound adapters."""


class PricingEngineError(Exception):
    """Base exception for expected business and system errors."""


class InvalidPricingRequestError(PricingEngineError):
    """The requested price decision violates a domain invariant."""


class DataContractError(PricingEngineError):
    """A training or inference record violates the data contract."""


class ModelUnavailableError(PricingEngineError):
    """No approved model is available to serve the recommendation."""


class PromotionRejectedError(PricingEngineError):
    """A candidate model did not meet the version-promotion policy."""


class StatisticalContractError(PricingEngineError):
    """A MARKET_EVIDENCE_STATISTICAL_V1 request violates its stable contract."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(detail)
