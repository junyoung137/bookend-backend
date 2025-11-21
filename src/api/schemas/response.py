# src/api/schemas/response.py
"""
API Response Schemas for Bookend Recommendation System

Includes:
- Recommendation responses
- Health check responses
- Error responses
- Batch & User feature responses
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


# =========================================================
# Recommendation Item
# =========================================================
class RecommendationItem(BaseModel):
    item_id: int = Field(..., description="Item database ID", examples=[1, 2, 3])
    item_code: str = Field(..., description="Item code identifier", examples=["paraphrase_formal", "grammar_check"])
    item_name: str = Field(..., description="Item display name", examples=["Formal Paraphrasing", "Grammar Checker"])
    score: float = Field(..., description="Recommendation score (0.0 - 1.0)", ge=0.0, le=1.0, examples=[0.87, 0.65])
    rank: int = Field(..., description="Rank in recommendation list (1-indexed)", ge=1, examples=[1, 2, 3])
    reason: str = Field(..., description="Human-readable explanation for recommendation",
                        examples=["지금 이 순간에 잘 맞아요", "최근 관심 보인 항목", "Similar to items you've used"])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional item metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this recommendation was generated")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "item_id": 5,
                "item_code": "paraphrase_formal",
                "item_name": "Formal Paraphrasing",
                "score": 0.87,
                "rank": 1,
                "reason": "지금 이 순간에 잘 맞아요",
                "metadata": {"category": "paraphrasing", "is_premium": False, "layout_optimized": True},
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }
    )


# =========================================================
# Recommendation Response
# =========================================================
class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")
    user_id: int = Field(..., description="User ID for which recommendations were generated")
    total_count: int = Field(..., description="Total number of recommendations returned", ge=0)
    model_name: str = Field(..., description="Name of recommender model used",
                            examples=["ambient_recommender", "temporal_recommender", "hybrid_recommender"])
    request_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the request (context, params)")
    performance_metrics: Optional[Dict[str, float]] = Field(default=None, description="Optional performance metrics (latency, scores)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response generation timestamp")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "recommendations": [
                    {
                        "item_id": 5,
                        "item_code": "paraphrase_formal",
                        "item_name": "Formal Paraphrasing",
                        "score": 0.87,
                        "rank": 1,
                        "reason": "지금 이 순간에 잘 맞아요",
                        "metadata": {"category": "paraphrasing"}
                    }
                ],
                "user_id": 123,
                "total_count": 1,
                "model_name": "ambient_recommender",
                "request_metadata": {"layout_type": "standard", "slot_type": "hero_banner"},
                "performance_metrics": {"latency_ms": 45.2, "avg_score": 0.72},
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }
    )


# =========================================================
# Error Response
# =========================================================
class ErrorResponse(BaseModel):
    error: Dict[str, Any] = Field(..., description="Error information")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "error": {
                    "code": 404,
                    "message": "User not found",
                    "detail": "User with ID 999 does not exist"
                }
            }
        }
    )


# =========================================================
# Health Check Response
# =========================================================
class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    status: ServiceStatus = Field(..., description="Component health status")
    message: Optional[str] = Field(default=None, description="Optional status message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional component details")


class HealthResponse(BaseModel):
    status: ServiceStatus = Field(..., description="Overall service health status")
    version: str = Field(..., description="API version", examples=["0.1.0"])
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    components: Dict[str, ComponentHealth] = Field(default_factory=dict, description="Health status of individual components")
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime in seconds")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2025-01-15T10:30:00Z",
                "components": {
                    "database": {"status": "healthy", "message": "PostgreSQL connected", "details": {"latency_ms": 2.3}},
                    "predictors": {"status": "healthy", "message": "All models loaded",
                                   "details": {"ambient": "fitted", "temporal": "fitted", "hybrid": "fitted"}}
                },
                "uptime_seconds": 3600.5
            }
        }
    )


# =========================================================
# User Feature Response
# =========================================================
class UserFeatureResponse(BaseModel):
    user_id: int = Field(..., description="User database ID")
    features: Dict[str, Any] = Field(..., description="Extracted feature dictionary")
    feature_types: List[str] = Field(..., description="Types of features included")
    computed_at: datetime = Field(default_factory=datetime.now, description="Feature computation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (lookback period, etc.)")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "user_id": 123,
                "features": {"total_paraphrases": 450, "last_7d_count": 23, "preferred_tone": "formal", "peak_hour": 14,
                             "engagement_score": 0.78},
                "feature_types": ["temporal", "behavioral"],
                "computed_at": "2025-01-15T10:30:00Z",
                "metadata": {"lookback_days": 90, "interaction_count": 450}
            }
        }
    )


# =========================================================
# Batch Recommendation Response
# =========================================================
class BatchRecommendationResponse(BaseModel):
    results: Dict[int, RecommendationResponse] = Field(..., description="Recommendations per user_id")
    total_users: int = Field(..., description="Total number of users processed", ge=0)
    success_count: int = Field(..., description="Number of successful recommendations", ge=0)
    failure_count: int = Field(..., description="Number of failed recommendations", ge=0)
    processing_time_seconds: float = Field(..., description="Total processing time", ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch processing timestamp")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "results": {123: {"recommendations": [], "user_id": 123, "total_count": 5, "model_name": "ambient_recommender"}},
                "total_users": 10,
                "success_count": 9,
                "failure_count": 1,
                "processing_time_seconds": 2.45,
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }
    )