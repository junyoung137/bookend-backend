# src/api/schemas/__init__.py
"""
API Schemas package for Bookend Recommendation System.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# =========================================================
# Context Data Schema (선택적 컨텍스트 정보)
# =========================================================

class ContextData(BaseModel):
    """
    추천 컨텍스트 정보 (선택사항)
    """
    timestamp: Optional[datetime] = Field(
        default=None,
        description="요청 시각"
    )
    
    device_type: Optional[str] = Field(
        default=None,
        description="디바이스 타입"
    )
    
    location: Optional[str] = Field(
        default=None,
        description="위치 정보"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="세션 ID"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="기타 메타데이터"
    )


# =========================================================
# Request Schemas
# =========================================================

class RecommendRequest(BaseModel):
    """기본 추천 요청"""
    user_id: int = Field(..., gt=0)
    limit: int = Field(default=10, ge=1, le=50)
    exclude_interacted: Optional[bool] = Field(default=True)
    context: Optional[ContextData] = Field(default=None)
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v


class HybridRecommendRequest(BaseModel):
    """Hybrid 추천 요청 (고급 옵션)"""
    user_id: int = Field(..., gt=0)
    limit: int = Field(default=10, ge=1, le=50)
    exclude_interacted: Optional[bool] = Field(default=True)
    context: Optional[Dict[str, Any]] = Field(default=None)
    min_score: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    include_reasons: Optional[bool] = Field(default=True)
    enable_diversity: Optional[bool] = Field(default=True)
    exclude_items: Optional[List[int]] = Field(default=None)
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v


class BatchRecommendRequest(BaseModel):
    """배치 추천 요청"""
    user_ids: List[int] = Field(..., min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=50)
    exclude_interacted: Optional[bool] = Field(default=True)
    
    @field_validator("user_ids")
    @classmethod
    def validate_user_ids(cls, v: List[int]) -> List[int]:
        if any(uid <= 0 for uid in v):
            raise ValueError("All user_ids must be positive")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate user_ids found")
        return v


# =========================================================
# Response Schemas
# =========================================================

class RecommendationItem(BaseModel):
    """개별 추천 아이템"""
    item_id: int = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)
    reasons: List[str] = Field(default_factory=list)
    item_name: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class RecommendResponse(BaseModel):
    """추천 응답"""
    user_id: int = Field(...)
    recommendations: List[RecommendationItem] = Field(...)
    total_count: int = Field(..., ge=0)
    model_name: str = Field(default="Hybrid v2 Rebalanced")
    is_cold_start: bool = Field(default=False)
    latency_ms: Optional[float] = Field(default=None)
    request_metadata: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchRecommendResponse(BaseModel):
    """배치 추천 응답"""
    results: Dict[int, RecommendResponse] = Field(...)
    total_users: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    failure_count: int = Field(..., ge=0)
    processing_time_seconds: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)


# =========================================================
# Error Response
# =========================================================

class ErrorDetail(BaseModel):
    """개별 에러 상세"""
    field: Optional[str] = Field(default=None)
    message: str = Field(...)
    error_type: Optional[str] = Field(default=None)


class ErrorResponse(BaseModel):
    """에러 응답"""
    error: Dict[str, Any] = Field(...)


# =========================================================
# Health Check Response
# =========================================================

class ComponentHealth(BaseModel):
    """개별 컴포넌트 헬스"""
    status: str = Field(...)
    message: Optional[str] = Field(default=None)
    details: Optional[Dict[str, Any]] = Field(default=None)


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(...)
    model_loaded: bool = Field(...)
    components: Optional[Dict[str, ComponentHealth]] = Field(default=None)
    model_info: Optional[Dict[str, Any]] = Field(default=None)
    uptime_seconds: Optional[float] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)


# =========================================================
# Model Info Response
# =========================================================

class ModelInfoResponse(BaseModel):
    """모델 정보 응답"""
    model_type: str = Field(...)
    model_name: str = Field(...)
    num_users: Optional[int] = Field(default=None)
    num_items: Optional[int] = Field(default=None)
    weights: Optional[Dict[str, float]] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)


# =========================================================
# Exports
# =========================================================

__all__ = [
    # Context
    "ContextData",
    
    # Request schemas
    "RecommendRequest",
    "HybridRecommendRequest",
    "BatchRecommendRequest",
    
    # Response schemas
    "RecommendationItem",
    "RecommendResponse",
    "BatchRecommendResponse",
    "ErrorDetail",
    "ErrorResponse",
    "ComponentHealth",
    "HealthResponse",
    "ModelInfoResponse",
]