# src/api/schemas.py
"""
API Schemas for Bookend Recommendation System

✅ 개선사항:
- RecommendationItem에 content/type 필드 추가
- Validation 강화
- 타입 안정성 개선
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# =========================================================
# Enums
# =========================================================

class RecommendationType(str, Enum):
    """추천 유형"""
    PARAPHRASE = "paraphrase"
    TONE = "tone"
    EXPAND = "expand"


# =========================================================
# Context Data Schema
# =========================================================

class ContextData(BaseModel):
    """추천 컨텍스트 정보"""
    timestamp: Optional[datetime] = Field(default=None, description="요청 시각")
    device_type: Optional[str] = Field(default=None, description="디바이스 타입")
    location: Optional[str] = Field(default=None, description="위치 정보")
    session_id: Optional[str] = Field(default=None, description="세션 ID")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="기타 메타데이터")


# =========================================================
# Request Schemas
# =========================================================

class RecommendRequest(BaseModel):
    """기본 추천 요청"""
    user_id: int = Field(..., description="사용자 ID", gt=0)
    limit: int = Field(default=10, description="추천 개수", ge=1, le=50)
    exclude_interacted: Optional[bool] = Field(default=True)
    context: Optional[ContextData] = Field(default=None)
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v


class HybridRecommendRequest(BaseModel):
    """Hybrid 추천 요청"""
    user_id: int = Field(..., description="사용자 ID", gt=0)
    limit: int = Field(default=10, ge=1, le=50)
    exclude_interacted: Optional[bool] = Field(default=True)
    context: Optional[Dict[str, Any]] = Field(default=None)
    min_score: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    include_reasons: Optional[bool] = Field(default=True)
    enable_diversity: Optional[bool] = Field(default=True)
    exclude_items: Optional[List[int]] = Field(default=None)


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
    """
    ✅ 개별 추천 아이템 (content/type 필드 추가)
    """
    item_id: int = Field(..., description="아이템 ID")
    score: float = Field(..., description="추천 점수", ge=0.0, le=1.0)
    rank: int = Field(..., description="순위", ge=1)
    
    # ✅ 새로 추가된 필드
    content: Optional[str] = Field(
        default=None,
        description="추천 문장 텍스트",
        examples=["이 문장을 더 간결하게 표현해보세요."]
    )
    type: Optional[str] = Field(
        default=None,
        description="추천 유형",
        examples=["paraphrase", "tone", "expand"]
    )
    
    reasons: List[str] = Field(default_factory=list, description="추천 이유")
    item_name: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    
    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ["paraphrase", "tone", "expand"]:
            raise ValueError("type must be one of: paraphrase, tone, expand")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": 5,
                "score": 0.87,
                "rank": 1,
                "content": "이 문장을 더 간결하고 명확하게 표현해보세요.",
                "type": "paraphrase",
                "reasons": ["similar_users", "popular"]
            }
        }


class RecommendResponse(BaseModel):
    """추천 응답"""
    user_id: int = Field(..., description="사용자 ID")
    recommendations: List[RecommendationItem] = Field(..., description="추천 리스트")
    total_count: int = Field(..., description="추천 개수", ge=0)
    model_name: str = Field(default="Hybrid v2 Rebalanced")
    is_cold_start: bool = Field(default=False)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    request_metadata: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchRecommendResponse(BaseModel):
    """배치 추천 응답"""
    results: Dict[int, RecommendResponse]
    total_users: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    failure_count: int = Field(..., ge=0)
    processing_time_seconds: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)


# =========================================================
# Health & Info Schemas
# =========================================================

class ComponentHealth(BaseModel):
    """컴포넌트 헬스"""
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    model_loaded: bool
    components: Optional[Dict[str, ComponentHealth]] = None
    model_info: Optional[Dict[str, Any]] = None
    uptime_seconds: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfoResponse(BaseModel):
    """모델 정보 응답"""
    model_type: str
    model_name: str
    num_users: Optional[int] = None
    num_items: Optional[int] = None
    weights: Optional[Dict[str, float]] = None
    config: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """에러 응답"""
    error: Dict[str, Any]