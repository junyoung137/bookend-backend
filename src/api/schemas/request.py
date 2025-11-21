
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


# =========================================================
# Enums for Validation
# =========================================================

class SlotType(str, Enum):
    """Layout slot types for ambient recommendations."""
    HERO_BANNER = "hero_banner"
    SIDEBAR_QUICK = "sidebar_quick"
    FOOTER_SUGGESTION = "footer_suggestion"
    DASHBOARD_WIDGET = "dashboard_widget"


class LayoutType(str, Enum):
    """Layout types for UI rendering."""
    STANDARD = "standard"
    CAROUSEL = "carousel"
    GRID = "grid"
    LIST = "list"


class DeviceType(str, Enum):
    """Device types for context."""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    UNKNOWN = "unknown"


# =========================================================
# Context Data Schema (Shared)
# =========================================================

class ContextData(BaseModel):
    """
    Contextual information for recommendations.
    
    Used across all recommendation types to provide
    time, device, and location context.
    """
    
    # Temporal context
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Request timestamp (defaults to current time if not provided)"
    )
    
    # Device context
    browser: Optional[str] = Field(
        default=None,
        description="Browser name (e.g., Chrome, Firefox)",
        examples=["Chrome", "Safari", "Firefox"]
    )
    
    browser_version: Optional[str] = Field(
        default=None,
        description="Browser version",
        examples=["140", "17.2"]
    )
    
    os: Optional[str] = Field(
        default=None,
        description="Operating system",
        examples=["Windows", "macOS", "iOS", "Android"]
    )
    
    device_type: Optional[DeviceType] = Field(
        default=None,
        description="Device type classification"
    )
    
    device_id: Optional[str] = Field(
        default=None,
        description="Unique device identifier"
    )
    
    # Location context
    city: Optional[str] = Field(
        default=None,
        description="City name"
    )
    
    country_code: Optional[str] = Field(
        default=None,
        description="ISO country code (e.g., KR, US)",
        examples=["KR", "US", "JP"]
    )
    
    timezone: Optional[str] = Field(
        default=None,
        description="Timezone identifier",
        examples=["Asia/Seoul", "America/New_York"]
    )
    
    # Session context
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for tracking"
    )
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-15T10:30:00Z",
                "browser": "Chrome",
                "browser_version": "140",
                "os": "Windows",
                "device_type": "desktop",
                "city": "Seoul",
                "country_code": "KR",
                "timezone": "Asia/Seoul"
            }
        }
    )


# =========================================================
# Ambient Recommendation Request
# =========================================================

class AmbientRecommendRequest(BaseModel):
    """
    Request schema for ambient (layout-aware) recommendations.
    
    Supports different layout types and slot positions
    with context-aware scoring.
    """
    
    user_id: int = Field(
        ...,
        description="User database ID (required)",
        gt=0,
        examples=[123, 456]
    )
    
    context: Optional[ContextData] = Field(
        default=None,
        description="Contextual information for recommendation"
    )
    
    layout_type: LayoutType = Field(
        default=LayoutType.STANDARD,
        description="UI layout type"
    )
    
    slot_type: SlotType = Field(
        default=SlotType.HERO_BANNER,
        description="Layout slot type for weight adjustment"
    )
    
    top_k: Optional[int] = Field(
        default=6,
        description="Number of recommendations to return",
        ge=1,
        le=20
    )
    
    force_refresh: bool = Field(
        default=False,
        description="Force cache refresh (bypass TTL)"
    )
    
    exclude_items: Optional[List[int]] = Field(
        default=None,
        description="Item IDs to exclude from results"
    )
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        """Validate user_id is positive."""
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 123,
                "context": {
                    "timestamp": "2025-01-15T10:30:00Z",
                    "browser": "Chrome",
                    "os": "Windows",
                    "device_type": "desktop",
                    "country_code": "KR"
                },
                "layout_type": "standard",
                "slot_type": "hero_banner",
                "top_k": 6,
                "force_refresh": False
            }
        }
    )


# =========================================================
# Temporal Recommendation Request
# =========================================================

class TemporalRecommendRequest(BaseModel):
    """
    Request schema for temporal (time-pattern based) recommendations.
    
    Emphasizes temporal context for time-aware scoring.
    """
    
    user_id: int = Field(
        ...,
        description="User database ID (required)",
        gt=0,
        examples=[123, 456]
    )
    
    context: Optional[ContextData] = Field(
        default=None,
        description="Contextual information (timestamp is critical)"
    )
    
    limit: int = Field(
        default=10,
        description="Maximum number of recommendations",
        ge=1,
        le=50
    )
    
    min_score: float = Field(
        default=0.0,
        description="Minimum score threshold (0.0 - 1.0)",
        ge=0.0,
        le=1.0
    )
    
    exclude_items: Optional[List[int]] = Field(
        default=None,
        description="Item IDs to exclude from results"
    )
    
    include_reasons: bool = Field(
        default=True,
        description="Include explanation reasons in response"
    )
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        """Validate user_id is positive."""
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 123,
                "context": {
                    "timestamp": "2025-01-15T14:30:00Z",
                    "browser": "Chrome",
                    "os": "macOS",
                    "timezone": "Asia/Seoul"
                },
                "limit": 10,
                "min_score": 0.3,
                "include_reasons": True
            }
        }
    )


# =========================================================
# Hybrid Recommendation Request
# =========================================================

class HybridRecommendRequest(BaseModel):
    """
    Request schema for hybrid (multi-strategy) recommendations.
    
    Combines User-CF, Item-CF, context, and recency scoring.
    """
    
    user_id: int = Field(
        ...,
        description="User database ID (required)",
        gt=0,
        examples=[123, 456]
    )
    
    context: Optional[ContextData] = Field(
        default=None,
        description="Contextual information for context-aware scoring"
    )
    
    limit: int = Field(
        default=10,
        description="Maximum number of recommendations",
        ge=1,
        le=50
    )
    
    min_score: float = Field(
        default=0.0,
        description="Minimum score threshold (0.0 - 1.0)",
        ge=0.0,
        le=1.0
    )
    
    exclude_items: Optional[List[int]] = Field(
        default=None,
        description="Item IDs to exclude from results"
    )
    
    include_reasons: bool = Field(
        default=True,
        description="Include explanation reasons in response"
    )
    
    enable_diversity: bool = Field(
        default=True,
        description="Enable MMR-based diversity re-ranking"
    )
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        """Validate user_id is positive."""
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 123,
                "context": {
                    "timestamp": "2025-01-15T10:30:00Z",
                    "browser": "Chrome",
                    "os": "Windows"
                },
                "limit": 10,
                "min_score": 0.2,
                "include_reasons": True,
                "enable_diversity": True
            }
        }
    )


# =========================================================
# Feature Extraction Request (Future)
# =========================================================

class FeatureExtractionRequest(BaseModel):
    """
    Request schema for on-demand feature extraction.
    
    Used for real-time feature computation.
    """
    
    user_id: int = Field(
        ...,
        description="User database ID",
        gt=0
    )
    
    feature_types: List[str] = Field(
        default=["temporal", "behavioral", "contextual"],
        description="Types of features to extract"
    )
    
    lookback_days: int = Field(
        default=90,
        description="Days to look back for feature computation",
        ge=1,
        le=365
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 123,
                "feature_types": ["temporal", "behavioral"],
                "lookback_days": 90
            }
        }
    ) 