from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"

class HealthResponse(BaseModel):
    status: ServiceStatus = Field(default=ServiceStatus.HEALTHY)
    uptime: float = Field(..., description="Seconds since service started")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
