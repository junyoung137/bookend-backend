# src/api/routers/health.py
"""
Health check endpoints for Kubernetes probes and monitoring.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import time

from src.api.dependencies import get_db_session, get_predictor_status
from src.api.schemas.response import HealthResponse, ComponentHealth, ServiceStatus
from config.settings import get_settings

router = APIRouter()

# Application start time (for uptime calculation)
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db_session)):
    """
    Comprehensive health check for all system components.
    
    Used by:
    - Kubernetes readiness probe
    - Monitoring systems (Prometheus, Datadog)
    - Load balancers
    """
    settings = get_settings()
    
    # Check database
    db_health = _check_database(db)
    
    # Check predictors
    predictor_health = _check_predictors()
    
    # Determine overall status
    overall_status = ServiceStatus.HEALTHY
    
    if db_health.status == ServiceStatus.UNHEALTHY or \
       predictor_health.status == ServiceStatus.UNHEALTHY:
        overall_status = ServiceStatus.UNHEALTHY
    elif db_health.status == ServiceStatus.DEGRADED or \
         predictor_health.status == ServiceStatus.DEGRADED:
        overall_status = ServiceStatus.DEGRADED
    
    return HealthResponse(
        status=overall_status,
        version=settings.api.version,
        timestamp=datetime.now(),
        components={
            "database": db_health,
            "predictors": predictor_health
        },
        uptime_seconds=time.time() - _start_time
    )


@router.get("/health/live", tags=["Health"])
async def liveness_probe():
    """
    Kubernetes liveness probe.
    
    Returns 200 if the application is running.
    Does NOT check dependencies (DB, cache, etc.)
    """
    return {"status": "alive"}


@router.get("/health/ready", tags=["Health"])
async def readiness_probe(db: Session = Depends(get_db_session)):
    """
    Kubernetes readiness probe.
    
    Returns 200 only if:
    - Database is accessible
    - Predictors are initialized
    
    If this fails, K8s will remove pod from service.
    """
    # Quick DB check
    try:
        db.execute("SELECT 1")
        db_ready = True
    except Exception:
        db_ready = False
    
    # Quick predictor check
    predictor_status = get_predictor_status()
    predictors_ready = all(
        status.get("initialized", False) 
        for status in predictor_status.values()
    )
    
    if db_ready and predictors_ready:
        return {"status": "ready"}
    else:
        return {"status": "not_ready"}, 503


def _check_database(db: Session) -> ComponentHealth:
    """Check PostgreSQL connection and latency."""
    try:
        start = time.time()
        db.execute("SELECT 1")
        latency_ms = (time.time() - start) * 1000
        
        if latency_ms > 100:
            return ComponentHealth(
                status=ServiceStatus.DEGRADED,
                message="Database latency high",
                details={"latency_ms": latency_ms}
            )
        
        return ComponentHealth(
            status=ServiceStatus.HEALTHY,
            message="PostgreSQL connected",
            details={"latency_ms": round(latency_ms, 2)}
        )
    
    except Exception as e:
        return ComponentHealth(
            status=ServiceStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}"
        )


def _check_predictors() -> ComponentHealth:
    """Check predictor initialization status."""
    status = get_predictor_status()
    
    all_initialized = all(s.get("initialized", False) for s in status.values())
    all_fitted = all(s.get("fitted", False) for s in status.values())
    
    if all_initialized and all_fitted:
        return ComponentHealth(
            status=ServiceStatus.HEALTHY,
            message="All models loaded and fitted",
            details=status
        )
    elif all_initialized:
        return ComponentHealth(
            status=ServiceStatus.DEGRADED,
            message="Models initialized but not all fitted",
            details=status
        )
    else:
        return ComponentHealth(
            status=ServiceStatus.UNHEALTHY,
            message="Models not initialized",
            details=status
        )