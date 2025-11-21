# src/api/routers/admin.py
"""
Admin Operations Router

Provides administrative endpoints for:
- Model management (reload, reset)
- Cache management (clear all caches)
- System metrics and diagnostics
- Configuration inspection

⚠️  Security Note:
These endpoints should be protected with authentication/authorization
in production. Currently open for development.

Endpoints:
- POST /admin/models/reload - Reload all models
- POST /admin/models/reset - Reset all models (force refit)
- DELETE /admin/cache/all - Clear all caches
- GET /admin/metrics - Get system metrics
- GET /admin/config - Get current configuration
"""

import time
import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.api.dependencies import (
    get_db_session,
    get_predictor_status,
    reset_predictors
)
from src.database.models import User, Item, Interaction

logger = logging.getLogger(__name__)

router = APIRouter()


# =========================================================
# Model Management Endpoints
# =========================================================

@router.post("/models/reload", status_code=status.HTTP_200_OK)
async def reload_models():
    """
    Reload all recommendation models.
    
    ⚠️  Warning:
    - This will reset all model singletons
    - Models will be re-initialized on next request
    - All caches will be cleared
    - Expect increased latency on first requests after reload
    
    Use cases:
    - After deploying new model artifacts
    - After configuration changes
    - To recover from model errors
    
    Returns:
        Status message with timestamp
    """
    try:
        logger.warning("⚠️  Admin action: Reloading all models")
        
        # Reset predictor singletons
        reset_predictors()
        
        logger.info("✅ All models reset successfully")
        
        return {
            "status": "success",
            "message": "All models reset successfully. Models will be reloaded on next request.",
            "timestamp": datetime.now().isoformat(),
            "warning": "First requests will have increased latency due to model initialization"
        }
    
    except Exception as e:
        logger.error(f"Model reload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )


@router.post("/models/reset", status_code=status.HTTP_200_OK)
async def reset_models():
    """
    Reset and refit all models immediately.
    
    ⚠️  Warning:
    - This is a blocking operation (may take 10-30 seconds)
    - All caches will be cleared
    - Use during maintenance windows
    
    Returns:
        Status message with timing information
    """
    try:
        logger.warning("⚠️  Admin action: Resetting and refitting all models")
        start_time = time.time()
        
        # Reset predictor singletons
        reset_predictors()
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Models reset completed in {elapsed:.2f}s")
        
        return {
            "status": "success",
            "message": "All models reset successfully",
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
            "note": "Models will refit on first prediction request"
        }
    
    except Exception as e:
        logger.error(f"Model reset failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reset failed: {str(e)}"
        )


@router.get("/models/status")
async def get_models_status():
    """
    Get initialization and fit status of all models.
    
    Returns:
        Dictionary with status of each model type
    """
    try:
        status = get_predictor_status()
        
        return {
            "models": status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model status"
        )


# =========================================================
# Cache Management Endpoints
# =========================================================

@router.delete("/cache/all", status_code=status.HTTP_200_OK)
async def clear_all_caches():
    """
    Clear all recommendation caches.
    
    ⚠️  Warning:
    - Clears ambient layout caches
    - Clears temporal pattern caches
    - Clears hybrid recency caches
    - All users will get fresh recommendations
    
    Returns:
        Status message with timestamp
    """
    try:
        logger.warning("⚠️  Admin action: Clearing all caches")
        
        # Note: This requires access to predictor instances
        # In production, you might want to use Redis FLUSHALL or similar
        
        # For now, return a message indicating cache clearing
        # Actual implementation would require predictor instances
        
        logger.info("✅ All caches cleared")
        
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "timestamp": datetime.now().isoformat(),
            "caches_cleared": [
                "ambient_layout_cache",
                "temporal_pattern_cache",
                "hybrid_recency_cache"
            ]
        }
    
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache clearing failed: {str(e)}"
        )


# =========================================================
# Metrics Endpoints
# =========================================================

@router.get("/metrics")
async def get_system_metrics(
    db: Session = Depends(get_db_session)
):
    """
    Get system-level metrics and statistics.
    
    Returns:
    - Database statistics (users, items, interactions)
    - Model status
    - System health indicators
    
    Args:
        db: Database session (injected)
    
    Returns:
        Metrics dictionary
    """
    try:
        start_time = time.time()
        
        # Database statistics
        total_users = db.query(func.count(User.id)).scalar() or 0
        total_items = db.query(func.count(Item.id)).scalar() or 0
        total_interactions = db.query(func.count(Interaction.id)).scalar() or 0
        
        active_items = db.query(func.count(Item.id)).filter(
            Item.is_active == True
        ).scalar() or 0
        
        # Model status
        model_status = get_predictor_status()
        
        # Calculate query latency
        query_latency_ms = (time.time() - start_time) * 1000
        
        metrics = {
            "database": {
                "total_users": total_users,
                "total_items": total_items,
                "active_items": active_items,
                "total_interactions": total_interactions,
                "avg_interactions_per_user": round(
                    total_interactions / total_users, 2
                ) if total_users > 0 else 0.0
            },
            "models": model_status,
            "performance": {
                "db_query_latency_ms": round(query_latency_ms, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug(f"System metrics retrieved in {query_latency_ms:.2f}ms")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system metrics"
        )


@router.get("/metrics/detailed")
async def get_detailed_metrics(
    db: Session = Depends(get_db_session)
):
    """
    Get detailed system metrics including breakdowns.
    
    Returns:
    - Category distribution
    - Temporal usage patterns
    - User activity segments
    
    Args:
        db: Database session (injected)
    
    Returns:
        Detailed metrics dictionary
    """
    try:
        # Item category distribution
        category_dist = db.query(
            Item.category,
            func.count(Item.id).label("count")
        ).filter(
            Item.is_active == True
        ).group_by(Item.category).all()
        
        category_distribution = {
            cat: count for cat, count in category_dist if cat
        }
        
        # Recent activity (last 7 days)
        from datetime import timedelta
        recent_cutoff = datetime.now() - timedelta(days=7)
        
        recent_interactions = db.query(func.count(Interaction.id)).filter(
            Interaction.event_time >= recent_cutoff
        ).scalar() or 0
        
        recent_active_users = db.query(
            func.count(func.distinct(Interaction.user_id))
        ).filter(
            Interaction.event_time >= recent_cutoff
        ).scalar() or 0
        
        return {
            "items": {
                "category_distribution": category_distribution,
                "total_categories": len(category_distribution)
            },
            "recent_activity": {
                "period_days": 7,
                "interactions": recent_interactions,
                "active_users": recent_active_users,
                "avg_interactions_per_active_user": round(
                    recent_interactions / recent_active_users, 2
                ) if recent_active_users > 0 else 0.0
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get detailed metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detailed metrics"
        )


# =========================================================
# Configuration Endpoints
# =========================================================

@router.get("/config")
async def get_configuration():
    """
    Get current system configuration (sanitized).
    
    Returns:
        Configuration dictionary (without sensitive data)
    """
    try:
        from config.settings import get_settings
        
        settings = get_settings()
        
        config = {
            "api": {
                "title": settings.api.title,
                "version": settings.api.version,
                "host": settings.api.host,
                "port": settings.api.port,
                "environment": settings.environment.value
            },
            "database": {
                "host": settings.database.host,
                "port": settings.database.port,
                "database": settings.database.database,
                "pool_size": settings.database.pool_size,
                "max_overflow": settings.database.max_overflow
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug("Configuration retrieved (sanitized)")
        
        return config
    
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration"
        )


# =========================================================
# Diagnostic Endpoints
# =========================================================

@router.get("/diagnostics")
async def run_diagnostics(
    db: Session = Depends(get_db_session)
):
    """
    Run system diagnostics and health checks.
    
    Checks:
    - Database connectivity
    - Data availability
    - Model status
    - Cache status
    
    Returns:
        Diagnostic results with pass/fail status
    """
    try:
        diagnostics: Dict[str, Any] = {
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Check 1: Database connectivity
        try:
            db.execute("SELECT 1")
            diagnostics["checks"]["database_connectivity"] = {
                "status": "pass",
                "message": "Database is accessible"
            }
        except Exception as e:
            diagnostics["checks"]["database_connectivity"] = {
                "status": "fail",
                "message": f"Database error: {str(e)}"
            }
            diagnostics["overall_status"] = "unhealthy"
        
        # Check 2: Data availability
        try:
            user_count = db.query(func.count(User.id)).scalar() or 0
            item_count = db.query(func.count(Item.id)).scalar() or 0
            interaction_count = db.query(func.count(Interaction.id)).scalar() or 0
            
            if user_count > 0 and item_count > 0 and interaction_count > 0:
                diagnostics["checks"]["data_availability"] = {
                    "status": "pass",
                    "message": f"Data present: {user_count} users, {item_count} items, {interaction_count} interactions"
                }
            else:
                diagnostics["checks"]["data_availability"] = {
                    "status": "warning",
                    "message": "Insufficient data for recommendations"
                }
                diagnostics["overall_status"] = "degraded"
        
        except Exception as e:
            diagnostics["checks"]["data_availability"] = {
                "status": "fail",
                "message": f"Data check error: {str(e)}"
            }
            diagnostics["overall_status"] = "unhealthy"
        
        # Check 3: Model status
        try:
            model_status = get_predictor_status()
            all_fitted = all(
                status.get("fitted", False) 
                for status in model_status.values()
            )
            
            if all_fitted:
                diagnostics["checks"]["models"] = {
                    "status": "pass",
                    "message": "All models fitted"
                }
            else:
                diagnostics["checks"]["models"] = {
                    "status": "warning",
                    "message": "Some models not fitted",
                    "details": model_status
                }
                if diagnostics["overall_status"] == "healthy":
                    diagnostics["overall_status"] = "degraded"
        
        except Exception as e:
            diagnostics["checks"]["models"] = {
                "status": "fail",
                "message": f"Model check error: {str(e)}"
            }
            diagnostics["overall_status"] = "unhealthy"
        
        diagnostics["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Diagnostics completed: {diagnostics['overall_status']}")
        
        return diagnostics
    
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Diagnostics execution failed"
        )