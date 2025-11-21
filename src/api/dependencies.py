# src/api/dependencies.py
"""
FastAPI Dependency Injection for Bookend Recommendation API

Provides:
- Database session management (per-request lifecycle)
- Predictor instances (singleton pattern)
- Authentication/authorization (future)
- Rate limiting (future)

Principles:
- Dependency Injection: Clean separation of concerns
- Resource Management: Automatic cleanup via generators
- Performance: Singleton predictors, pooled connections
- Error Handling: Graceful degradation on failures
"""

from typing import Generator, Optional, Dict, Any
from functools import lru_cache
import logging

from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session

from config.settings import get_settings
from src.database.postgres_singleton import get_postgres
from src.models.hybrid.ambient_recommender import AmbientRecommender
from src.models.hybrid.temporal_recommender import TemporalRecommender
from src.models.hybrid.hybrid_recommender import HybridRecommender

logger = logging.getLogger(__name__)


# =========================================================
# Database Session Dependency
# =========================================================

def get_db_session() -> Generator[Session, None, None]:
    """
    Provide database session for request lifecycle.
    
    Yields:
        SQLAlchemy Session instance
    
    Usage:
        @app.get("/users/{user_id}")
        def get_user(user_id: int, db: Session = Depends(get_db_session)):
            return db.query(User).filter(User.id == user_id).first()
    
    Note:
        - Session is automatically closed after request
        - Rollback occurs on exception
        - Connection returned to pool after close
    """
    pg = get_postgres()
    
    try:
        with pg.transaction() as session:
            yield session
    except Exception as e:
        logger.error(f"Database session error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Database connection failed"
        )


# =========================================================
# Model Configuration Cache
# =========================================================

@lru_cache()
def get_model_config() -> Dict[str, Any]:
    """
    Get cached model configuration from settings.
    
    Returns:
        Dictionary of model hyperparameters
    
    Note:
        - Cached for application lifetime
        - Can be overridden via config file
    """
    settings = get_settings()
    
    # Load from YAML config file if available
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path(settings.model_config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded model config from {config_path}")
            return config
        
        else:
            logger.warning(
                f"Model config file not found: {config_path}, "
                "using defaults"
            )
    
    except Exception as e:
        logger.error(f"Failed to load model config: {e}", exc_info=True)
    
    # Default configuration
    return {
        "ambient": {
            "layout_positions": 6,
            "refresh_interval_hours": 6,
            "activity_refresh_threshold": 5,
            "diversity_penalty": 0.2,
            "min_category_gap": 2,
            "personalization_strength": 0.7,
            "context_awareness": 0.8,
            "enable_mmr": True,
            "mmr_lambda": 0.5,
            "embedding_dim": 128,
            "explanation_language": "ko",
        },
        "temporal": {
            "temporal_weight": 0.4,
            "recency_weight": 0.3,
            "base_weight": 0.3,
            "lookback_days": 90,
            "recency_decay_hours": 24,
            "peak_hour_boost": 1.3,
            "enable_trend_analysis": True,
            "explanation_language": "ko",
            "max_items_to_score": 1000,
        },
        "hybrid": {
            "user_cf_weight": 0.3,
            "item_cf_weight": 0.3,
            "context_weight": 0.2,
            "recency_weight": 0.2,
            "enable_mmr": True,
            "mmr_lambda": 0.5,
            "recency_cache_ttl": 3600,
        }
    }


# =========================================================
# Predictor Singletons (Lazy Initialization)
# =========================================================

# Module-level caches for singleton instances
_ambient_predictor: Optional[AmbientRecommender] = None
_temporal_predictor: Optional[TemporalRecommender] = None
_hybrid_predictor: Optional[HybridRecommender] = None

# Fitted state flags
_predictors_fitted: Dict[str, bool] = {
    "ambient": False,
    "temporal": False,
    "hybrid": False,
}


def get_ambient_predictor(
    db: Session = Depends(get_db_session)
) -> AmbientRecommender:
    """
    Get or create Ambient Recommender singleton.
    
    Args:
        db: Database session (injected)
    
    Returns:
        Fitted AmbientRecommender instance
    
    Usage:
        @app.post("/ambient/recommend")
        def recommend(predictor: AmbientRecommender = Depends(get_ambient_predictor)):
            return predictor.recommend_for_layout(...)
    
    Note:
        - Singleton pattern: One instance per application
        - Lazy initialization: Created on first request
        - Auto-fitted on creation
    """
    global _ambient_predictor, _predictors_fitted
    
    try:
        # Create singleton if not exists
        if _ambient_predictor is None:
            logger.info("Initializing Ambient Recommender singleton")
            
            config = get_model_config()
            ambient_config = config.get("ambient", {})
            
            _ambient_predictor = AmbientRecommender(db, ambient_config)
            
            logger.info("Ambient Recommender created successfully")
        
        # Fit if not already fitted
        if not _predictors_fitted["ambient"]:
            logger.info("Fitting Ambient Recommender (first request)")
            
            try:
                _ambient_predictor.fit(
                    weighting="count",
                    min_interactions=2,
                    lookback_days=90
                )
                
                _predictors_fitted["ambient"] = True
                logger.info("✅ Ambient Recommender fitted successfully")
            
            except Exception as fit_error:
                logger.warning(
                    f"Ambient fit failed (continuing anyway): {fit_error}"
                )
                # Continue even if fit fails (may work with limited data)
        
        return _ambient_predictor
    
    except Exception as e:
        logger.error(f"Failed to get Ambient Predictor: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Ambient recommendation service unavailable"
        )


def get_temporal_predictor(
    db: Session = Depends(get_db_session)
) -> TemporalRecommender:
    """
    Get or create Temporal Recommender singleton.
    
    Args:
        db: Database session (injected)
    
    Returns:
        Fitted TemporalRecommender instance
    
    Usage:
        @app.post("/temporal/recommend")
        def recommend(predictor: TemporalRecommender = Depends(get_temporal_predictor)):
            return predictor.recommend(...)
    """
    global _temporal_predictor, _predictors_fitted
    
    try:
        # Create singleton if not exists
        if _temporal_predictor is None:
            logger.info("Initializing Temporal Recommender singleton")
            
            config = get_model_config()
            temporal_config = config.get("temporal", {})
            
            _temporal_predictor = TemporalRecommender(db, temporal_config)
            
            logger.info("Temporal Recommender created successfully")
        
        # Fit if not already fitted
        if not _predictors_fitted["temporal"]:
            logger.info("Fitting Temporal Recommender (first request)")
            
            try:
                _temporal_predictor.fit(
                    weighting="count",
                    min_interactions=2,
                    lookback_days=90
                )
                
                _predictors_fitted["temporal"] = True
                logger.info("✅ Temporal Recommender fitted successfully")
            
            except Exception as fit_error:
                logger.warning(
                    f"Temporal fit failed (continuing anyway): {fit_error}"
                )
        
        return _temporal_predictor
    
    except Exception as e:
        logger.error(f"Failed to get Temporal Predictor: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Temporal recommendation service unavailable"
        )


def get_hybrid_predictor(
    db: Session = Depends(get_db_session)
) -> HybridRecommender:
    """
    Get or create Hybrid Recommender singleton.
    
    Args:
        db: Database session (injected)
    
    Returns:
        Fitted HybridRecommender instance
    
    Usage:
        @app.post("/hybrid/recommend")
        def recommend(predictor: HybridRecommender = Depends(get_hybrid_predictor)):
            return predictor.recommend(...)
    """
    global _hybrid_predictor, _predictors_fitted
    
    try:
        # Create singleton if not exists
        if _hybrid_predictor is None:
            logger.info("Initializing Hybrid Recommender singleton")
            
            config = get_model_config()
            hybrid_config = config.get("hybrid", {})
            
            _hybrid_predictor = HybridRecommender(db, hybrid_config)
            
            logger.info("Hybrid Recommender created successfully")
        
        # Fit if not already fitted
        if not _predictors_fitted["hybrid"]:
            logger.info("Fitting Hybrid Recommender (first request)")
            
            try:
                _hybrid_predictor.fit(
                    weighting="count",
                    min_interactions=2,
                    lookback_days=90
                )
                
                _predictors_fitted["hybrid"] = True
                logger.info("✅ Hybrid Recommender fitted successfully")
            
            except Exception as fit_error:
                logger.warning(
                    f"Hybrid fit failed (continuing anyway): {fit_error}"
                )
        
        return _hybrid_predictor
    
    except Exception as e:
        logger.error(f"Failed to get Hybrid Predictor: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Hybrid recommendation service unavailable"
        )


# =========================================================
# Authentication Dependencies (Future Implementation)
# =========================================================

async def get_current_user(
    authorization: Optional[str] = Header(None)
) -> Optional[Dict[str, Any]]:
    """
    Extract and validate current user from JWT token.
    
    Args:
        authorization: Authorization header (Bearer token)
    
    Returns:
        User information dictionary or None
    
    Usage:
        @app.get("/me")
        def get_me(user: dict = Depends(get_current_user)):
            return user
    
    Note:
        - Currently returns None (authentication disabled)
        - TODO: Implement JWT validation
        - TODO: Add user role/permissions
    """
    # TODO: Implement JWT token validation
    # For now, return None (no authentication)
    
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        logger.debug(f"Received token: {token[:10]}...")
        
        # TODO: Decode and validate JWT
        # TODO: Extract user_id, role, permissions
        
        return None  # Placeholder
    
    return None


async def require_authentication(
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require authenticated user (raise 401 if not authenticated).
    
    Args:
        user: Current user (injected)
    
    Returns:
        User information dictionary
    
    Raises:
        HTTPException: 401 if not authenticated
    
    Usage:
        @app.get("/protected")
        def protected_route(user: dict = Depends(require_authentication)):
            return {"message": f"Hello {user['name']}"}
    """
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def require_admin(
    user: Dict[str, Any] = Depends(require_authentication)
) -> Dict[str, Any]:
    """
    Require admin role (raise 403 if not admin).
    
    Args:
        user: Current user (injected)
    
    Returns:
        User information dictionary
    
    Raises:
        HTTPException: 403 if not admin
    
    Usage:
        @app.post("/admin/deploy")
        def deploy_model(user: dict = Depends(require_admin)):
            return {"message": "Model deployed"}
    """
    # TODO: Implement role checking
    # For now, accept any authenticated user
    
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    return user


# =========================================================
# Utility Functions
# =========================================================

def reset_predictors() -> None:
    """
    Reset all predictor singletons (for testing/reloading).
    
    Warning:
        - This will force re-initialization on next request
        - Use carefully (e.g., after model update)
    """
    global _ambient_predictor, _temporal_predictor, _hybrid_predictor
    global _predictors_fitted
    
    logger.warning("Resetting all predictor singletons")
    
    _ambient_predictor = None
    _temporal_predictor = None
    _hybrid_predictor = None
    
    _predictors_fitted = {
        "ambient": False,
        "temporal": False,
        "hybrid": False,
    }
    
    logger.info("✅ Predictors reset complete")


def get_predictor_status() -> Dict[str, Any]:
    """
    Get status of all predictors (for health checks).
    
    Returns:
        Dictionary with predictor initialization/fit status
    """
    return {
        "ambient": {
            "initialized": _ambient_predictor is not None,
            "fitted": _predictors_fitted.get("ambient", False)
        },
        "temporal": {
            "initialized": _temporal_predictor is not None,
            "fitted": _predictors_fitted.get("temporal", False)
        },
        "hybrid": {
            "initialized": _hybrid_predictor is not None,
            "fitted": _predictors_fitted.get("hybrid", False)
        }
    }


# =========================================================
# Testing Utilities (Development Only)
# =========================================================

if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "="*70)
    print("🔧 DEPENDENCIES MODULE TEST")
    print("="*70)
    
    # Test configuration loading
    print("\n📋 Model Configuration:")
    config = get_model_config()
    import json
    print(json.dumps(config, indent=2))
    
    # Test predictor status
    print("\n📊 Predictor Status:")
    status = get_predictor_status()
    print(json.dumps(status, indent=2))
    
    print("\n✅ Dependencies module test complete")