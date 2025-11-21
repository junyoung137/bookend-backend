# src/api/routers/ambient.py
"""
Ambient Recommendations Router

Provides layout-aware, context-sensitive recommendations
optimized for different UI slots and device contexts.

Endpoints:
- POST /ambient/recommend - Generate ambient layout recommendations
- POST /ambient/register-click - Register user interaction for cache refresh
- DELETE /ambient/cache/{user_id} - Clear user's ambient cache
"""

import time
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session, get_ambient_predictor
from src.api.schemas.request import AmbientRecommendRequest
from src.api.schemas.response import RecommendationResponse, RecommendationItem
from src.api.schemas.errors import NotFoundError, ServiceUnavailableError
from src.models.hybrid.ambient_recommender import AmbientRecommender
from src.database.models import User

logger = logging.getLogger(__name__)

router = APIRouter()


# =========================================================
# Main Recommendation Endpoint
# =========================================================

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_ambient(
    request: AmbientRecommendRequest,
    db: Session = Depends(get_db_session),
    predictor: AmbientRecommender = Depends(get_ambient_predictor)
):
    """
    Generate ambient (layout-aware) recommendations.
    
    Features:
    - Layout-specific scoring (hero banner, sidebar, footer)
    - Context-aware ranking (time, device, location)
    - Activity-based cache refresh
    - MMR-based diversity
    
    Args:
        request: Ambient recommendation request
        db: Database session (injected)
        predictor: Ambient recommender instance (injected)
    
    Returns:
        RecommendationResponse with layout-optimized items
    
    Raises:
        404: User not found
        503: Recommendation service unavailable
    """
    start_time = time.time()
    
    try:
        # Validate user exists
        user = db.get(User, request.user_id)
        if not user:
            logger.warning(f"User not found: {request.user_id}")
            raise NotFoundError(
                resource="User",
                resource_id=request.user_id
            )
        
        # Prepare context dictionary
        context_dict = None
        if request.context:
            context_dict = request.context.model_dump()
        
        # Generate recommendations
        logger.info(
            f"Generating ambient recommendations for user {request.user_id} "
            f"(slot_type={request.slot_type}, layout_type={request.layout_type})"
        )
        
        recommendations = predictor.recommend_for_layout(
            user_id=request.user_id,
            context=context_dict,
            layout_type=request.layout_type,
            slot_type=request.slot_type,
            top_k=request.top_k,
            force_refresh=request.force_refresh
        )
        
        # Filter excluded items if specified
        if request.exclude_items:
            recommendations = [
                rec for rec in recommendations 
                if rec.item_id not in request.exclude_items
            ]
        
        # Convert to response items
        response_items = [
            RecommendationItem(
                item_id=rec.item_id,
                item_code=rec.item_code,
                item_name=rec.item_name,
                score=rec.score,
                rank=rec.rank,
                reason=rec.reason,
                metadata=rec.metadata,
                timestamp=datetime.now()
            )
            for rec in recommendations
        ]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Ambient recommendations generated: {len(response_items)} items "
            f"in {latency_ms:.2f}ms"
        )
        
        # Build response
        return RecommendationResponse(
            recommendations=response_items,
            user_id=request.user_id,
            total_count=len(response_items),
            model_name=predictor.get_model_name(),
            request_metadata={
                "layout_type": request.layout_type,
                "slot_type": request.slot_type,
                "force_refresh": request.force_refresh,
                "context_provided": context_dict is not None
            },
            performance_metrics={
                "latency_ms": round(latency_ms, 2),
                "avg_score": round(
                    sum(item.score for item in response_items) / len(response_items), 4
                ) if response_items else 0.0
            },
            timestamp=datetime.now()
        )
    
    except NotFoundError:
        raise
    
    except Exception as e:
        logger.error(
            f"Ambient recommendation failed for user {request.user_id}: {e}",
            exc_info=True
        )
        raise ServiceUnavailableError(
            service="Ambient Recommender",
            message=f"Failed to generate recommendations: {str(e)}"
        )


# =========================================================
# Activity Tracking Endpoint
# =========================================================

@router.post("/register-click/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def register_click(
    user_id: int,
    db: Session = Depends(get_db_session),
    predictor: AmbientRecommender = Depends(get_ambient_predictor)
):
    """
    Register user interaction for activity-based cache refresh.
    
    When a user clicks on a recommended item, call this endpoint
    to increment their interaction counter. Once the threshold is reached,
    the cache will be automatically refreshed on the next recommendation request.
    
    Args:
        user_id: User database ID
        db: Database session (injected)
        predictor: Ambient recommender instance (injected)
    
    Returns:
        204 No Content on success
    
    Raises:
        404: User not found
    """
    try:
        # Validate user exists
        user = db.get(User, user_id)
        if not user:
            logger.warning(f"User not found: {user_id}")
            raise NotFoundError(resource="User", resource_id=user_id)
        
        # Register click
        predictor.register_click(user_id)
        
        logger.debug(f"Registered click for user {user_id}")
        
        return None  # 204 No Content
    
    except NotFoundError:
        raise
    
    except Exception as e:
        logger.error(f"Failed to register click for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register interaction"
        )


# =========================================================
# Cache Management Endpoint
# =========================================================

@router.delete("/cache/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_cache(
    user_id: int,
    predictor: AmbientRecommender = Depends(get_ambient_predictor)
):
    """
    Clear ambient recommendation cache for a specific user.
    
    Use this endpoint to force a fresh recommendation generation
    on the next request (alternative to force_refresh parameter).
    
    Args:
        user_id: User database ID
        predictor: Ambient recommender instance (injected)
    
    Returns:
        204 No Content on success
    """
    try:
        predictor.clear_cache(user_id)
        logger.info(f"Cleared ambient cache for user {user_id}")
        return None  # 204 No Content
    
    except Exception as e:
        logger.error(f"Failed to clear cache for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
async def clear_all_caches(
    predictor: AmbientRecommender = Depends(get_ambient_predictor)
):
    """
    Clear all ambient recommendation caches.
    
    ⚠️  Warning: This will force cache refresh for ALL users
    on their next recommendation request. Use with caution.
    
    Args:
        predictor: Ambient recommender instance (injected)
    
    Returns:
        204 No Content on success
    """
    try:
        predictor.clear_cache()  # No user_id = clear all
        logger.warning("Cleared ALL ambient caches (admin action)")
        return None  # 204 No Content
    
    except Exception as e:
        logger.error(f"Failed to clear all caches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear caches"
        )