# src/api/routers/temporal.py
"""
Temporal Flow Recommendations Router

Provides time-pattern based recommendations that adapt to:
- User temporal patterns (peak activity hours/days)
- Item temporal patterns (when items are typically used)
- Current time context (real-time matching)
- Recency decay (time-since-last-interaction)

Endpoints:
- POST /temporal/recommend - Generate temporal recommendations
- GET /temporal/pattern/{user_id} - Get user's temporal pattern analysis
"""

import time
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session, get_temporal_predictor
from src.api.schemas.request import TemporalRecommendRequest
from src.api.schemas.response import RecommendationResponse, RecommendationItem
from src.api.schemas.errors import NotFoundError, ServiceUnavailableError
from src.models.hybrid.temporal_recommender import TemporalRecommender
from src.database.models import User

logger = logging.getLogger(__name__)

router = APIRouter()


# =========================================================
# Main Recommendation Endpoint
# =========================================================

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_temporal(
    request: TemporalRecommendRequest,
    db: Session = Depends(get_db_session),
    predictor: TemporalRecommender = Depends(get_temporal_predictor)
):
    """
    Generate temporal (time-pattern based) recommendations.
    
    Features:
    - User temporal pattern matching (peak hours/days)
    - Item temporal pattern analysis
    - Real-time context matching
    - Recency decay scoring
    - Peak hour boosting
    
    Args:
        request: Temporal recommendation request
        db: Database session (injected)
        predictor: Temporal recommender instance (injected)
    
    Returns:
        RecommendationResponse with time-aware items
    
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
        else:
            # Default to current time if no context provided
            context_dict = {"timestamp": datetime.now()}
        
        # Generate recommendations
        logger.info(
            f"Generating temporal recommendations for user {request.user_id} "
            f"(limit={request.limit}, min_score={request.min_score})"
        )
        
        recommendations = predictor.recommend(
            user_id=request.user_id,
            context=context_dict,
            limit=request.limit,
            min_score=request.min_score,
            include_reasons=request.include_reasons
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
                reason=rec.reason if request.include_reasons else "Recommended",
                metadata=rec.metadata,
                timestamp=datetime.now()
            )
            for rec in recommendations
        ]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Temporal recommendations generated: {len(response_items)} items "
            f"in {latency_ms:.2f}ms"
        )
        
        # Build response
        return RecommendationResponse(
            recommendations=response_items,
            user_id=request.user_id,
            total_count=len(response_items),
            model_name=predictor.get_model_name(),
            request_metadata={
                "limit": request.limit,
                "min_score": request.min_score,
                "include_reasons": request.include_reasons,
                "context_timestamp": context_dict.get("timestamp").isoformat() if context_dict.get("timestamp") else None
            },
            performance_metrics={
                "latency_ms": round(latency_ms, 2),
                "avg_score": round(
                    sum(item.score for item in response_items) / len(response_items), 4
                ) if response_items else 0.0,
                "min_score_threshold": request.min_score
            },
            timestamp=datetime.now()
        )
    
    except NotFoundError:
        raise
    
    except Exception as e:
        logger.error(
            f"Temporal recommendation failed for user {request.user_id}: {e}",
            exc_info=True
        )
        raise ServiceUnavailableError(
            service="Temporal Recommender",
            message=f"Failed to generate recommendations: {str(e)}"
        )


# =========================================================
# Pattern Analysis Endpoint
# =========================================================

@router.get("/pattern/{user_id}")
async def get_user_pattern(
    user_id: int,
    use_cache: bool = True,
    db: Session = Depends(get_db_session),
    predictor: TemporalRecommender = Depends(get_temporal_predictor)
):
    """
    Get temporal pattern analysis for a user.
    
    Returns detailed analysis of user's temporal behavior:
    - Hour distribution (activity by hour of day)
    - Day distribution (activity by day of week)
    - Peak hours and days
    - Time-of-day ratios (morning, afternoon, evening, night)
    - Weekday vs weekend behavior
    
    Args:
        user_id: User database ID
        use_cache: Use cached pattern if available (default: True)
        db: Database session (injected)
        predictor: Temporal recommender instance (injected)
    
    Returns:
        Temporal pattern analysis dictionary
    
    Raises:
        404: User not found
    """
    try:
        # Validate user exists
        user = db.get(User, user_id)
        if not user:
            logger.warning(f"User not found: {user_id}")
            raise NotFoundError(resource="User", resource_id=user_id)
        
        # Get pattern from analyzer
        pattern = predictor.temporal_analyzer.get_user_temporal_pattern(
            user_id,
            use_cache=use_cache
        )
        
        logger.debug(f"Retrieved temporal pattern for user {user_id}")
        
        return {
            "user_id": user_id,
            "pattern": pattern,
            "cached": use_cache,
            "timestamp": datetime.now().isoformat()
        }
    
    except NotFoundError:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get pattern for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve temporal pattern"
        )


@router.delete("/pattern/cache/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_pattern_cache(
    user_id: int,
    predictor: TemporalRecommender = Depends(get_temporal_predictor)
):
    """
    Clear temporal pattern cache for a specific user.
    
    Forces re-computation of temporal patterns on next request.
    
    Args:
        user_id: User database ID
        predictor: Temporal recommender instance (injected)
    
    Returns:
        204 No Content on success
    """
    try:
        predictor.temporal_analyzer.clear_cache(user_id)
        logger.info(f"Cleared temporal pattern cache for user {user_id}")
        return None  # 204 No Content
    
    except Exception as e:
        logger.error(f"Failed to clear pattern cache for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear pattern cache"
        )