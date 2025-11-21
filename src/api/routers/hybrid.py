# src/api/routers/hybrid.py
"""
Hybrid Recommendations Router

Provides multi-strategy recommendations combining:
- User-based Collaborative Filtering
- Item-based Collaborative Filtering
- Context-aware features (time, device, location)
- Recency boost
- Optional MMR diversity

Endpoints:
- POST /hybrid/recommend - Generate hybrid recommendations
- POST /hybrid/batch - Batch recommendations for multiple users
"""

import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from src.api.dependencies import get_db_session, get_hybrid_predictor
from src.api.schemas.request import HybridRecommendRequest
from src.api.schemas.response import RecommendationResponse, RecommendationItem
from src.api.schemas.errors import NotFoundError, ServiceUnavailableError
from src.models.hybrid.hybrid_recommender import HybridRecommender
from src.database.models import User

logger = logging.getLogger(__name__)

router = APIRouter()


# =========================================================
# Batch Request Schema
# =========================================================

class BatchRecommendRequest(BaseModel):
    """Request schema for batch recommendations."""
    
    user_ids: List[int] = Field(
        ...,
        description="List of user IDs to generate recommendations for",
        min_length=1,
        max_length=100
    )
    
    limit: int = Field(
        default=10,
        description="Number of recommendations per user",
        ge=1,
        le=50
    )
    
    min_score: float = Field(
        default=0.0,
        description="Minimum score threshold",
        ge=0.0,
        le=1.0
    )
    
    include_reasons: bool = Field(
        default=False,
        description="Include explanation reasons (slower)"
    )
    
    enable_diversity: bool = Field(
        default=True,
        description="Enable MMR-based diversity"
    )


# =========================================================
# Main Recommendation Endpoint
# =========================================================

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_hybrid(
    request: HybridRecommendRequest,
    db: Session = Depends(get_db_session),
    predictor: HybridRecommender = Depends(get_hybrid_predictor)
):
    """
    Generate hybrid (multi-strategy) recommendations.
    
    Features:
    - User-CF: Recommendations from similar users
    - Item-CF: Recommendations from similar items
    - Context: Time/device/location adjustments
    - Recency: Boost for recently popular items
    - Diversity: Optional MMR-based diversity re-ranking
    
    Args:
        request: Hybrid recommendation request
        db: Database session (injected)
        predictor: Hybrid recommender instance (injected)
    
    Returns:
        RecommendationResponse with blended recommendations
    
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
        
        # Temporarily override MMR setting if requested
        original_mmr = predictor.enable_mmr
        if not request.enable_diversity:
            predictor.enable_mmr = False
        
        try:
            # Generate recommendations
            logger.info(
                f"Generating hybrid recommendations for user {request.user_id} "
                f"(limit={request.limit}, min_score={request.min_score}, "
                f"diversity={request.enable_diversity})"
            )
            
            recommendations = predictor.recommend(
                user_id=request.user_id,
                context=context_dict,
                limit=request.limit,
                min_score=request.min_score,
                include_reasons=request.include_reasons
            )
            
        finally:
            # Restore original MMR setting
            predictor.enable_mmr = original_mmr
        
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
                reason=rec.reason if request.include_reasons else "Recommended for you",
                metadata=rec.metadata,
                timestamp=datetime.now()
            )
            for rec in recommendations
        ]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Hybrid recommendations generated: {len(response_items)} items "
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
                "enable_diversity": request.enable_diversity,
                "context_provided": context_dict is not None,
                "weights": {
                    "user_cf": predictor.user_cf_weight,
                    "item_cf": predictor.item_cf_weight,
                    "context": predictor.context_weight,
                    "recency": predictor.recency_weight
                }
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
            f"Hybrid recommendation failed for user {request.user_id}: {e}",
            exc_info=True
        )
        raise ServiceUnavailableError(
            service="Hybrid Recommender",
            message=f"Failed to generate recommendations: {str(e)}"
        )


# =========================================================
# Batch Recommendation Endpoint
# =========================================================

@router.post("/batch")
async def recommend_batch(
    request: BatchRecommendRequest,
    db: Session = Depends(get_db_session),
    predictor: HybridRecommender = Depends(get_hybrid_predictor)
):
    """
    Generate recommendations for multiple users in batch.
    
    ⚠️  Performance Note:
    - Use sparingly (max 100 users per request)
    - Consider disabling reasons for better performance
    - Results are not cached
    
    Args:
        request: Batch recommendation request
        db: Database session (injected)
        predictor: Hybrid recommender instance (injected)
    
    Returns:
        Dictionary with results per user_id
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"Batch recommendation request for {len(request.user_ids)} users "
            f"(limit={request.limit})"
        )
        
        results: Dict[int, Any] = {}
        success_count = 0
        failure_count = 0
        
        # Temporarily override MMR setting if requested
        original_mmr = predictor.enable_mmr
        if not request.enable_diversity:
            predictor.enable_mmr = False
        
        try:
            for user_id in request.user_ids:
                try:
                    # Validate user exists
                    user = db.get(User, user_id)
                    if not user:
                        results[user_id] = {
                            "success": False,
                            "error": "User not found"
                        }
                        failure_count += 1
                        continue
                    
                    # Generate recommendations
                    recommendations = predictor.recommend(
                        user_id=user_id,
                        context=None,  # No context in batch mode
                        limit=request.limit,
                        min_score=request.min_score,
                        include_reasons=request.include_reasons
                    )
                    
                    # Convert to response items
                    response_items = [
                        {
                            "item_id": rec.item_id,
                            "item_code": rec.item_code,
                            "item_name": rec.item_name,
                            "score": rec.score,
                            "rank": rec.rank,
                            "reason": rec.reason if request.include_reasons else "Recommended"
                        }
                        for rec in recommendations
                    ]
                    
                    results[user_id] = {
                        "success": True,
                        "recommendations": response_items,
                        "total_count": len(response_items)
                    }
                    success_count += 1
                
                except Exception as e:
                    logger.error(f"Batch recommendation failed for user {user_id}: {e}")
                    results[user_id] = {
                        "success": False,
                        "error": str(e)
                    }
                    failure_count += 1
        
        finally:
            # Restore original MMR setting
            predictor.enable_mmr = original_mmr
        
        # Calculate total latency
        processing_time = time.time() - start_time
        
        logger.info(
            f"Batch recommendations completed: {success_count} success, "
            f"{failure_count} failures in {processing_time:.2f}s"
        )
        
        return {
            "results": results,
            "total_users": len(request.user_ids),
            "success_count": success_count,
            "failure_count": failure_count,
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch recommendation endpoint failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch recommendation processing failed"
        )


# =========================================================
# Model Info Endpoint
# =========================================================

@router.get("/info")
async def get_model_info(
    predictor: HybridRecommender = Depends(get_hybrid_predictor)
):
    """
    Get hybrid recommender configuration and status.
    
    Returns:
        Model configuration and component status
    """
    try:
        return {
            "model_name": predictor.get_model_name(),
            "is_fitted": predictor.is_fitted,
            "weights": {
                "user_cf": predictor.user_cf_weight,
                "item_cf": predictor.item_cf_weight,
                "context": predictor.context_weight,
                "recency": predictor.recency_weight
            },
            "configuration": {
                "enable_mmr": predictor.enable_mmr,
                "mmr_lambda": predictor.mmr_lambda,
                "recency_cache_ttl": predictor.recency_cache_ttl
            },
            "components": {
                "user_cf": {
                    "fitted": predictor.user_cf.is_fitted if hasattr(predictor.user_cf, 'is_fitted') else False
                },
                "item_cf": {
                    "fitted": predictor.item_cf.is_fitted if hasattr(predictor.item_cf, 'is_fitted') else False
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )