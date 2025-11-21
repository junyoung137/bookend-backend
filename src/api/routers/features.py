# src/api/routers/features.py
"""
Feature Extraction Router

Provides on-demand feature extraction and analysis endpoints.

Endpoints:
- GET /features/user/{user_id} - Extract user features
- GET /features/item/{item_id} - Extract item features
- POST /features/batch - Batch feature extraction
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.api.dependencies import get_db_session
from src.api.schemas.request import FeatureExtractionRequest
from src.api.schemas.response import UserFeatureResponse
from src.api.schemas.errors import NotFoundError
from src.database.models import User, Item, Interaction, UserFeature, ItemFeature

logger = logging.getLogger(__name__)

router = APIRouter()


# =========================================================
# User Feature Extraction
# =========================================================

@router.get("/user/{user_id}", response_model=UserFeatureResponse)
async def extract_user_features(
    user_id: int,
    lookback_days: int = Query(default=90, ge=1, le=365),
    feature_types: List[str] = Query(default=["temporal", "behavioral"]),
    db: Session = Depends(get_db_session)
):
    """
    Extract features for a specific user.
    
    Available feature types:
    - temporal: Time-based usage patterns
    - behavioral: Interaction counts and preferences
    - contextual: Device and location patterns
    
    Args:
        user_id: User database ID
        lookback_days: Days to look back for feature computation
        feature_types: Types of features to extract
        db: Database session (injected)
    
    Returns:
        UserFeatureResponse with extracted features
    
    Raises:
        404: User not found
    """
    try:
        # Validate user exists
        user = db.get(User, user_id)
        if not user:
            logger.warning(f"User not found: {user_id}")
            raise NotFoundError(resource="User", resource_id=user_id)
        
        # Get cutoff date
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Initialize features dictionary
        features: Dict[str, Any] = {}
        
        # Extract temporal features
        if "temporal" in feature_types:
            temporal_features = _extract_temporal_features(
                db, user_id, cutoff_date
            )
            features.update(temporal_features)
        
        # Extract behavioral features
        if "behavioral" in feature_types:
            behavioral_features = _extract_behavioral_features(
                db, user_id, cutoff_date
            )
            features.update(behavioral_features)
        
        # Extract contextual features
        if "contextual" in feature_types:
            contextual_features = _extract_contextual_features(
                db, user_id, cutoff_date
            )
            features.update(contextual_features)
        
        logger.info(
            f"Extracted {len(features)} features for user {user_id} "
            f"(types={feature_types}, lookback={lookback_days}d)"
        )
        
        return UserFeatureResponse(
            user_id=user_id,
            features=features,
            feature_types=feature_types,
            computed_at=datetime.now(),
            metadata={
                "lookback_days": lookback_days,
                "interaction_count": features.get("total_interactions", 0),
                "cutoff_date": cutoff_date.isoformat()
            }
        )
    
    except NotFoundError:
        raise
    
    except Exception as e:
        logger.error(f"Feature extraction failed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature extraction failed"
        )


# =========================================================
# Item Feature Extraction
# =========================================================

@router.get("/item/{item_id}")
async def extract_item_features(
    item_id: int,
    lookback_days: int = Query(default=90, ge=1, le=365),
    db: Session = Depends(get_db_session)
):
    """
    Extract features for a specific item.
    
    Returns:
    - Basic item metadata
    - Interaction statistics
    - Popularity metrics
    - Temporal usage patterns
    
    Args:
        item_id: Item database ID
        lookback_days: Days to look back for statistics
        db: Database session (injected)
    
    Returns:
        Item features dictionary
    
    Raises:
        404: Item not found
    """
    try:
        # Validate item exists
        item = db.get(Item, item_id)
        if not item:
            logger.warning(f"Item not found: {item_id}")
            raise NotFoundError(resource="Item", resource_id=item_id)
        
        # Get cutoff date
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Basic item metadata
        features: Dict[str, Any] = {
            "item_id": item.id,
            "item_code": getattr(item, "item_code", None),
            "item_name": getattr(item, "item_name", None),
            "category": getattr(item, "category", None),
            "is_active": getattr(item, "is_active", True)
        }
        
        # Interaction statistics
        interaction_count = db.query(func.count(Interaction.id)).filter(
            Interaction.item_id == item_id,
            Interaction.event_time >= cutoff_date
        ).scalar() or 0
        
        unique_users = db.query(func.count(func.distinct(Interaction.user_id))).filter(
            Interaction.item_id == item_id,
            Interaction.event_time >= cutoff_date
        ).scalar() or 0
        
        features.update({
            "total_interactions": interaction_count,
            "unique_users": unique_users,
            "avg_interactions_per_user": round(
                interaction_count / unique_users, 2
            ) if unique_users > 0 else 0.0
        })
        
        # Get stored item features if available
        item_feature = db.query(ItemFeature).filter(
            ItemFeature.item_id == item_id
        ).first()
        
        if item_feature:
            features.update({
                "popularity_score": getattr(item_feature, "popularity_score", 0.0),
                "trending_score": getattr(item_feature, "trending_score", 0.0),
                "freshness_score": getattr(item_feature, "freshness_score", 0.0)
            })
        
        logger.info(f"Extracted features for item {item_id} (lookback={lookback_days}d)")
        
        return {
            "features": features,
            "computed_at": datetime.now().isoformat(),
            "metadata": {
                "lookback_days": lookback_days,
                "cutoff_date": cutoff_date.isoformat()
            }
        }
    
    except NotFoundError:
        raise
    
    except Exception as e:
        logger.error(f"Feature extraction failed for item {item_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature extraction failed"
        )


# =========================================================
# Batch Feature Extraction
# =========================================================

@router.post("/batch")
async def extract_batch_features(
    request: FeatureExtractionRequest,
    db: Session = Depends(get_db_session)
):
    """
    Extract features for a user (batch endpoint for future expansion).
    
    Currently supports single user extraction with configurable
    feature types and lookback period.
    
    Args:
        request: Feature extraction request
        db: Database session (injected)
    
    Returns:
        UserFeatureResponse with extracted features
    """
    # Delegate to single user extraction
    return await extract_user_features(
        user_id=request.user_id,
        lookback_days=request.lookback_days,
        feature_types=request.feature_types,
        db=db
    )


# =========================================================
# Helper Functions
# =========================================================

def _extract_temporal_features(
    db: Session,
    user_id: int,
    cutoff_date: datetime
) -> Dict[str, Any]:
    """Extract temporal usage pattern features."""
    try:
        interactions = db.query(Interaction).filter(
            Interaction.user_id == user_id,
            Interaction.event_time >= cutoff_date
        ).all()
        
        if not interactions:
            return {
                "total_interactions": 0,
                "peak_hour": None,
                "peak_day": None
            }
        
        # Hour distribution
        hour_counts: Dict[int, int] = {}
        day_counts: Dict[int, int] = {}
        
        for interaction in interactions:
            hour = interaction.event_time.hour
            day = interaction.event_time.weekday()
            
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
            day_counts[day] = day_counts.get(day, 0) + 1
        
        peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else None
        peak_day = max(day_counts, key=day_counts.get) if day_counts else None
        
        return {
            "total_interactions": len(interactions),
            "peak_hour": peak_hour,
            "peak_day": peak_day,
            "hour_distribution": hour_counts,
            "day_distribution": day_counts
        }
    
    except Exception as e:
        logger.error(f"Temporal feature extraction failed: {e}")
        return {}


def _extract_behavioral_features(
    db: Session,
    user_id: int,
    cutoff_date: datetime
) -> Dict[str, Any]:
    """Extract behavioral pattern features."""
    try:
        # Category preferences
        category_counts = db.query(
            Item.category,
            func.count(Interaction.id).label("count")
        ).join(
            Interaction, Interaction.item_id == Item.id
        ).filter(
            Interaction.user_id == user_id,
            Interaction.event_time >= cutoff_date
        ).group_by(Item.category).all()
        
        category_preferences = {
            cat: count for cat, count in category_counts if cat
        }
        
        preferred_category = max(
            category_preferences,
            key=category_preferences.get
        ) if category_preferences else None
        
        # Recent activity (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_count = db.query(func.count(Interaction.id)).filter(
            Interaction.user_id == user_id,
            Interaction.event_time >= recent_cutoff
        ).scalar() or 0
        
        return {
            "category_preferences": category_preferences,
            "preferred_category": preferred_category,
            "last_7d_count": recent_count
        }
    
    except Exception as e:
        logger.error(f"Behavioral feature extraction failed: {e}")
        return {}


def _extract_contextual_features(
    db: Session,
    user_id: int,
    cutoff_date: datetime
) -> Dict[str, Any]:
    """Extract contextual pattern features."""
    try:
        # Device type distribution (if available in Interaction model)
        # This is a placeholder - adjust based on your actual schema
        
        return {
            "has_contextual_data": False,
            "note": "Contextual features require device/location data in Interaction model"
        }
    
    except Exception as e:
        logger.error(f"Contextual feature extraction failed: {e}")
        return {}