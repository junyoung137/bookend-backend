from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

from src.database.models import Item, Interaction, ItemFeature

logger = logging.getLogger(__name__)


class ItemFeatureExtractor:
    """
    Extract and compute item-level features from interaction data.
    
    Features computed:
    - Popularity score (total runs, unique users)
    - Trending score (recent activity growth)
    - Quality metrics (success rate, avg response time)
    - Temporal patterns (peak usage times)
    - Co-occurrence patterns (related items)
    """
    
    def __init__(self, session: Session):
        """
        Initialize feature extractor.
        
        Args:
            session: Database session
        """
        self.session = session
        self.logger = logging.getLogger(__name__)
    
    def extract_all_features(self, item_id: int) -> Optional[Dict[str, Any]]:
        """
        Extract all features for a single item.
        
        Args:
            item_id: Item database ID
        
        Returns:
            Dictionary of computed features, or None if item not found
        
        Example:
            >>> extractor = ItemFeatureExtractor(session)
            >>> features = extractor.extract_all_features(item_id=5)
        """
        # Get item
        item = self.session.get(Item, item_id)
        if not item:
            self.logger.warning(f"Item {item_id} not found")
            return None
        
        # Get item interactions
        interactions = self._get_item_interactions(item_id)
        
        if not interactions:
            self.logger.warning(f"No interactions found for item {item_id}")
            return self._get_default_features(item_id)
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'user_id': i.user_id,
            'event_name': i.event_name,
            'event_time': i.event_time,
            'response_time_ms': i.response_time_ms,
            'properties': i.properties,
        } for i in interactions])
        
        # Compute features
        features = {
            'item_id': item_id,
            **self._compute_popularity_features(df),
            **self._compute_trending_features(df),
            **self._compute_quality_features(df),
            **self._compute_temporal_features(df),
            **self._compute_retention_features(df),
            **self._compute_category_features(item, df),
        }
        
        # Add computed timestamp
        features['computed_at'] = datetime.now()
        
        self.logger.debug(f"Extracted {len(features)} features for item {item_id}")
        return features
    
    def extract_batch_features(self, item_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Extract features for multiple items in batch.
        
        Args:
            item_ids: List of item database IDs
        
        Returns:
            Dictionary mapping item_id to features
        
        Example:
            >>> extractor = ItemFeatureExtractor(session)
            >>> features = extractor.extract_batch_features([1, 2, 3])
        """
        results = {}
        total = len(item_ids)
        
        for idx, item_id in enumerate(item_ids, 1):
            try:
                features = self.extract_all_features(item_id)
                if features:
                    results[item_id] = features
                
                # Progress log
                if idx % 100 == 0 or idx == total:
                    self.logger.info(f"Progress: {idx}/{total} items processed")
                    
            except Exception as e:
                self.logger.error(f"Failed to extract features for item {item_id}: {e}")
        
        self.logger.info(f"✅ Extracted features for {len(results)}/{len(item_ids)} items")
        return results
    
    def compute_co_occurrence(
        self,
        item_id: int,
        window_hours: int = 24,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Compute items frequently used together with this item.
        
        Args:
            item_id: Item database ID
            window_hours: Time window for co-occurrence (hours)
            top_k: Number of top co-occurring items to return
        
        Returns:
            List of (co_item_id, co_occurrence_score) tuples
        
        Example:
            >>> extractor = ItemFeatureExtractor(session)
            >>> related = extractor.compute_co_occurrence(item_id=5, top_k=5)
        """
        # Get users who interacted with this item
        user_interactions = self.session.query(
            Interaction.user_id,
            Interaction.event_time
        ).filter(
            Interaction.item_id == item_id
        ).all()
        
        if not user_interactions:
            return []
        
        # For each user, find other items they used within time window
        co_occurrence_counts = {}
        
        for user_id, event_time in user_interactions:
            window_start = event_time - timedelta(hours=window_hours)
            window_end = event_time + timedelta(hours=window_hours)
            
            # Get other items this user interacted with in window
            other_items = self.session.query(
                Interaction.item_id,
                func.count(Interaction.id).label('count')
            ).filter(
                and_(
                    Interaction.user_id == user_id,
                    Interaction.item_id != item_id,
                    Interaction.event_time >= window_start,
                    Interaction.event_time <= window_end
                )
            ).group_by(Interaction.item_id).all()
            
            for other_item_id, count in other_items:
                if other_item_id not in co_occurrence_counts:
                    co_occurrence_counts[other_item_id] = 0
                co_occurrence_counts[other_item_id] += count
        
        # Sort by co-occurrence count and return top-k
        sorted_items = sorted(
            co_occurrence_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Normalize scores
        max_count = max(co_occurrence_counts.values()) if co_occurrence_counts else 1
        return [(item_id, count / max_count) for item_id, count in sorted_items]
    
    def _get_item_interactions(self, item_id: int) -> List[Interaction]:
        """Get all interactions for an item."""
        return self.session.query(Interaction).filter(
            Interaction.item_id == item_id
        ).order_by(Interaction.event_time.desc()).all()
    
    def _compute_popularity_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute popularity-based features."""
        # Count event types
        run_events = len(df[df['event_name'].str.contains('run', case=False, na=False)])
        select_events = len(df[df['event_name'].str.contains('select', case=False, na=False)])
        copy_events = len(df[df['event_name'].str.contains('copy', case=False, na=False)])
        
        # Unique users - all time
        unique_users_all_time = df['user_id'].nunique()
        
        # Popularity score: weighted combination
        popularity_score = (
            run_events * 1.0 +
            select_events * 1.5 +
            copy_events * 2.0
        ) / max(len(df), 1)
        
        return {
            'popularity_score': float(popularity_score),
            'total_runs': run_events,
            'total_selections': select_events,
            'total_copies': copy_events,
            'unique_users_all_time': unique_users_all_time,
        }
    
    def _compute_trending_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute trending/recency features."""
        now = pd.Timestamp.now(tz='UTC')
        seven_days_ago = now - pd.Timedelta(days=7)
        thirty_days_ago = now - pd.Timedelta(days=30)
        
        # Ensure timezone-aware
        df['event_time'] = pd.to_datetime(df['event_time'], utc=True)
        
        # Count recent activity
        recent_7d = len(df[df['event_time'] >= seven_days_ago])
        recent_30d = len(df[df['event_time'] >= thirty_days_ago])
        
        # Unique users in recent periods
        unique_users_7d = df[df['event_time'] >= seven_days_ago]['user_id'].nunique()
        unique_users_30d = df[df['event_time'] >= thirty_days_ago]['user_id'].nunique()
        
        # Trending score: recent activity / all-time activity
        total_activity = len(df)
        trending_score = recent_7d / max(total_activity, 1)
        
        # Freshness score: inverse of days since last use
        last_use = df['event_time'].max()
        days_since_last_use = (now - last_use).days if pd.notna(last_use) else 999
        freshness_score = 1.0 / (1.0 + days_since_last_use)
        
        return {
            'trending_score': float(trending_score),
            'freshness_score': float(freshness_score),
            'unique_users_7d': unique_users_7d,
            'unique_users_30d': unique_users_30d,
        }
    
    def _compute_quality_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute quality-related features."""
        # Average response time
        avg_response_time = df['response_time_ms'].mean() if not df['response_time_ms'].isna().all() else None
        
        # Success rate: ratio of selections/copies to runs
        run_events = len(df[df['event_name'].str.contains('run', case=False, na=False)])
        select_or_copy = len(df[df['event_name'].str.contains('select|copy', case=False, na=False)])
        success_rate = select_or_copy / max(run_events, 1)
        
        # Quality score: combination of response time and success rate
        time_score = 1.0 - min(avg_response_time / 10000.0, 1.0) if avg_response_time else 0.5
        quality_score = (time_score * 0.3 + success_rate * 0.7)
        
        return {
            'avg_response_time_ms': float(avg_response_time) if avg_response_time else None,
            'success_rate': float(success_rate),
            'quality_score': float(quality_score),
        }
    
    def _compute_temporal_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute temporal usage patterns."""
        if df['event_time'].isna().all():
            return {
                'peak_usage_hour': None,
                'peak_usage_day': None,
            }
        
        df['hour'] = pd.to_datetime(df['event_time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['event_time']).dt.dayofweek
        
        return {
            'peak_usage_hour': int(df['hour'].mode()[0]) if not df['hour'].isna().all() else None,
            'peak_usage_day': int(df['day_of_week'].mode()[0]) if not df['day_of_week'].isna().all() else None,
        }
    
    def _compute_retention_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute user retention metrics."""
        # Get unique users who used this item
        unique_users = df['user_id'].unique()
        
        if len(unique_users) == 0:
            return {'user_retention_rate': 0.0}
        
        # Count users who used it more than once
        repeat_users = df.groupby('user_id').size()
        returning_users = (repeat_users > 1).sum()
        
        retention_rate = returning_users / len(unique_users)
        
        return {
            'user_retention_rate': float(retention_rate),
        }
    
    def _compute_category_features(self, item: Item, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute category-relative features."""
        if not item.category:
            return {
                'category_rank': None,
                'category_percentile': None,
            }
        
        # Get all items in same category
        category_items = self.session.query(Item).filter(
            Item.category == item.category,
            Item.is_active == True
        ).all()
        
        if len(category_items) <= 1:
            return {
                'category_rank': 1,
                'category_percentile': 1.0,
            }
        
        # Rank by total usage
        item_usage_counts = {}
        for cat_item in category_items:
            count = self.session.query(func.count(Interaction.id)).filter(
                Interaction.item_id == cat_item.id
            ).scalar() or 0
            item_usage_counts[cat_item.id] = count
        
        # Sort and get rank
        sorted_items = sorted(item_usage_counts.items(), key=lambda x: x[1], reverse=True)
        rank = next((i + 1 for i, (iid, _) in enumerate(sorted_items) if iid == item.id), None)
        
        percentile = 1.0 - (rank - 1) / len(sorted_items) if rank else None
        
        return {
            'category_rank': rank,
            'category_percentile': float(percentile) if percentile else None,
        }
    
    def _get_default_features(self, item_id: int) -> Dict[str, Any]:
        """Get default feature values for items with no interactions."""
        return {
            'item_id': item_id,
            'popularity_score': 0.0,
            'total_runs': 0,
            'total_selections': 0,
            'total_copies': 0,
            'unique_users_all_time': 0,
            'trending_score': 0.0,
            'freshness_score': 0.0,
            'unique_users_7d': 0,
            'unique_users_30d': 0,
            'avg_response_time_ms': None,
            'success_rate': 0.0,
            'quality_score': 0.0,
            'peak_usage_hour': None,
            'peak_usage_day': None,
            'user_retention_rate': 0.0,
            'category_rank': None,
            'category_percentile': None,
        }


def save_features_to_db(session: Session, features: Dict[int, Dict[str, Any]]) -> int:
    """
    Save extracted features to item_features table.
    
    Args:
        session: Database session
        features: Dictionary mapping item_id to features
    
    Returns:
        Number of records saved
    """
    saved_count = 0
    
    for item_id, feature_dict in features.items():
        try:
            # Check if record exists
            existing = session.query(ItemFeature).filter_by(item_id=item_id).first()
            
            if existing:
                # Update existing record
                for key, value in feature_dict.items():
                    if key != 'item_id' and hasattr(existing, key):
                        setattr(existing, key, value)
                logger.debug(f"Updated features for item {item_id}")
            else:
                # Create new record
                item_feature = ItemFeature(**feature_dict)
                session.add(item_feature)
                logger.debug(f"Created features for item {item_id}")
            
            saved_count += 1
            
        except Exception as e:
            logger.error(f"Failed to save features for item {item_id}: {e}")
            continue
    
    session.commit()
    logger.info(f"✅ Saved {saved_count} item features to database")
    return saved_count


if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres
    
    setup_logging()
    
    pg = get_postgres()
    
    with pg.transaction() as session:
        extractor = ItemFeatureExtractor(session)
        
        # 전체 아이템 수 확인
        total_items = session.query(Item).count()
        print(f"\n{'='*60}")
        print(f"📊 Total Items in Database: {total_items:,}")
        print(f"{'='*60}\n")
        
        # Interactions가 있는 아이템만 찾기
        items_with_interactions = session.query(
            Item.id, Item.item_name, func.count(Interaction.id).label('interaction_count')
        ).join(
            Interaction, Item.id == Interaction.item_id
        ).group_by(
            Item.id, Item.item_name
        ).order_by(
            func.count(Interaction.id).desc()
        ).limit(10).all()
        
        print(f"🔥 Top 10 Items with Most Interactions:")
        print(f"{'='*60}")
        for idx, (item_id, item_name, count) in enumerate(items_with_interactions, 1):
            print(f"{idx:2d}. Item ID: {item_id:5d} | Name: {item_name[:40]:40s} | Events: {count:5d}")
        print(f"{'='*60}\n")
        
        if items_with_interactions:
            # 첫 번째 인기 아이템의 피처 추출
            first_item_id = items_with_interactions[0][0]
            first_item_name = items_with_interactions[0][1]
            
            print(f"🎯 Extracting features for top item: {first_item_name}")
            print(f"{'='*60}\n")
            
            features = extractor.extract_all_features(first_item_id)
            
            if features:
                print(f"📊 Item Features:")
                print(f"{'='*60}")
                for key, value in features.items():
                    if key != 'item_id' and key != 'computed_at':
                        print(f"  {key:30s}: {value}")
                print(f"{'='*60}\n")
                
                # 연관 아이템 찾기
                print(f"🔗 Co-occurring Items (Top 5):")
                print(f"{'='*60}")
                related = extractor.compute_co_occurrence(first_item_id, top_k=5)
                if related:
                    for related_id, score in related:
                        related_item = session.query(Item).get(related_id)
                        item_name = related_item.item_name if related_item else "Unknown"
                        print(f"  Item {related_id:5d} ({item_name[:30]:30s}): {score:.3f}")
                else:
                    print("  No co-occurring items found")
                print(f"{'='*60}\n")
                
                # 배치로 모든 아이템 피처 추출
                print(f"🚀 Starting batch feature extraction for all items...")
                all_item_ids = [item_id for item_id, _, _ in items_with_interactions]
                batch_features = extractor.extract_batch_features(all_item_ids)
                
                # DB에 저장
                print(f"\n💾 Saving features to database...")
                saved_count = save_features_to_db(session, batch_features)
                
                print(f"\n✅ Feature extraction completed!")
                print(f"{'='*60}")
                print(f"  Total processed: {len(batch_features)}")
                print(f"  Saved to DB: {saved_count}")
                print(f"{'='*60}\n")
        else:
            print("❌ No items with interactions found!")