from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from src.database.models import User, Interaction, UserFeature

logger = logging.getLogger(__name__)


class UserFeatureExtractor:
    """
    Extract and compute user-level features from raw data.
    
    Features computed:
    - Total paraphrases (all-time, 7d, 30d)
    - Preferred tone, maintenance, language
    - Most active hour and day
    - Device/browser preferences
    - Engagement and exploration scores
    """
    
    def __init__(self, session: Session):
        """
        Initialize feature extractor.
        
        Args:
            session: Database session
        """
        self.session = session
        self.logger = logging.getLogger(__name__)
    
    def extract_all_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Extract all features for a single user.
        
        Args:
            user_id: User database ID
        
        Returns:
            Dictionary of computed features, or None if user not found
        
        Example:
            >>> extractor = UserFeatureExtractor(session)
            >>> features = extractor.extract_all_features(user_id=123)
        """
        # Get user
        user = self.session.get(User, user_id)
        if not user:
            self.logger.warning(f"User {user_id} not found")
            return None
        
        # Get user interactions
        interactions = self._get_user_interactions(user_id)
        
        if not interactions:
            self.logger.warning(f"No interactions found for user {user_id}")
            return self._get_default_features()
        
        # Convert to DataFrame for easier computation
        df = pd.DataFrame([{
            'event_name': i.event_name,
            'event_time': i.event_time,
            'tone': i.tone,
            'maintenance': i.maintenance,
            'target_language': i.target_language,
            'response_time_ms': i.response_time_ms,
            'input_sentence_length': i.input_sentence_length,
            'browser': i.browser,
            'os': i.os,
            'device_id': i.device_id,
        } for i in interactions])
        
        # Compute features
        features = {
            'user_id': user_id,
            **self._compute_count_features(df),
            **self._compute_preference_features(df),
            **self._compute_temporal_features(df),
            **self._compute_device_features(df),
            **self._compute_quality_features(df),
            **self._compute_engagement_features(df),
            **self._compute_recency_features(df, user),
        }
        
        # Add computed timestamp
        features['computed_at'] = datetime.now()
        
        self.logger.debug(f"Extracted {len(features)} features for user {user_id}")
        return features
    
    def extract_batch_features(self, user_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Extract features for multiple users in batch.
        
        Args:
            user_ids: List of user database IDs
        
        Returns:
            Dictionary mapping user_id to features
        
        Example:
            >>> extractor = UserFeatureExtractor(session)
            >>> features = extractor.extract_batch_features([1, 2, 3])
        """
        results = {}
        total = len(user_ids)
        
        for idx, user_id in enumerate(user_ids, 1):
            try:
                features = self.extract_all_features(user_id)
                if features:
                    results[user_id] = features
                
                # Progress log
                if idx % 100 == 0 or idx == total:
                    self.logger.info(f"Progress: {idx}/{total} users processed")
                    
            except Exception as e:
                self.logger.error(f"Failed to extract features for user {user_id}: {e}")
        
        self.logger.info(f"✅ Extracted features for {len(results)}/{len(user_ids)} users")
        return results
    
    def _get_user_interactions(self, user_id: int) -> List[Interaction]:
        """Get all interactions for a user."""
        return self.session.query(Interaction).filter(
            Interaction.user_id == user_id
        ).order_by(Interaction.event_time.desc()).all()
    
    def _compute_count_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute count-based features."""
        now = pd.Timestamp.now(tz='UTC')
        seven_days_ago = now - pd.Timedelta(days=7)
        thirty_days_ago = now - pd.Timedelta(days=30)
        
        # Ensure event_time is timezone-aware
        df['event_time'] = pd.to_datetime(df['event_time'], utc=True)
        
        return {
            'total_paraphrases': len(df),
            'last_7d_count': len(df[df['event_time'] >= seven_days_ago]),
            'last_30d_count': len(df[df['event_time'] >= thirty_days_ago]),
        }
    
    def _compute_preference_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute user preference features."""
        return {
            'preferred_tone': df['tone'].mode()[0] if not df['tone'].isna().all() else None,
            'preferred_maintenance': df['maintenance'].mode()[0] if not df['maintenance'].isna().all() else None,
            'preferred_language': df['target_language'].mode()[0] if not df['target_language'].isna().all() else None,
        }
    
    def _compute_temporal_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute temporal activity features."""
        if df['event_time'].isna().all():
            return {
                'most_active_hour': None,
                'most_active_day_of_week': None,
            }
        
        df['hour'] = pd.to_datetime(df['event_time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['event_time']).dt.dayofweek
        
        return {
            'most_active_hour': int(df['hour'].mode()[0]) if not df['hour'].isna().all() else None,
            'most_active_day_of_week': int(df['day_of_week'].mode()[0]) if not df['day_of_week'].isna().all() else None,
        }
    
    def _compute_device_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute device/browser preference features."""
        return {
            'primary_browser': df['browser'].mode()[0] if not df['browser'].isna().all() else None,
            'primary_os': df['os'].mode()[0] if not df['os'].isna().all() else None,
            'primary_device': df['device_id'].mode()[0] if not df['device_id'].isna().all() else None,
        }
    
    def _compute_quality_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute quality-related features."""
        return {
            'avg_response_time_ms': float(df['response_time_ms'].mean()) if not df['response_time_ms'].isna().all() else None,
            'avg_input_length': float(df['input_sentence_length'].mean()) if not df['input_sentence_length'].isna().all() else None,
        }
    
    def _compute_engagement_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute engagement metrics."""
        # Count different event types
        run_events = len(df[df['event_name'].str.contains('run', case=False, na=False)])
        select_events = len(df[df['event_name'].str.contains('select', case=False, na=False)])
        copy_events = len(df[df['event_name'].str.contains('copy', case=False, na=False)])
        
        total_events = len(df)
        
        # Engagement score: weighted combination of interactions
        engagement_score = (
            run_events * 1.0 +
            select_events * 1.5 +
            copy_events * 2.0
        ) / max(total_events, 1)
        
        # Exploration score: variety of tones/maintenance used
        tone_variety = df['tone'].nunique() if not df['tone'].isna().all() else 0
        maintenance_variety = df['maintenance'].nunique() if not df['maintenance'].isna().all() else 0
        exploration_score = (tone_variety + maintenance_variety) / 10.0  # Normalize
        
        return {
            'engagement_score': float(engagement_score),
            'exploration_score': float(exploration_score),
            'avg_selections_per_session': float(select_events / max(run_events, 1)),
            'copy_rate': float(copy_events / max(run_events, 1)),
        }
    
    def _compute_recency_features(self, df: pd.DataFrame, user: User) -> Dict[str, Any]:
        """Compute recency-based features."""
        now = datetime.now(tz=pd.Timestamp.now(tz='UTC').tz)
        
        # Days since first interaction
        first_interaction = df['event_time'].min()
        days_since_first = (now - first_interaction).days if pd.notna(first_interaction) else None
        
        # Days since last interaction
        last_interaction = df['event_time'].max()
        days_since_last = (now - last_interaction).days if pd.notna(last_interaction) else None
        
        return {
            'days_since_first_interaction': days_since_first,
            'days_since_last_interaction': days_since_last,
        }
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Get default feature values for users with no interactions."""
        return {
            'total_paraphrases': 0,
            'last_7d_count': 0,
            'last_30d_count': 0,
            'preferred_tone': None,
            'preferred_maintenance': None,
            'preferred_language': None,
            'most_active_hour': None,
            'most_active_day_of_week': None,
            'primary_browser': None,
            'primary_os': None,
            'primary_device': None,
            'avg_response_time_ms': None,
            'avg_input_length': None,
            'engagement_score': 0.0,
            'exploration_score': 0.0,
            'avg_selections_per_session': 0.0,
            'copy_rate': 0.0,
            'days_since_first_interaction': None,
            'days_since_last_interaction': None,
        }


def save_features_to_db(session: Session, features: Dict[int, Dict[str, Any]]) -> int:
    """
    Save extracted features to user_features table.
    
    Args:
        session: Database session
        features: Dictionary mapping user_id to features
    
    Returns:
        Number of records saved
    """
    saved_count = 0
    
    for user_id, feature_dict in features.items():
        try:
            # Check if record exists
            existing = session.query(UserFeature).filter_by(user_id=user_id).first()
            
            if existing:
                # Update existing record
                for key, value in feature_dict.items():
                    if key != 'user_id' and hasattr(existing, key):
                        setattr(existing, key, value)
                logger.debug(f"Updated features for user {user_id}")
            else:
                # Create new record
                user_feature = UserFeature(**feature_dict)
                session.add(user_feature)
                logger.debug(f"Created features for user {user_id}")
            
            saved_count += 1
            
        except Exception as e:
            logger.error(f"Failed to save features for user {user_id}: {e}")
            continue
    
    session.commit()
    logger.info(f"✅ Saved {saved_count} user features to database")
    return saved_count


if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres
    
    setup_logging()
    
    pg = get_postgres()
    
    with pg.transaction() as session:
        extractor = UserFeatureExtractor(session)
        
        # 전체 유저 수 확인
        total_users = session.query(User).count()
        print(f"\n{'='*60}")
        print(f"📊 Total Users in Database: {total_users:,}")
        print(f"{'='*60}\n")
        
        # Interactions가 있는 유저만 찾기
        users_with_interactions = session.query(
            User.id, User.distinct_id, func.count(Interaction.id).label('interaction_count')
        ).join(
            Interaction, User.id == Interaction.user_id
        ).group_by(
            User.id, User.distinct_id
        ).order_by(
            func.count(Interaction.id).desc()
        ).limit(10).all()
        
        print(f"🔥 Top 10 Users with Most Interactions:")
        print(f"{'='*60}")
        for idx, (user_id, distinct_id, count) in enumerate(users_with_interactions, 1):
            print(f"{idx:2d}. User ID: {user_id:5d} | Distinct ID: {distinct_id[:40]:40s} | Events: {count:5d}")
        print(f"{'='*60}\n")
        
        if users_with_interactions:
            # 첫 번째 파워 유저의 피처 추출
            first_user_id = users_with_interactions[0][0]
            first_user_distinct_id = users_with_interactions[0][1]
            
            print(f"🎯 Extracting features for top user: {first_user_distinct_id}")
            print(f"{'='*60}\n")
            
            features = extractor.extract_all_features(first_user_id)
            
            if features:
                print(f"📊 User Features:")
                print(f"{'='*60}")
                for key, value in features.items():
                    if key != 'user_id' and key != 'computed_at':
                        print(f"  {key:30s}: {value}")
                print(f"{'='*60}\n")
                
                # 배치로 모든 유저 피처 추출
                print(f"🚀 Starting batch feature extraction for all users...")
                all_user_ids = [uid for uid, _, _ in users_with_interactions]
                batch_features = extractor.extract_batch_features(all_user_ids)
                
                # DB에 저장
                print(f"\n💾 Saving features to database...")
                saved_count = save_features_to_db(session, batch_features)
                
                print(f"\n✅ Feature extraction completed!")
                print(f"{'='*60}")
                print(f"  Total processed: {len(batch_features)}")
                print(f"  Saved to DB: {saved_count}")
                print(f"{'='*60}\n")
        else:
            print("❌ No users with interactions found!")