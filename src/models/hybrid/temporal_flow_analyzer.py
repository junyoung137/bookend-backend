from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_

from src.database.models import Interaction, User

logger = logging.getLogger(__name__)


class TemporalFlowAnalyzer:
    """Analyze temporal patterns in user-item interactions."""
    
    def __init__(self, session: Session, lookback_days: int = 90):
        self.session = session
        self.lookback_days = lookback_days
        self.logger = logger
        self._user_pattern_cache: Dict[int, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[int, datetime] = {}
        self._cache_ttl_seconds = 3600  # 1 hour

    # ================= Public API =================
    def get_user_temporal_pattern(self, user_id: int, use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and self._is_cache_valid(user_id):
            return self._user_pattern_cache[user_id]

        interactions = self._get_user_interactions(user_id)
        if not interactions:
            return self._get_default_pattern()

        df = self._interactions_to_dataframe(interactions)

        pattern = {
            'hour_distribution': self._compute_hour_distribution(df),
            'day_distribution': self._compute_day_distribution(df),
            'peak_hour': self._compute_peak_hour(df),
            'peak_day': self._compute_peak_day(df),
            'morning_ratio': self._compute_ratio(df, 6, 12),
            'afternoon_ratio': self._compute_ratio(df, 12, 18),
            'evening_ratio': self._compute_ratio(df, 18, 22),
            'night_ratio': self._compute_ratio(df, 22, 6),
            'weekday_ratio': self._compute_ratio(df, None, None, weekday=True),
            'weekend_ratio': self._compute_ratio(df, None, None, weekend=True),
            'business_hours_ratio': self._compute_ratio(df, 9, 18, weekday=True),
            'total_interactions': len(df),
            'analyzed_period_days': self.lookback_days,
        }

        self._user_pattern_cache[user_id] = pattern
        self._cache_timestamps[user_id] = datetime.now()
        return pattern

    def get_item_temporal_pattern(self, item_id: int) -> Dict[str, Any]:
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        interactions = self.session.query(Interaction).filter(
            and_(Interaction.item_id == item_id, Interaction.event_time >= cutoff_date)
        ).all()
        if not interactions:
            return self._get_default_pattern()

        df = self._interactions_to_dataframe(interactions)
        return {
            'hour_distribution': self._compute_hour_distribution(df),
            'day_distribution': self._compute_day_distribution(df),
            'peak_hour': self._compute_peak_hour(df),
            'peak_day': self._compute_peak_day(df),
            'total_interactions': len(df),
        }

    def compute_temporal_affinity(self, user_pattern: Dict[str, Any], item_pattern: Dict[str, Any], current_context: Optional[Dict[str, Any]] = None) -> float:
        user_hour_dist = user_pattern.get('hour_distribution', {})
        item_hour_dist = item_pattern.get('hour_distribution', {})
        if not user_hour_dist or not item_hour_dist:
            return 0.5

        hour_affinity = self._distribution_similarity(user_hour_dist, item_hour_dist)
        day_affinity = self._distribution_similarity(user_pattern.get('day_distribution', {}), item_pattern.get('day_distribution', {}))
        base_affinity = 0.7 * hour_affinity + 0.3 * day_affinity

        if current_context:
            current_hour = current_context.get('hour')
            peak_hour = user_pattern.get('peak_hour')
            if current_hour is not None and peak_hour is not None:
                hour_distance = min(abs(current_hour - peak_hour), 24 - abs(current_hour - peak_hour))
                if hour_distance <= 2:
                    boost = 1.0 - (hour_distance / 3.0) * 0.2
                    base_affinity *= boost

        return float(np.clip(base_affinity, 0.0, 1.0))

    # ================= Private Helpers =================
    def _get_user_interactions(self, user_id: int) -> List[Interaction]:
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        try:
            return self.session.query(Interaction).filter(
                and_(Interaction.user_id == user_id, Interaction.event_time >= cutoff_date)
            ).order_by(Interaction.event_time).all()
        except Exception as e:
            logger.error(f"Fetch interactions failed: {e}")
            return []

    def _interactions_to_dataframe(self, interactions: List[Interaction]) -> pd.DataFrame:
        data = [{'event_time': i.event_time, 'hour': i.event_time.hour, 'day_of_week': i.event_time.weekday()} for i in interactions if i.event_time]
        return pd.DataFrame(data)

    def _compute_hour_distribution(self, df: pd.DataFrame) -> Dict[int, float]:
        return self._compute_distribution(df, 'hour')

    def _compute_day_distribution(self, df: pd.DataFrame) -> Dict[int, float]:
        return self._compute_distribution(df, 'day_of_week')

    def _compute_peak_hour(self, df: pd.DataFrame) -> Optional[int]:
        return self._compute_peak(df, 'hour')

    def _compute_peak_day(self, df: pd.DataFrame) -> Optional[int]:
        return self._compute_peak(df, 'day_of_week')

    def _compute_ratio(self, df: pd.DataFrame, start_hour: Optional[int], end_hour: Optional[int], weekday=False, weekend=False) -> float:
        if df.empty:
            return 0.0
        if weekday:
            mask = df['day_of_week'] < 5
        elif weekend:
            mask = df['day_of_week'] >= 5
        elif start_hour is not None and end_hour is not None:
            if start_hour < end_hour:
                mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
            else:
                mask = (df['hour'] >= start_hour) | (df['hour'] < end_hour)
        else:
            return 0.0
        return float(mask.sum() / len(df))

    def _compute_distribution(self, df: pd.DataFrame, column: str) -> Dict[int, float]:
        if df.empty:
            return {}
        counts = df[column].value_counts().sort_index()
        total = counts.sum()
        return {int(k): float(v / total) for k, v in counts.items()} if total > 0 else {}

    def _compute_peak(self, df: pd.DataFrame, column: str) -> Optional[int]:
        if df.empty:
            return None
        counts = df[column].value_counts()
        return int(counts.idxmax()) if not counts.empty else None

    def _distribution_similarity(self, dist_a: Dict[int, float], dist_b: Dict[int, float]) -> float:
        all_keys = set(dist_a.keys()) | set(dist_b.keys())
        if not all_keys:
            return 0.5
        vec_a = np.array([dist_a.get(k, 0.0) for k in sorted(all_keys)])
        vec_b = np.array([dist_b.get(k, 0.0) for k in sorted(all_keys)])
        norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    def _get_default_pattern(self) -> Dict[str, Any]:
        return {
            'hour_distribution': {},
            'day_distribution': {},
            'peak_hour': None,
            'peak_day': None,
            'morning_ratio': 0.0,
            'afternoon_ratio': 0.0,
            'evening_ratio': 0.0,
            'night_ratio': 0.0,
            'weekday_ratio': 0.0,
            'weekend_ratio': 0.0,
            'business_hours_ratio': 0.0,
            'total_interactions': 0,
            'analyzed_period_days': self.lookback_days,
        }

    def _is_cache_valid(self, user_id: int) -> bool:
        if user_id not in self._cache_timestamps:
            return False
        age = (datetime.now() - self._cache_timestamps[user_id]).total_seconds()
        return age < self._cache_ttl_seconds

    def clear_cache(self, user_id: Optional[int] = None) -> None:
        if user_id is not None:
            self._user_pattern_cache.pop(user_id, None)
            self._cache_timestamps.pop(user_id, None)
        else:
            self._user_pattern_cache.clear()
            self._cache_timestamps.clear()
