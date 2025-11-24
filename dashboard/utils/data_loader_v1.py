from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import pandas as pd

from sqlalchemy import text
from sqlalchemy.orm import Session

# Import from config
from config.database import session_scope
from config.settings import get_settings

logger = logging.getLogger(__name__)


# 이벤트명 한국어 매핑
EVENT_NAME_KOREAN = {
    'run_paraphrasing': '패러프레이징 실행',
    'selected_paraphrasing': '패러프레이징 선택',
    'editor_run_paraphrasing': '에디터 실행',
    'editor_selected_paraphrasing': '에디터 선택',
    'pageview_ad_inflow': '광고 유입',
    'tutorial_start': '튜토리얼 시작',
    'open_sidepanel': '사이드패널 열기',
    'tutorial_complete': '튜토리얼 완료',
    'copy_sentence': '문장 복사',
    'preview_paraphrasing': '미리보기',
    'toggle_field': '필드 전환',
    'change_tone': '톤 변경',
    'change_maintenance': '유지 설정',
    'change_language': '언어 변경',
}


def translate_event_name(event_name: str) -> str:
    """이벤트명을 한국어로 변환"""
    return EVENT_NAME_KOREAN.get(event_name, event_name)


class DataLoader:
    """
    Natural data flow manager for dashboard queries.
    
    Philosophy: Like a river gathering water from tributaries,
    this class collects data from various sources efficiently.
    """

    def __init__(self):
        self.settings = get_settings()
        logger.info("✅ DataLoader initialized - Natural flow begins")

    # ========================================
    # Temporal Flow Queries
    # ========================================

    def get_hourly_activity(
        self,
        days: int = 7,
        user_id: Optional[int] = None
    ) -> pd.DataFrame:
        """Get hourly activity pattern like sun moving across sky."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT
                    EXTRACT(HOUR FROM event_time) as hour,
                    COUNT(*) as count,
                    AVG(response_time_ms) as avg_response_time_ms,
                    COUNT(DISTINCT user_id) as unique_users
                FROM interactions
                WHERE event_time >= :cutoff_date
                    AND (:user_id IS NULL OR user_id = :user_id)
                GROUP BY EXTRACT(HOUR FROM event_time)
                ORDER BY hour
            """)
            
            with session_scope() as session:
                result = session.execute(
                    query,
                    {"cutoff_date": cutoff_date, "user_id": user_id}
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.debug(f"Loaded hourly activity: {len(df)} hours")
            return df
            
        except Exception as e:
            logger.error(f"Error loading hourly activity: {e}")
            return pd.DataFrame()

    def get_daily_rhythm(
        self,
        days: int = 30,
        event_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Get daily rhythm pattern like seasons changing."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT
                    DATE(event_time) as date,
                    event_name,
                    COUNT(*) as count,
                    COUNT(DISTINCT user_id) as unique_users
                FROM interactions
                WHERE event_time >= :cutoff_date
                    AND (:event_name IS NULL OR event_name = :event_name)
                GROUP BY DATE(event_time), event_name
                ORDER BY date, event_name
            """)
            
            with session_scope() as session:
                result = session.execute(
                    query,
                    {"cutoff_date": cutoff_date, "event_name": event_name}
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.debug(f"Loaded daily rhythm: {len(df)} days")
            return df
            
        except Exception as e:
            logger.error(f"Error loading daily rhythm: {e}")
            return pd.DataFrame()

    def get_time_of_day_distribution(
        self,
        days: int = 7
    ) -> Dict[str, int]:
        """Classify time into natural periods: Dawn, Morning, Afternoon, Evening, Night."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT
                    period,
                    COUNT(*) as count
                FROM (
                    SELECT
                        CASE
                            WHEN EXTRACT(HOUR FROM event_time) BETWEEN 5 AND 7 THEN '새벽'
                            WHEN EXTRACT(HOUR FROM event_time) BETWEEN 8 AND 11 THEN '오전'
                            WHEN EXTRACT(HOUR FROM event_time) BETWEEN 12 AND 17 THEN '오후'
                            WHEN EXTRACT(HOUR FROM event_time) BETWEEN 18 AND 21 THEN '저녁'
                            ELSE '밤'
                        END as period
                    FROM interactions
                    WHERE event_time >= :cutoff_date
                ) AS time_periods
                GROUP BY period
                ORDER BY
                    CASE period
                        WHEN '새벽' THEN 1
                        WHEN '오전' THEN 2
                        WHEN '오후' THEN 3
                        WHEN '저녁' THEN 4
                        ELSE 5
                    END
            """)
            
            with session_scope() as session:
                result = session.execute(query, {"cutoff_date": cutoff_date})
                distribution = {row.period: row.count for row in result.fetchall()}
            
            logger.debug(f"Time distribution: {distribution}")
            return distribution
            
        except Exception as e:
            logger.error(f"Error loading time distribution: {e}")
            return {}

    # ========================================
    # Soft Loop Queries
    # ========================================

    def get_repetition_patterns(
        self,
        user_id: int,
        window_size: int = 10,
        min_repetitions: int = 3
    ) -> pd.DataFrame:
        """Detect repetitive patterns like waves returning to shore."""
        try:
            query = text("""
                WITH user_events AS (
                    SELECT
                        event_name,
                        event_time,
                        LAG(event_name) OVER (ORDER BY event_time) as prev_event,
                        LAG(event_time) OVER (ORDER BY event_time) as prev_time
                    FROM interactions
                    WHERE user_id = :user_id
                    ORDER BY event_time DESC
                    LIMIT :window_size
                )
                SELECT
                    event_name,
                    COUNT(*) as repetitions,
                    'consecutive' as pattern_type
                FROM user_events
                WHERE event_name = prev_event
                GROUP BY event_name
                HAVING COUNT(*) >= :min_repetitions
            """)
            
            with session_scope() as session:
                result = session.execute(
                    query,
                    {
                        "user_id": user_id,
                        "window_size": window_size,
                        "min_repetitions": min_repetitions
                    }
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.debug(f"Detected {len(df)} repetition patterns")
            return df
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return pd.DataFrame()

    def get_diversity_score(
        self,
        user_id: int,
        days: int = 7
    ) -> float:
        """Calculate diversity score like ecosystem biodiversity."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                WITH event_counts AS (
                    SELECT
                        event_name,
                        COUNT(*) as count
                    FROM interactions
                    WHERE user_id = :user_id
                        AND event_time >= :cutoff_date
                    GROUP BY event_name
                ),
                total AS (
                    SELECT SUM(count) as total_count FROM event_counts
                )
                SELECT
                    -SUM((count::float / total_count) *
                         LN(count::float / total_count)) as diversity
                FROM event_counts, total
                WHERE total_count > 0
            """)
            
            with session_scope() as session:
                result = session.execute(
                    query,
                    {"user_id": user_id, "cutoff_date": cutoff_date}
                )
                diversity = result.scalar()
            
            # Normalize to 0-100 scale (Shannon entropy for 5 events is ~1.6)
            normalized = (diversity * 100 / 1.6) if diversity else 0.0
            normalized = min(100, max(0, normalized))  # Clamp to 0-100
            
            logger.debug(f"Diversity score: {normalized:.1f}")
            return normalized
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return 0.0

    # ========================================
    # Quiet Growth Queries
    # ========================================

    def get_user_growth_metrics(
        self,
        user_id: int
    ) -> Dict[str, Any]:
        """Get comprehensive growth metrics like plant growing over time."""
        try:
            query = text("""
                SELECT
                    u.distinct_id,
                    uf.total_paraphrases,
                    uf.last_7d_count,
                    uf.last_30d_count,
                    uf.avg_input_length,
                    uf.vocabulary_diversity,
                    uf.engagement_score,
                    uf.days_since_first_interaction,
                    uf.days_since_last_interaction,
                    u.total_sessions,
                    u.last_seen
                FROM users u
                LEFT JOIN user_features uf ON u.id = uf.user_id
                WHERE u.id = :user_id
            """)
            
            with session_scope() as session:
                result = session.execute(query, {"user_id": user_id})
                row = result.fetchone()
                
                if not row:
                    return {}
                
                return {
                    "distinct_id": row.distinct_id,
                    "total_activities": row.total_paraphrases or 0,
                    "active_days": row.last_30d_count or 0,
                    "avg_daily": (row.last_7d_count or 0) / 7.0,
                    "streak_days": max(0, 7 - (row.days_since_last_interaction or 7)),
                    "weekly_activity": row.last_7d_count or 0,
                    "monthly_activity": row.last_30d_count or 0,
                    "avg_length": row.avg_input_length or 0,
                    "vocabulary_richness": row.vocabulary_diversity or 0,
                    "engagement": row.engagement_score or 0,
                    "journey_days": row.days_since_first_interaction or 0,
                    "days_inactive": row.days_since_last_interaction or 0,
                    "total_sessions": row.total_sessions or 0,
                    "last_active": row.last_seen
                }
            
        except Exception as e:
            logger.error(f"Error loading growth metrics: {e}")
            return {}

    def get_growth_timeline(
        self,
        user_id: int,
        days: int = 90
    ) -> pd.DataFrame:
        """Get activity timeline like tree rings showing growth."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                WITH daily_counts AS (
                    SELECT
                        DATE(event_time) as date,
                        COUNT(*) as daily_count
                    FROM interactions
                    WHERE user_id = :user_id
                        AND event_time >= :cutoff_date
                    GROUP BY DATE(event_time)
                )
                SELECT
                    date,
                    daily_count,
                    SUM(daily_count) OVER (ORDER BY date) as cumulative_count
                FROM daily_counts
                ORDER BY date
            """)
            
            with session_scope() as session:
                result = session.execute(
                    query,
                    {"user_id": user_id, "cutoff_date": cutoff_date}
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.debug(f"Growth timeline: {len(df)} days")
            return df
            
        except Exception as e:
            logger.error(f"Error loading timeline: {e}")
            return pd.DataFrame()

    # ========================================
    # Context Echo Queries
    # ========================================

    def get_tone_distribution(
        self,
        days: int = 30,
        user_id: Optional[int] = None
    ) -> Dict[str, int]:
        """Get tone preference distribution like emotional landscape."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT
                    COALESCE(tone, 'neutral') as tone,
                    COUNT(*) as count
                FROM interactions
                WHERE event_time >= :cutoff_date
                    AND tone IS NOT NULL
                    AND (:user_id IS NULL OR user_id = :user_id)
                GROUP BY tone
                ORDER BY count DESC
            """)
            
            with session_scope() as session:
                result = session.execute(
                    query,
                    {"cutoff_date": cutoff_date, "user_id": user_id}
                )
                distribution = {row.tone: row.count for row in result.fetchall()}
            
            logger.debug(f"Tone distribution: {len(distribution)} tones")
            return distribution
            
        except Exception as e:
            logger.error(f"Error loading tone distribution: {e}")
            return {}

    def get_context_flow(
        self,
        user_id: int,
        limit: int = 20
    ) -> pd.DataFrame:
        """Get recent context flow like stream of consciousness."""
        try:
            query = text("""
                SELECT
                    event_time,
                    event_name,
                    tone,
                    maintenance,
                    target_language,
                    field,
                    input_sentence_length
                FROM interactions
                WHERE user_id = :user_id
                ORDER BY event_time DESC
                LIMIT :limit
            """)
            
            with session_scope() as session:
                result = session.execute(
                    query,
                    {"user_id": user_id, "limit": limit}
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.debug(f"Context flow: {len(df)} events")
            return df
            
        except Exception as e:
            logger.error(f"Error loading context flow: {e}")
            return pd.DataFrame()

    def get_event_distribution(
        self,
        days: int = 7,
        user_id: Optional[int] = None
    ) -> Dict[str, int]:
        """Get event distribution for diversity analysis (한국어)."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT
                    event_name,
                    COUNT(*) as count
                FROM interactions
                WHERE event_time >= :cutoff_date
                    AND (:user_id IS NULL OR user_id = :user_id)
                GROUP BY event_name
                ORDER BY count DESC
            """)
            
            with session_scope() as session:
                result = session.execute(
                    query,
                    {"cutoff_date": cutoff_date, "user_id": user_id}
                )
                # 이벤트명을 한국어로 변환
                distribution = {
                    translate_event_name(row.event_name): row.count 
                    for row in result.fetchall()
                }
            
            logger.debug(f"Event distribution: {len(distribution)} events")
            return distribution
            
        except Exception as e:
            logger.error(f"Error loading event distribution: {e}")
            return {}

    # ========================================
    # General Dashboard Queries
    # ========================================

    def get_kpi_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get key performance indicators like vital signs of ecosystem."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(DISTINCT DATE(event_time)) as active_days,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(DISTINCT event_name) as unique_events,
                    (SELECT COUNT(*) FROM users) as total_users
                FROM interactions
                WHERE event_time >= :cutoff_date
            """)
            
            with session_scope() as session:
                result = session.execute(query, {"cutoff_date": cutoff_date})
                row = result.fetchone()
                
                return {
                    "total_interactions": row.total_interactions or 0,
                    "active_users": row.active_users or 0,
                    "total_users": row.total_users or 0,
                    "active_days": row.active_days or 0,
                    "avg_response_time": row.avg_response_time or 0,
                    "unique_events": row.unique_events or 0,
                    "period_days": days
                }
            
        except Exception as e:
            logger.error(f"Error loading KPIs: {e}")
            return {}

    def get_top_users(
        self,
        limit: int = 10,
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """Get most active users like top of the forest canopy."""
        try:
            if days:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                where_clause = "WHERE i.event_time >= :cutoff_date"
            else:
                cutoff_date = None
                where_clause = ""
            
            query = text(f"""
                SELECT
                    u.id as user_id,
                    u.distinct_id,
                    u.email,
                    COUNT(i.id) as interaction_count,
                    MAX(i.event_time) as last_interaction
                FROM users u
                LEFT JOIN interactions i ON u.id = i.user_id
                {where_clause}
                GROUP BY u.id, u.distinct_id, u.email
                HAVING COUNT(i.id) > 0
                ORDER BY interaction_count DESC
                LIMIT :limit
            """)
            
            with session_scope() as session:
                if cutoff_date:
                    result = session.execute(
                        query,
                        {"cutoff_date": cutoff_date, "limit": limit}
                    )
                else:
                    result = session.execute(query, {"limit": limit})
                    
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.debug(f"Top users: {len(df)} users")
            return df
            
        except Exception as e:
            logger.error(f"Error loading top users: {e}")
            return pd.DataFrame()

    def get_all_users_with_activity(self) -> pd.DataFrame:
        """Get all users with their total activity count for segmentation."""
        try:
            query = text("""
                SELECT
                    u.id as user_id,
                    u.distinct_id,
                    COUNT(i.id) as interaction_count
                FROM users u
                LEFT JOIN interactions i ON u.id = i.user_id
                GROUP BY u.id, u.distinct_id
                ORDER BY interaction_count DESC
            """)
            
            with session_scope() as session:
                result = session.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.debug(f"All users with activity: {len(df)} users")
            return df
            
        except Exception as e:
            logger.error(f"Error loading all users: {e}")
            return pd.DataFrame()

    # ========================================
    # Utility Methods
    # ========================================

    @lru_cache(maxsize=32)
    def get_available_users(self) -> List[Tuple[int, str]]:
        """Get list of users for dropdown filters. Cached to reduce database load."""
        try:
            query = text("""
                SELECT id, distinct_id
                FROM users
                WHERE last_seen >= NOW() - INTERVAL '30 days'
                ORDER BY last_seen DESC
                LIMIT 100
            """)
            
            with session_scope() as session:
                result = session.execute(query)
                users = [(row.id, row.distinct_id) for row in result.fetchall()]
            
            return users
            
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return []

    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            with session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Singleton instance
_data_loader_instance: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    """Get singleton DataLoader instance."""
    global _data_loader_instance
    if _data_loader_instance is None:
        _data_loader_instance = DataLoader()
    return _data_loader_instance


if __name__ == "__main__":
    # Test data loader
    loader = get_data_loader()
    
    print("Testing DataLoader...")
    print(f"Health check: {'✅ Healthy' if loader.health_check() else '❌ Failed'}")
    
    kpis = loader.get_kpi_summary(days=7)
    print(f"\n📊 KPIs (last 7 days):")
    for key, value in kpis.items():
        print(f"  {key}: {value}")
    
    hourly = loader.get_hourly_activity(days=7)
    print(f"\n⏰ Hourly activity shape: {hourly.shape}")
    if not hourly.empty:
        print(hourly.head())