from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, text

from src.database.models import Interaction, User, Item, InteractionFeature

logger = logging.getLogger(__name__)


class InteractionFeatureExtractor:
    """
    Extract and compute interaction sequence features.
    
    Features computed:
    - Sequential patterns (Soft Loop 지원)
    - Temporal gaps (Temporal Flow 지원)
    - Context switches (다양성 분석)
    - Session characteristics (Ghost Preview 지원)
    - 실제 통계 기반 파워유저/신규유저 구분
    """
    
    def __init__(self, session: Session):
        """
        Initialize feature extractor.
        
        Args:
            session: Database session
        """
        self.session = session
        self.logger = logging.getLogger(__name__)
    
    def extract_session_features(
        self,
        user_id: int,
        session_gap_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Extract features for user sessions.
        
        Args:
            user_id: User database ID
            session_gap_minutes: Minutes of inactivity to define new session
        
        Returns:
            List of session feature dictionaries
        
        Example:
            >>> extractor = InteractionFeatureExtractor(session)
            >>> sessions = extractor.extract_session_features(user_id=123)
        """
        # Get user interactions ordered by time
        interactions = self.session.query(Interaction).filter(
            Interaction.user_id == user_id
        ).order_by(Interaction.event_time).all()
        
        if not interactions:
            self.logger.warning(f"No interactions found for user {user_id}")
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'id': i.id,
            'event_name': i.event_name,
            'event_time': i.event_time,
            'item_id': i.item_id,
            'device_id': i.device_id,
            'browser': i.browser,
            'os': i.os,
            'tone': i.tone,
            'maintenance': i.maintenance,
            'response_time_ms': i.response_time_ms,
        } for i in interactions])
        
        # Identify sessions based on time gaps
        df['event_time'] = pd.to_datetime(df['event_time'], utc=True)
        df['time_gap_minutes'] = df['event_time'].diff().dt.total_seconds() / 60.0
        df['session_id'] = (df['time_gap_minutes'] > session_gap_minutes).cumsum()
        
        # Extract features for each session
        session_features = []
        
        for session_id, session_df in df.groupby('session_id'):
            features = self._compute_single_session_features(session_df, session_id, user_id)
            session_features.append(features)
        
        self.logger.debug(f"Extracted features for {len(session_features)} sessions (user {user_id})")
        return session_features
    
    def extract_batch_session_features(
        self,
        user_ids: List[int],
        session_gap_minutes: int = 30
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract session features for multiple users in batch.
        
        Args:
            user_ids: List of user database IDs
            session_gap_minutes: Session gap threshold
        
        Returns:
            Dictionary mapping user_id to list of session features
        """
        results = {}
        total = len(user_ids)
        
        for idx, user_id in enumerate(user_ids, 1):
            try:
                sessions = self.extract_session_features(user_id, session_gap_minutes)
                if sessions:
                    results[user_id] = sessions
                
                # Progress log
                if idx % 100 == 0 or idx == total:
                    self.logger.info(f"Progress: {idx}/{total} users processed")
                    
            except Exception as e:
                self.logger.error(f"Failed to extract session features for user {user_id}: {e}")
        
        self.logger.info(f"✅ Extracted session features for {len(results)}/{total} users")
        return results
    
    def extract_sequence_patterns(
        self,
        user_id: int,
        max_sequence_length: int = 5,
        min_frequency: int = 2
    ) -> List[Tuple[List[str], int]]:
        """
        Extract common event sequences for a user (Soft Loop 지원).
        
        Args:
            user_id: User database ID
            max_sequence_length: Maximum sequence length to extract
            min_frequency: Minimum frequency to consider as pattern
        
        Returns:
            List of (sequence, frequency) tuples
        
        Example:
            >>> extractor = InteractionFeatureExtractor(session)
            >>> patterns = extractor.extract_sequence_patterns(user_id=123, max_sequence_length=3)
        """
        interactions = self.session.query(Interaction).filter(
            Interaction.user_id == user_id
        ).order_by(Interaction.event_time).all()
        
        if not interactions:
            return []
        
        # Extract event names
        events = [i.event_name for i in interactions]
        
        # Generate sequences
        sequences = []
        for i in range(len(events)):
            for length in range(2, min(max_sequence_length + 1, len(events) - i + 1)):
                sequence = tuple(events[i:i + length])
                sequences.append(sequence)
        
        # Count frequencies
        sequence_counts = Counter(sequences)
        
        # Filter by minimum frequency and sort
        frequent_patterns = [
            (list(seq), count) 
            for seq, count in sequence_counts.items() 
            if count >= min_frequency
        ]
        frequent_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return frequent_patterns
    
    def compute_temporal_gaps(
        self,
        user_id: int,
        window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compute temporal gap statistics for a user.
        
        Args:
            user_id: User database ID
            window_days: Number of days to look back
        
        Returns:
            Dictionary of gap statistics
        
        Example:
            >>> extractor = InteractionFeatureExtractor(session)
            >>> gaps = extractor.compute_temporal_gaps(user_id=123)
        """
        cutoff_date = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=window_days)
        
        interactions = self.session.query(Interaction).filter(
            and_(
                Interaction.user_id == user_id,
                Interaction.event_time >= cutoff_date
            )
        ).order_by(Interaction.event_time).all()
        
        if len(interactions) < 2:
            return self._get_default_gap_features()
        
        # Compute gaps
        event_times = [i.event_time for i in interactions]
        gaps_minutes = [
            (event_times[i+1] - event_times[i]).total_seconds() / 60.0 
            for i in range(len(event_times) - 1)
        ]
        
        if not gaps_minutes:
            return self._get_default_gap_features()
        
        return {
            'avg_gap_minutes': float(np.mean(gaps_minutes)),
            'median_gap_minutes': float(np.median(gaps_minutes)),
            'min_gap_minutes': float(np.min(gaps_minutes)),
            'max_gap_minutes': float(np.max(gaps_minutes)),
            'std_gap_minutes': float(np.std(gaps_minutes)),
            'total_gaps': len(gaps_minutes),
        }
    
    def compute_context_switches(
        self,
        user_id: int,
        window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compute context switch frequency (device, tone, etc.).
        
        Args:
            user_id: User database ID
            window_days: Number of days to look back
        
        Returns:
            Dictionary of context switch metrics
        
        Example:
            >>> extractor = InteractionFeatureExtractor(session)
            >>> switches = extractor.compute_context_switches(user_id=123)
        """
        cutoff_date = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=window_days)
        
        interactions = self.session.query(Interaction).filter(
            and_(
                Interaction.user_id == user_id,
                Interaction.event_time >= cutoff_date
            )
        ).order_by(Interaction.event_time).all()
        
        if len(interactions) < 2:
            return self._get_default_switch_features()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'device_id': i.device_id,
            'browser': i.browser,
            'os': i.os,
            'tone': i.tone,
            'maintenance': i.maintenance,
        } for i in interactions])
        
        # Count switches
        device_switches = (df['device_id'] != df['device_id'].shift()).sum() - 1
        browser_switches = (df['browser'] != df['browser'].shift()).sum() - 1
        os_switches = (df['os'] != df['os'].shift()).sum() - 1
        tone_switches = (df['tone'] != df['tone'].shift()).sum() - 1
        maintenance_switches = (df['maintenance'] != df['maintenance'].shift()).sum() - 1
        
        total_interactions = len(interactions)
        
        # Tone diversity (Context Echo 지원)
        tone_diversity = df['tone'].nunique()
        
        return {
            'device_switch_rate': float(device_switches / max(total_interactions - 1, 1)),
            'browser_switch_rate': float(browser_switches / max(total_interactions - 1, 1)),
            'os_switch_rate': float(os_switches / max(total_interactions - 1, 1)),
            'tone_switch_rate': float(tone_switches / max(total_interactions - 1, 1)),
            'maintenance_switch_rate': float(maintenance_switches / max(total_interactions - 1, 1)),
            'total_device_switches': int(device_switches),
            'total_browser_switches': int(browser_switches),
            'total_os_switches': int(os_switches),
            'total_tone_switches': int(tone_switches),
            'total_maintenance_switches': int(maintenance_switches),
            'tone_diversity': int(tone_diversity),
        }
    
    def classify_user_type(self, user_id: int) -> Dict[str, Any]:
        """
        실제 통계 기반 사용자 분류.
        
        통계 기반:
        - 파워유저: 72명 (3.3%) - 평균 92.6회/유저
        - 성장 유저: 116명 (5.3%) - 15-49회
        - 신규 유저: 2,010명 (91.4%) - 1-14회
        
        Args:
            user_id: User database ID
        
        Returns:
            Dictionary with user type classification
        """
        total_interactions = self.session.query(func.count(Interaction.id)).filter(
            Interaction.user_id == user_id
        ).scalar() or 0
        
        # 실제 통계 기반 분류
        if total_interactions >= 50:
            user_type = 'power'  # 파워유저 (3.3%)
            recommendation_strategy = 'soft_loop'  # Soft Loop 적용
        elif total_interactions >= 15:
            user_type = 'growth'  # 성장 유저 (5.3%)
            recommendation_strategy = 'context_echo'  # Context Echo로 다양성 유도
        elif total_interactions >= 1:
            user_type = 'casual'  # 일반 유저
            recommendation_strategy = 'ghost_preview'  # Ghost Preview로 탐색 유도
        else:
            user_type = 'new'  # 신규 유저 (91.4%)
            recommendation_strategy = 'ambient'  # Ambient Recommendation
        
        return {
            'user_type': user_type,
            'total_interactions': total_interactions,
            'recommendation_strategy': recommendation_strategy,
        }
    
    def _compute_single_session_features(
        self,
        session_df: pd.DataFrame,
        session_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Compute features for a single session."""
        # Basic session info
        session_start = session_df['event_time'].min()
        session_end = session_df['event_time'].max()
        session_duration_minutes = (session_end - session_start).total_seconds() / 60.0
        
        # Temporal Flow: 시간대 분류
        hour_of_day = session_start.hour
        time_of_day = self._classify_time_of_day(hour_of_day)
        is_peak_hours = 1 if hour_of_day in [7, 8] else 0  # 통계 기반 피크 시간
        
        # Event counts
        total_events = len(session_df)
        unique_items = session_df['item_id'].nunique()
        
        # Event type distribution (Ghost Preview 지원)
        event_type_counts = session_df['event_name'].value_counts().to_dict()
        run_count = event_type_counts.get('run_paraphrasing', 0) + event_type_counts.get('editor_run_paraphrasing', 0)
        select_count = event_type_counts.get('selected_paraphrasing', 0) + event_type_counts.get('editor_selected_paraphrasing', 0)
        copy_count = event_type_counts.get('copy_sentence', 0)
        
        # Ghost Preview Metrics
        preview_to_select_ratio = select_count / max(run_count, 1)
        
        # Device/tone consistency (Soft Loop 지원)
        device_changes = (session_df['device_id'] != session_df['device_id'].shift()).sum() - 1
        browser_changes = (session_df['browser'] != session_df['browser'].shift()).sum() - 1
        tone_changes = (session_df['tone'] != session_df['tone'].shift()).sum() - 1
        maintenance_changes = (session_df['maintenance'] != session_df['maintenance'].shift()).sum() - 1
        
        # Tone diversity (Context Echo 지원)
        tone_diversity = session_df['tone'].nunique()
        
        # Response time stats
        avg_response_time = session_df['response_time_ms'].mean() if not session_df['response_time_ms'].isna().all() else None
        
        # Exploration score (다양성)
        exploration_score = (unique_items / max(total_events, 1)) * (tone_diversity / 4.0)
        
        # Repeat pattern score (반복성)
        event_sequence = session_df['event_name'].tolist()
        repeat_pattern_score = self._compute_repeat_score(event_sequence)
        
        # Sequence diversity
        unique_sequences = len(set(tuple(event_sequence[i:i+3]) for i in range(len(event_sequence) - 2)))
        sequence_diversity = unique_sequences / max(len(event_sequence) - 2, 1)
        
        # Temporal gaps within session
        gaps_minutes = session_df['time_gap_minutes'].dropna().tolist()
        avg_gap_minutes = float(np.mean(gaps_minutes)) if gaps_minutes else None
        median_gap_minutes = float(np.median(gaps_minutes)) if gaps_minutes else None
        max_gap_minutes = float(np.max(gaps_minutes)) if gaps_minutes else None
        
        # Common sequences
        common_sequences = self._extract_common_sequences(event_sequence)
        
        return {
            'user_id': user_id,
            'session_id': f"{user_id}_session_{session_id}",
            'session_start': session_start,
            'session_end': session_end,
            'session_duration_minutes': float(session_duration_minutes),
            'hour_of_day': int(hour_of_day),
            'time_of_day': time_of_day,
            'is_peak_hours': is_peak_hours,
            'total_events': int(total_events),
            'unique_items': int(unique_items),
            'run_count': int(run_count),
            'select_count': int(select_count),
            'copy_count': int(copy_count),
            'event_type_distribution': event_type_counts,
            'device_changes': int(device_changes),
            'tone_changes': int(tone_changes),
            'maintenance_changes': int(maintenance_changes),
            'tone_diversity': int(tone_diversity),
            'avg_response_time_ms': float(avg_response_time) if avg_response_time else None,
            'events_per_minute': float(total_events / max(session_duration_minutes, 1)),
            'preview_to_select_ratio': float(preview_to_select_ratio),
            'exploration_score': float(exploration_score),
            'repeat_pattern_score': float(repeat_pattern_score),
            'sequence_diversity': float(sequence_diversity),
            'avg_gap_minutes': avg_gap_minutes,
            'median_gap_minutes': median_gap_minutes,
            'max_gap_minutes': max_gap_minutes,
            'common_sequences': common_sequences,
        }
    
    def _classify_time_of_day(self, hour: int) -> str:
        """실제 통계 기반 시간대 분류"""
        if 0 <= hour < 6:
            return 'dawn'      # 새벽 (33.8% 트래픽)
        elif 6 <= hour < 12:
            return 'morning'   # 오전 (41.6% 트래픽)
        elif 12 <= hour < 18:
            return 'afternoon' # 오후 (17.5% 트래픽)
        elif 18 <= hour < 22:
            return 'evening'   # 저녁 (7.1% 트래픽)
        else:
            return 'night'
    
    def _compute_repeat_score(self, event_sequence: List[str]) -> float:
        """
        반복 패턴 점수 계산 (Soft Loop 지원).
        
        높은 점수 = 반복이 많음 → Soft Loop 적용 필요
        """
        if len(event_sequence) < 3:
            return 0.0
        
        # 3-gram 반복 비율
        trigrams = [tuple(event_sequence[i:i+3]) for i in range(len(event_sequence) - 2)]
        trigram_counts = Counter(trigrams)
        
        # 가장 많이 반복된 패턴의 비율
        if not trigram_counts:
            return 0.0
        
        max_repeat = max(trigram_counts.values())
        total_trigrams = len(trigrams)
        
        return max_repeat / max(total_trigrams, 1)
    
    def _extract_common_sequences(self, event_sequence: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """자주 나타나는 시퀀스 추출"""
        if len(event_sequence) < 2:
            return []
        
        # 2-gram, 3-gram 추출
        sequences = []
        for length in [2, 3]:
            for i in range(len(event_sequence) - length + 1):
                seq = tuple(event_sequence[i:i+length])
                sequences.append(seq)
        
        # 빈도 계산
        sequence_counts = Counter(sequences)
        
        # Top-K 반환
        common = [
            {'sequence': list(seq), 'count': count}
            for seq, count in sequence_counts.most_common(top_k)
        ]
        
        return common
    
    def _get_default_gap_features(self) -> Dict[str, Any]:
        """Get default gap feature values."""
        return {
            'avg_gap_minutes': 0.0,
            'median_gap_minutes': 0.0,
            'min_gap_minutes': 0.0,
            'max_gap_minutes': 0.0,
            'std_gap_minutes': 0.0,
            'total_gaps': 0,
        }
    
    def _get_default_switch_features(self) -> Dict[str, Any]:
        """Get default switch feature values."""
        return {
            'device_switch_rate': 0.0,
            'browser_switch_rate': 0.0,
            'os_switch_rate': 0.0,
            'tone_switch_rate': 0.0,
            'maintenance_switch_rate': 0.0,
            'total_device_switches': 0,
            'total_browser_switches': 0,
            'total_os_switches': 0,
            'total_tone_switches': 0,
            'total_maintenance_switches': 0,
            'tone_diversity': 0,
        }


# =========================================================
# DB Save Function
# =========================================================

def save_features_to_db(
    session: Session, 
    features: Dict[int, List[Dict[str, Any]]]
) -> int:
    """
    Save extracted interaction features to database.
    
    Args:
        session: Database session
        features: Dictionary mapping user_id to list of session features
    
    Returns:
        Number of records saved
    """
    # 테이블 존재 확인
    try:
        session.execute(text("SELECT 1 FROM interaction_features LIMIT 1"))
    except Exception as e:
        logger.error(
            "interaction_features table does not exist! "
            "Please run: python scripts/create_interaction_features_table.py"
        )
        raise
    
    saved_count = 0
    
    for user_id, session_list in features.items():
        for session_features in session_list:
            try:
                # Check if session already exists
                existing = session.query(InteractionFeature).filter_by(
                    session_id=session_features['session_id']
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in session_features.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    logger.debug(f"Updated session {session_features['session_id']}")
                else:
                    # Create new record
                    interaction_feature = InteractionFeature(**session_features)
                    session.add(interaction_feature)
                    logger.debug(f"Created session {session_features['session_id']}")
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Failed to save session {session_features.get('session_id')}: {e}")
                continue
    
    session.commit()
    logger.info(f"✅ Saved {saved_count} interaction session features to database")
    return saved_count


# =========================================================
# Main Demo
# =========================================================

if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres
    
    setup_logging()
    
    pg = get_postgres()
    
    with pg.transaction() as session:
        extractor = InteractionFeatureExtractor(session)
        
        # Get users with interactions
        users_with_interactions = session.query(
            User.id, User.distinct_id, func.count(Interaction.id).label('interaction_count')
        ).join(
            Interaction, User.id == Interaction.user_id
        ).group_by(
            User.id, User.distinct_id
        ).order_by(
            func.count(Interaction.id).desc()
        ).limit(5).all()
        
        print(f"\n{'='*60}")
        print(f"📊 Top 5 Users with Most Interactions")
        print(f"{'='*60}\n")
        
        for idx, (user_id, distinct_id, count) in enumerate(users_with_interactions, 1):
            print(f"{idx}. User {user_id} ({distinct_id[:20]}...): {count} events")
        
        if users_with_interactions:
            first_user_id = users_with_interactions[0][0]
            
            print(f"\n{'='*60}")
            print(f"🎯 Analyzing User {first_user_id}")
            print(f"{'='*60}\n")
            
            # 1. User type classification
            user_type_info = extractor.classify_user_type(first_user_id)
            print(f"👤 User Type:")
            for key, value in user_type_info.items():
                print(f"   {key}: {value}")
            
            # 2. Session features
            print(f"\n📊 Session Features:")
            sessions = extractor.extract_session_features(first_user_id)
            print(f"   Total sessions: {len(sessions)}")
            if sessions:
                print(f"\n   First session details:")
                for key, value in list(sessions[0].items())[:10]:
                    print(f"      {key}: {value}")
            
            # 3. Sequence patterns
            print(f"\n🔁 Sequence Patterns (Soft Loop):")
            patterns = extractor.extract_sequence_patterns(first_user_id, max_sequence_length=3)
            for i, (seq, freq) in enumerate(patterns[:5], 1):
                print(f"   {i}. {' → '.join(seq)} (x{freq})")
            
            # 4. Temporal gaps
            print(f"\n⏱️  Temporal Gaps:")
            gaps = extractor.compute_temporal_gaps(first_user_id)
            for key, value in gaps.items():
                print(f"   {key}: {value}")
            
            # 5. Context switches
            print(f"\n🔄 Context Switches:")
            switches = extractor.compute_context_switches(first_user_id)
            for key, value in switches.items():
                print(f"   {key}: {value}")
            
            # 6. Batch extraction & save
            print(f"\n{'='*60}")
            print(f"💾 Saving features to database...")
            print(f"{'='*60}\n")
            
            user_ids = [uid for uid, _, _ in users_with_interactions]
            batch_features = extractor.extract_batch_session_features(user_ids)
            
            saved_count = save_features_to_db(session, batch_features)
            
            print(f"\n✅ Completed!")
            print(f"{'='*60}")
            print(f"   Total users processed: {len(batch_features)}")
            print(f"   Total sessions saved: {saved_count}")
            print(f"{'='*60}\n")