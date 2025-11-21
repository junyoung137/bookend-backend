"""
Interaction Repository for Bookend Recommendation System.

Handles all interaction/event-related database operations:
1. Event tracking and creation
2. User activity analysis
3. Temporal pattern detection
4. Interaction matrix generation
5. Session management
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import Session

from ..models import Interaction, User, Item
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class InteractionRepository(BaseRepository[Interaction]):
    """
    Repository for Interaction model with specialized queries.
    
    Follows principles:
    - Single Responsibility: Only interaction-related queries
    - One Source of Truth: All interaction queries go through this class
    - Error Handling: Proper exception handling and logging
    - Modularity: Each function does one thing well
    """
    
    def get_repository_name(self) -> str:
        """Get repository name for logging."""
        return "InteractionRepository"
    
    # ========================================
    # Single Interaction Queries
    # ========================================
    
    def get_by_insert_id(self, insert_id: str) -> Optional[Interaction]:
        """
        Get interaction by unique insert_id (from Mixpanel).
        
        Args:
            insert_id: Unique event identifier
        
        Returns:
            Interaction instance or None
        
        Example:
            >>> event = repo.get_by_insert_id("4abhlvqowv8x6uwl")
        """
        if not insert_id:
            self.logger.warning("insert_id is empty")
            return None
        
        try:
            return self.get_by_field("insert_id", insert_id)
        except Exception as e:
            self.logger.error(f"Error fetching interaction by insert_id: {e}")
            return None
    
    # ========================================
    # Interaction Creation
    # ========================================
    
    def create_from_event(self, event_data: Dict[str, Any]) -> Optional[Interaction]:
        """
        Create interaction from raw event data (Mixpanel format).
        
        One Source of Truth: This is the only way to create interactions from events.
        
        Args:
            event_data: Event dictionary from Mixpanel
                - event_name: Event type
                - distinct_id: User identifier
                - time: Unix timestamp
                - insert_id: Unique event ID
                - properties: Event properties dict
        
        Returns:
            Created interaction or None
        
        Example:
            >>> interaction = repo.create_from_event({
            ...     "event_name": "run_paraphrasing",
            ...     "distinct_id": "user123",
            ...     "time": 1759462545,
            ...     "insert_id": "abc123",
            ...     "properties": {...}
            ... })
        """
        if not event_data:
            self.logger.error("event_data is empty")
            return None
        
        try:
            # Extract required fields
            event_name = event_data.get("event_name")
            insert_id = event_data.get("insert_id")
            time = event_data.get("time")
            
            if not event_name or not insert_id or not time:
                self.logger.error("Missing required event fields")
                return None
            
            # Check if event already exists (idempotency)
            existing = self.get_by_insert_id(insert_id)
            if existing:
                self.logger.debug(f"Interaction already exists: {insert_id}")
                return existing
            
            # Extract properties
            props = event_data.get("properties", {})
            
            # Build interaction data
            interaction_data = self._extract_interaction_data(
                event_data,
                props
            )
            
            # Create interaction
            interaction = self.create(**interaction_data)
            if interaction:
                self.logger.info(f"Created interaction: {insert_id}")
            
            return interaction
            
        except Exception as e:
            self.logger.error(f"Error creating interaction from event: {e}")
            return None
    
    def _extract_interaction_data(
        self,
        event_data: Dict[str, Any],
        props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Helper: Extract and transform interaction data from event.
        
        Single Responsibility: Data extraction logic.
        """
        return {
            # Core fields
            "event_name": event_data.get("event_name"),
            "insert_id": event_data.get("insert_id"),
            "event_time": datetime.fromtimestamp(event_data.get("time", 0)),
            "device_id": event_data.get("device_id"),
            
            # Device info
            "browser": props.get("browser"),
            "os": props.get("os"),
            "current_url": props.get("current_url"),
            
            # Interaction specifics
            "field": props.get("field"),
            "tone": props.get("tone"),
            "maintenance": props.get("maintenance"),
            "target_language": props.get("target_language"),
            
            # Performance metrics
            "response_time_ms": self._extract_response_time(props),
            "input_sentence_length": props.get("input_sentence_length"),
            
            # LLM information
            "llm_provider": self._extract_llm_provider(props),
            "llm_name": self._extract_llm_name(props),
            "llm_version": self._extract_llm_version(props),
            
            # UI context
            "position": props.get("position"),
            "trigger": props.get("trigger"),
            
            # Store all properties as JSONB
            "properties": props
        }
    
    def _extract_response_time(self, props: Dict[str, Any]) -> Optional[int]:
        """Helper: Extract response time from properties."""
        return props.get("response_time_ms") or props.get("response_time_seconds")
    
    def _extract_llm_provider(self, props: Dict[str, Any]) -> Optional[str]:
        """Helper: Extract LLM provider from properties."""
        return props.get("llm_provider") or props.get("llm_info_provider")
    
    def _extract_llm_name(self, props: Dict[str, Any]) -> Optional[str]:
        """Helper: Extract LLM name from properties."""
        return props.get("llm_name") or props.get("llm_info_name")
    
    def _extract_llm_version(self, props: Dict[str, Any]) -> Optional[str]:
        """Helper: Extract LLM version from properties."""
        return props.get("llm_version") or props.get("llm_info_version")
    
    # ========================================
    # User Interaction Queries
    # ========================================
    
    def get_user_interactions(
        self,
        user_id: int,
        limit: Optional[int] = None,
        event_name: Optional[str] = None,
        days: Optional[int] = None
    ) -> List[Interaction]:
        """
        Get interactions for a specific user.
        
        Args:
            user_id: User ID
            limit: Maximum number of interactions
            event_name: Filter by specific event type
            days: Only return interactions from last N days
        
        Returns:
            List of interactions ordered by time (newest first)
        
        Example:
            >>> events = repo.get_user_interactions(
            ...     user_id=123,
            ...     limit=50,
            ...     event_name="run_paraphrasing",
            ...     days=7
            ... )
        """
        if not user_id or user_id <= 0:
            self.logger.warning(f"Invalid user_id: {user_id}")
            return []
        
        try:
            conditions = [Interaction.user_id == user_id]
            
            # Add event_name filter
            if event_name:
                conditions.append(Interaction.event_name == event_name)
            
            # Add time filter
            if days and days > 0:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                conditions.append(Interaction.event_time >= cutoff_date)
            
            stmt = select(Interaction).where(
                and_(*conditions)
            ).order_by(desc(Interaction.event_time))
            
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            interactions = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(interactions)} interactions for user {user_id}")
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error fetching user interactions: {e}")
            return []
    
    def get_user_event_sequence(
        self,
        user_id: int,
        limit: int = 50
    ) -> List[Interaction]:
        """
        Get chronological event sequence for user (oldest first).
        
        Args:
            user_id: User ID
            limit: Maximum events (default: 50)
        
        Returns:
            List of interactions in chronological order
        
        Example:
            >>> sequence = repo.get_user_event_sequence(user_id=123, limit=20)
        """
        if not user_id or user_id <= 0:
            self.logger.warning(f"Invalid user_id: {user_id}")
            return []
        
        if limit <= 0:
            limit = 50
        
        try:
            stmt = select(Interaction).where(
                Interaction.user_id == user_id
            ).order_by(asc(Interaction.event_time)).limit(limit)
            
            result = self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            self.logger.error(f"Error fetching event sequence: {e}")
            return []
    
    # ========================================
    # Time-based Queries
    # ========================================
    
    def get_recent_interactions(
        self,
        hours: int = 24,
        limit: Optional[int] = None,
        event_name: Optional[str] = None
    ) -> List[Interaction]:
        """
        Get recent interactions within time window.
        
        Args:
            hours: Time window in hours (default: 24)
            limit: Maximum number of interactions
            event_name: Optional event type filter
        
        Returns:
            List of recent interactions
        
        Example:
            >>> recent = repo.get_recent_interactions(hours=1, limit=100)
        """
        if hours <= 0:
            hours = 24
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            conditions = [Interaction.event_time >= cutoff_time]
            if event_name:
                conditions.append(Interaction.event_name == event_name)
            
            stmt = select(Interaction).where(
                and_(*conditions)
            ).order_by(desc(Interaction.event_time))
            
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            interactions = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(interactions)} recent interactions")
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error fetching recent interactions: {e}")
            return []
    
    # ========================================
    # Event Distribution and Statistics
    # ========================================
    
    def get_event_distribution(
        self,
        days: int = 7,
        user_id: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Get distribution of event types.
        
        Args:
            days: Time window in days (default: 7)
            user_id: Optional user filter
        
        Returns:
            Dictionary mapping event_name to count
        
        Example:
            >>> dist = repo.get_event_distribution(days=7)
            >>> print(f"Runs: {dist.get('run_paraphrasing', 0)}")
        """
        if days <= 0:
            days = 7
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            conditions = [Interaction.event_time >= cutoff_date]
            if user_id and user_id > 0:
                conditions.append(Interaction.user_id == user_id)
            
            stmt = select(
                Interaction.event_name,
                func.count().label('count')
            ).where(
                and_(*conditions)
            ).group_by(Interaction.event_name)
            
            result = self.session.execute(stmt)
            distribution = {row.event_name: row.count for row in result.all()}
            
            self.logger.debug(f"Event distribution: {distribution}")
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error getting event distribution: {e}")
            return {}
    
    def get_hourly_activity(
        self,
        user_id: Optional[int] = None,
        days: int = 7
    ) -> Dict[int, int]:
        """
        Get activity distribution by hour of day (0-23).
        
        Args:
            user_id: Optional user ID filter
            days: Time window in days (default: 7)
        
        Returns:
            Dictionary mapping hour (0-23) to count
        
        Example:
            >>> activity = repo.get_hourly_activity(user_id=123, days=30)
            >>> peak_hour = max(activity, key=activity.get)
        """
        if days <= 0:
            days = 7
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            conditions = [Interaction.event_time >= cutoff_date]
            if user_id and user_id > 0:
                conditions.append(Interaction.user_id == user_id)
            
            stmt = select(
                func.extract('hour', Interaction.event_time).label('hour'),
                func.count().label('count')
            ).where(
                and_(*conditions)
            ).group_by('hour')
            
            result = self.session.execute(stmt)
            activity = {int(row.hour): row.count for row in result.all()}
            
            self.logger.debug(f"Hourly activity: {len(activity)} hours with data")
            return activity
            
        except Exception as e:
            self.logger.error(f"Error getting hourly activity: {e}")
            return {}
    
    def get_avg_response_time(
        self,
        user_id: Optional[int] = None,
        event_name: Optional[str] = None,
        days: int = 7
    ) -> Optional[float]:
        """
        Get average response time in milliseconds.
        
        Args:
            user_id: Optional user filter
            event_name: Optional event filter
            days: Time window (default: 7)
        
        Returns:
            Average response time in milliseconds or None
        
        Example:
            >>> avg_time = repo.get_avg_response_time(
            ...     event_name="run_paraphrasing",
            ...     days=7
            ... )
        """
        if days <= 0:
            days = 7
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            conditions = [
                Interaction.event_time >= cutoff_date,
                Interaction.response_time_ms.isnot(None)
            ]
            
            if user_id and user_id > 0:
                conditions.append(Interaction.user_id == user_id)
            if event_name:
                conditions.append(Interaction.event_name == event_name)
            
            stmt = select(
                func.avg(Interaction.response_time_ms)
            ).where(and_(*conditions))
            
            result = self.session.execute(stmt)
            avg_time = result.scalar()
            
            if avg_time:
                self.logger.debug(f"Average response time: {avg_time:.2f}ms")
            
            return avg_time
            
        except Exception as e:
            self.logger.error(f"Error getting avg response time: {e}")
            return None
    
    # ========================================
    # Device and Context Analytics
    # ========================================
    
    def get_device_distribution(
        self,
        user_id: Optional[int] = None,
        days: int = 30
    ) -> Dict[str, int]:
        """
        Get device/browser distribution.
        
        Args:
            user_id: Optional user filter
            days: Time window (default: 30)
        
        Returns:
            Dictionary mapping browser to count
        """
        if days <= 0:
            days = 30
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            conditions = [
                Interaction.event_time >= cutoff_date,
                Interaction.browser.isnot(None)
            ]
            
            if user_id and user_id > 0:
                conditions.append(Interaction.user_id == user_id)
            
            stmt = select(
                Interaction.browser,
                func.count().label('count')
            ).where(
                and_(*conditions)
            ).group_by(Interaction.browser)
            
            result = self.session.execute(stmt)
            distribution = {row.browser: row.count for row in result.all()}
            
            self.logger.debug(f"Device distribution: {distribution}")
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error getting device distribution: {e}")
            return {}
    
    def get_tone_distribution(
        self,
        user_id: Optional[int] = None,
        days: int = 30
    ) -> Dict[str, int]:
        """
        Get tone preference distribution.
        
        Args:
            user_id: Optional user filter
            days: Time window (default: 30)
        
        Returns:
            Dictionary mapping tone to count
        """
        if days <= 0:
            days = 30
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            conditions = [
                Interaction.event_time >= cutoff_date,
                Interaction.tone.isnot(None)
            ]
            
            if user_id and user_id > 0:
                conditions.append(Interaction.user_id == user_id)
            
            stmt = select(
                Interaction.tone,
                func.count().label('count')
            ).where(
                and_(*conditions)
            ).group_by(Interaction.tone)
            
            result = self.session.execute(stmt)
            distribution = {row.tone: row.count for row in result.all()}
            
            self.logger.debug(f"Tone distribution: {distribution}")
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error getting tone distribution: {e}")
            return {}
    
    # ========================================
    # Interaction Matrix for Recommendations
    # ========================================
    
    def get_user_item_interactions(
        self,
        user_id: int,
        days: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Get user-item interaction pairs.
        
        Args:
            user_id: User ID
            days: Optional time window
        
        Returns:
            List of (user_id, item_id) tuples
        """
        if not user_id or user_id <= 0:
            self.logger.warning(f"Invalid user_id: {user_id}")
            return []
        
        try:
            conditions = [
                Interaction.user_id == user_id,
                Interaction.item_id.isnot(None)
            ]
            
            if days and days > 0:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                conditions.append(Interaction.event_time >= cutoff_date)
            
            stmt = select(
                Interaction.user_id,
                Interaction.item_id
            ).where(and_(*conditions))
            
            result = self.session.execute(stmt)
            pairs = [(row.user_id, row.item_id) for row in result.all()]
            
            self.logger.debug(f"Found {len(pairs)} user-item pairs")
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error getting user-item interactions: {e}")
            return []
    
    def build_interaction_matrix(
        self,
        days: int = 30,
        min_interactions: int = 3
    ) -> Dict[int, Dict[int, int]]:
        """
        Build user-item interaction matrix for collaborative filtering.
        
        Single Responsibility: Matrix generation logic.
        
        Args:
            days: Time window in days (default: 30)
            min_interactions: Minimum interactions per user (default: 3)
        
        Returns:
            Nested dict: {user_id: {item_id: count}}
        
        Example:
            >>> matrix = repo.build_interaction_matrix(days=30)
            >>> user_123_items = matrix.get(123, {})
            >>> item_456_count = user_123_items.get(456, 0)
        """
        if days <= 0:
            days = 30
        if min_interactions < 1:
            min_interactions = 1
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            stmt = select(
                Interaction.user_id,
                Interaction.item_id,
                func.count().label('count')
            ).where(
                and_(
                    Interaction.event_time >= cutoff_date,
                    Interaction.item_id.isnot(None)
                )
            ).group_by(
                Interaction.user_id,
                Interaction.item_id
            )
            
            result = self.session.execute(stmt)
            
            # Build matrix
            matrix = defaultdict(dict)
            for row in result.all():
                matrix[row.user_id][row.item_id] = row.count
            
            # Filter users with minimum interactions
            filtered_matrix = {
                user_id: items
                for user_id, items in matrix.items()
                if len(items) >= min_interactions
            }
            
            self.logger.info(
                f"Built interaction matrix: {len(filtered_matrix)} users, "
                f"{sum(len(items) for items in filtered_matrix.values())} interactions"
            )
            
            return filtered_matrix
            
        except Exception as e:
            self.logger.error(f"Error building interaction matrix: {e}")
            return {}
    
    # ========================================
    # Session Management
    # ========================================
    
    def get_session_events(
        self,
        user_id: int,
        session_window_minutes: int = 30
    ) -> List[List[Interaction]]:
        """
        Group interactions into sessions based on time gaps.
        
        Single Responsibility: Session detection logic.
        
        Args:
            user_id: User ID
            session_window_minutes: Max gap between events in same session (default: 30)
        
        Returns:
            List of sessions (each session is a list of interactions)
        
        Example:
            >>> sessions = repo.get_session_events(
            ...     user_id=123,
            ...     session_window_minutes=30
            ... )
            >>> print(f"Total sessions: {len(sessions)}")
            >>> print(f"First session length: {len(sessions[0])}")
        """
        if not user_id or user_id <= 0:
            self.logger.warning(f"Invalid user_id: {user_id}")
            return []
        
        if session_window_minutes <= 0:
            session_window_minutes = 30
        
        try:
            # Get all user interactions in chronological order
            interactions = self.get_user_event_sequence(user_id, limit=1000)
            
            if not interactions:
                return []
            
            # Group into sessions
            sessions = self._group_interactions_into_sessions(
                interactions,
                session_window_minutes
            )
            
            self.logger.debug(f"Found {len(sessions)} sessions for user {user_id}")
            return sessions
            
        except Exception as e:
            self.logger.error(f"Error getting session events: {e}")
            return []
    
    def _group_interactions_into_sessions(
        self,
        interactions: List[Interaction],
        session_window_minutes: int
    ) -> List[List[Interaction]]:
        """
        Helper: Group interactions into sessions.
        
        Single Responsibility: Session grouping algorithm.
        """
        if not interactions:
            return []
        
        sessions = []
        current_session = [interactions[0]]
        session_gap = timedelta(minutes=session_window_minutes)
        
        for interaction in interactions[1:]:
            time_gap = interaction.event_time - current_session[-1].event_time
            
            if time_gap <= session_gap:
                current_session.append(interaction)
            else:
                # Start new session
                sessions.append(current_session)
                current_session = [interaction]
        
        # Add last session
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    # ========================================
    # Pattern Detection
    # ========================================
    
    def get_repetition_patterns(
        self,
        user_id: int,
        window_size: int = 7,
        min_repetitions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detect repetitive event patterns for Soft Loop detection.
        
        Args:
            user_id: User ID
            window_size: Rolling window size (not currently used in simple algorithm)
            min_repetitions: Minimum repetitions to report (default: 3)
        
        Returns:
            List of detected patterns
        
        Example:
            >>> patterns = repo.get_repetition_patterns(user_id=123)
            >>> for pattern in patterns:
            ...     print(f"{pattern['event_name']}: {pattern['repetitions']} times")
        """
        if not user_id or user_id <= 0:
            self.logger.warning(f"Invalid user_id: {user_id}")
            return []
        
        if min_repetitions < 2:
            min_repetitions = 2
        
        try:
            interactions = self.get_user_event_sequence(user_id, limit=100)
            
            if not interactions:
                return []
            
            # Detect patterns
            patterns = self._detect_consecutive_patterns(
                interactions,
                min_repetitions
            )
            
            self.logger.debug(f"Detected {len(patterns)} patterns for user {user_id}")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting repetition patterns: {e}")
            return []
    
    def _detect_consecutive_patterns(
        self,
        interactions: List[Interaction],
        min_repetitions: int
    ) -> List[Dict[str, Any]]:
        """
        Helper: Detect consecutive repetition patterns.
        
        Single Responsibility: Pattern detection algorithm.
        """
        patterns = []
        event_names = [i.event_name for i in interactions]
        
        current_event = None
        current_count = 0
        
        for event in event_names:
            if event == current_event:
                current_count += 1
            else:
                # Check if previous pattern meets threshold
                if current_count >= min_repetitions:
                    patterns.append({
                        "event_name": current_event,
                        "repetitions": current_count,
                        "pattern_type": "consecutive"
                    })
                
                # Start new pattern
                current_event = event
                current_count = 1
        
        # Check last pattern
        if current_count >= min_repetitions:
            patterns.append({
                "event_name": current_event,
                "repetitions": current_count,
                "pattern_type": "consecutive"
            })
        
        return patterns