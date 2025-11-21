"""
User Repository for Bookend Recommendation System.

Handles all user-related database operations:
1. User profile management
2. User feature retrieval
3. User activity tracking
4. User segmentation and cohorts
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import Session, joinedload

from ..models import User, UserFeature, Interaction
from .base_repository import BaseRepository, RecordNotFoundError

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository[User]):
    """
    Repository for User model with specialized queries.
    
    Follows principles:
    - Single Responsibility: Only user-related queries
    - One Source of Truth: All user queries go through this class
    - Error Handling: Proper exception handling and logging
    """
    
    def get_repository_name(self) -> str:
        """Get repository name for logging."""
        return "UserRepository"
    
    # ========================================
    # Single User Queries
    # ========================================
    
    def get_by_distinct_id(self, distinct_id: str) -> Optional[User]:
        """
        Get user by distinct_id (Mixpanel identifier).
        
        Args:
            distinct_id: Unique identifier from analytics
        
        Returns:
            User instance or None
        
        Example:
            >>> user = repo.get_by_distinct_id("66ea52cb71fd9eea359b4d0c")
        """
        if not distinct_id:
            self.logger.warning("distinct_id is empty")
            return None
        
        try:
            return self.get_by_field("distinct_id", distinct_id)
        except Exception as e:
            self.logger.error(f"Error fetching user by distinct_id: {e}")
            return None
    
    def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: User email
        
        Returns:
            User instance or None
        """
        if not email:
            self.logger.warning("email is empty")
            return None
        
        try:
            return self.get_by_field("email", email)
        except Exception as e:
            self.logger.error(f"Error fetching user by email: {e}")
            return None
    
    def get_with_features(self, user_id: int) -> Optional[User]:
        """
        Get user with preloaded features (eager loading).
        
        Args:
            user_id: User ID
        
        Returns:
            User instance with features loaded
        
        Example:
            >>> user = repo.get_with_features(123)
            >>> if user and user.features:
            ...     print(f"Engagement: {user.features.engagement_score}")
        """
        if not user_id or user_id <= 0:
            self.logger.warning(f"Invalid user_id: {user_id}")
            return None
        
        try:
            stmt = select(User).options(
                joinedload(User.features)
            ).where(User.id == user_id)
            
            result = self.session.execute(stmt)
            return result.unique().scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error fetching user with features: {e}")
            return None
    
    # ========================================
    # User Creation and Upsert
    # ========================================
    
    def get_or_create_by_distinct_id(
        self,
        distinct_id: str,
        **kwargs
    ) -> Tuple[Optional[User], bool]:
        """
        Get existing user or create new one by distinct_id.
        
        One Source of Truth: This is the only way to create users.
        
        Args:
            distinct_id: Unique identifier
            **kwargs: Additional user attributes
        
        Returns:
            Tuple of (User instance, created flag)
            Returns (None, False) on error
        
        Example:
            >>> user, created = repo.get_or_create_by_distinct_id(
            ...     distinct_id="abc123",
            ...     email="user@example.com",
            ...     country_code="KR"
            ... )
        """
        if not distinct_id:
            self.logger.error("distinct_id is required")
            return None, False
        
        try:
            # Try to get existing user
            user = self.get_by_distinct_id(distinct_id)
            if user:
                self.logger.debug(f"User already exists: {distinct_id}")
                return user, False
            
            # Create new user
            user = self.create(distinct_id=distinct_id, **kwargs)
            if user:
                self.logger.info(f"Created new user: {distinct_id}")
                return user, True
            else:
                return None, False
                
        except Exception as e:
            self.logger.error(f"Error in get_or_create_by_distinct_id: {e}")
            return None, False
    
    # ========================================
    # User Filters and Searches
    # ========================================
    
    def get_active_users(
        self,
        days: int = 7,
        limit: Optional[int] = None
    ) -> List[User]:
        """
        Get users active within last N days.
        
        Args:
            days: Number of days to look back (default: 7)
            limit: Maximum number of users
        
        Returns:
            List of active users, ordered by last_seen (newest first)
        """
        if days <= 0:
            self.logger.warning(f"Invalid days value: {days}, using default 7")
            days = 7
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            stmt = select(User).where(
                User.last_seen >= cutoff_date
            ).order_by(desc(User.last_seen))
            
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            users = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(users)} active users in last {days} days")
            return users
            
        except Exception as e:
            self.logger.error(f"Error fetching active users: {e}")
            return []
    
    def get_users_by_country(
        self,
        country_code: str,
        limit: Optional[int] = None
    ) -> List[User]:
        """
        Get users by country code.
        
        Args:
            country_code: ISO country code (e.g., 'KR', 'US')
            limit: Maximum number of users
        
        Returns:
            List of users from specified country
        """
        if not country_code or len(country_code) != 2:
            self.logger.warning(f"Invalid country_code: {country_code}")
            return []
        
        try:
            stmt = select(User).where(
                User.country_code == country_code.upper()
            ).order_by(desc(User.last_seen))
            
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            users = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(users)} users from {country_code}")
            return users
            
        except Exception as e:
            self.logger.error(f"Error fetching users by country: {e}")
            return []
    
    def get_logged_in_users(
        self,
        limit: Optional[int] = None
    ) -> List[User]:
        """
        Get all logged-in users.
        
        Args:
            limit: Maximum number of users
        
        Returns:
            List of logged-in users
        """
        try:
            stmt = select(User).where(
                User.is_logged_in == True
            ).order_by(desc(User.last_seen))
            
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            self.logger.error(f"Error fetching logged-in users: {e}")
            return []
    
    def search_users(
        self,
        query: str,
        fields: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[User]:
        """
        Search users by text query across multiple fields.
        
        Args:
            query: Search query
            fields: List of fields to search (default: name, email)
            limit: Maximum results (default: 20)
        
        Returns:
            List of matching users
        """
        if not query or not query.strip():
            self.logger.warning("Empty search query")
            return []
        
        if fields is None:
            fields = ["name", "email"]
        
        try:
            conditions = []
            for field in fields:
                if hasattr(User, field):
                    conditions.append(
                        getattr(User, field).ilike(f"%{query}%")
                    )
            
            if not conditions:
                self.logger.warning(f"No valid search fields: {fields}")
                return []
            
            stmt = select(User).where(or_(*conditions)).limit(limit)
            result = self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            self.logger.error(f"Error searching users: {e}")
            return []
    
    # ========================================
    # User Statistics
    # ========================================
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a user.
        
        Single Responsibility: Aggregates all user-related stats in one place.
        
        Args:
            user_id: User ID
        
        Returns:
            Dictionary with user statistics
        
        Example:
            >>> stats = repo.get_user_stats(123)
            >>> print(f"Total interactions: {stats['total_interactions']}")
        """
        if not user_id or user_id <= 0:
            self.logger.warning(f"Invalid user_id: {user_id}")
            return {}
        
        try:
            user = self.get_with_features(user_id)
            if not user:
                self.logger.warning(f"User not found: {user_id}")
                return {}
            
            # Get interaction count
            interaction_count = self._get_user_interaction_count(user_id)
            
            # Get last interaction time
            last_interaction = self._get_user_last_interaction_time(user_id)
            
            # Get event distribution
            event_distribution = self._get_user_event_distribution(user_id)
            
            return {
                "user_id": user_id,
                "distinct_id": user.distinct_id,
                "email": user.email,
                "name": user.name,
                "is_logged_in": user.is_logged_in,
                "total_sessions": user.total_sessions,
                "last_seen": user.last_seen,
                "total_interactions": interaction_count,
                "last_interaction": last_interaction,
                "event_distribution": event_distribution,
                "features": user.features.to_dict() if user.features else None,
                "location": {
                    "country_code": user.country_code,
                    "city": user.city,
                    "region": user.region
                },
                "device": {
                    "browser": user.browser,
                    "os": user.os
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            return {}
    
    def _get_user_interaction_count(self, user_id: int) -> int:
        """Helper: Get total interaction count for user."""
        try:
            stmt = select(func.count()).select_from(Interaction).where(
                Interaction.user_id == user_id
            )
            return self.session.execute(stmt).scalar() or 0
        except Exception as e:
            self.logger.error(f"Error counting interactions: {e}")
            return 0
    
    def _get_user_last_interaction_time(self, user_id: int) -> Optional[datetime]:
        """Helper: Get last interaction time for user."""
        try:
            stmt = select(func.max(Interaction.event_time)).where(
                Interaction.user_id == user_id
            )
            return self.session.execute(stmt).scalar()
        except Exception as e:
            self.logger.error(f"Error getting last interaction time: {e}")
            return None
    
    def _get_user_event_distribution(self, user_id: int) -> Dict[str, int]:
        """Helper: Get event type distribution for user."""
        try:
            stmt = select(
                Interaction.event_name,
                func.count().label('count')
            ).where(
                Interaction.user_id == user_id
            ).group_by(Interaction.event_name)
            
            result = self.session.execute(stmt)
            return {row.event_name: row.count for row in result.all()}
        except Exception as e:
            self.logger.error(f"Error getting event distribution: {e}")
            return {}
    
    # ========================================
    # User Cohort Analysis
    # ========================================
    
    def get_user_cohort(
        self,
        min_interactions: int = 10,
        engagement_threshold: float = 0.5,
        days_active: int = 30
    ) -> List[User]:
        """
        Get users matching cohort criteria.
        
        Args:
            min_interactions: Minimum interaction count
            engagement_threshold: Minimum engagement score
            days_active: Active within last N days
        
        Returns:
            List of users in cohort
        """
        if min_interactions < 0:
            min_interactions = 0
        if engagement_threshold < 0 or engagement_threshold > 1:
            engagement_threshold = 0.5
        if days_active <= 0:
            days_active = 30
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_active)
            
            stmt = select(User).join(
                UserFeature, User.id == UserFeature.user_id
            ).where(
                and_(
                    User.last_seen >= cutoff_date,
                    UserFeature.total_paraphrases >= min_interactions,
                    UserFeature.engagement_score >= engagement_threshold
                )
            ).order_by(desc(UserFeature.engagement_score))
            
            result = self.session.execute(stmt)
            users = list(result.scalars().all())
            
            self.logger.info(f"Found {len(users)} users in cohort")
            return users
            
        except Exception as e:
            self.logger.error(f"Error getting user cohort: {e}")
            return []
    
    # ========================================
    # User Updates
    # ========================================
    
    def update_last_seen(self, user_id: int) -> bool:
        """
        Update user's last_seen timestamp to now.
        
        Args:
            user_id: User ID
        
        Returns:
            True if successful, False otherwise
        """
        if not user_id or user_id <= 0:
            return False
        
        try:
            user = self.update(user_id, last_seen=datetime.utcnow())
            return user is not None
        except Exception as e:
            self.logger.error(f"Error updating last_seen: {e}")
            return False
    
    def increment_session_count(self, user_id: int) -> bool:
        """
        Increment user's total session count and update last_seen.
        
        Args:
            user_id: User ID
        
        Returns:
            True if successful, False otherwise
        """
        if not user_id or user_id <= 0:
            return False
        
        try:
            user = self.get_by_id(user_id)
            if not user:
                return False
            
            user.total_sessions = (user.total_sessions or 0) + 1
            user.last_seen = datetime.utcnow()
            self.session.flush()
            
            self.logger.debug(f"Incremented session count for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error incrementing session count: {e}")
            return False
    
    # ========================================
    # UTM and Marketing Queries
    # ========================================
    
    def get_users_by_utm(
        self,
        utm_source: Optional[str] = None,
        utm_medium: Optional[str] = None,
        utm_campaign: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[User]:
        """
        Get users by UTM parameters.
        
        Args:
            utm_source: UTM source (e.g., "google", "facebook")
            utm_medium: UTM medium (e.g., "cpc", "email")
            utm_campaign: UTM campaign name
            limit: Maximum results
        
        Returns:
            List of matching users
        """
        conditions = []
        
        if utm_source:
            conditions.append(User.initial_utm_source == utm_source)
        if utm_medium:
            conditions.append(User.initial_utm_medium == utm_medium)
        if utm_campaign:
            conditions.append(User.initial_utm_campaign == utm_campaign)
        
        if not conditions:
            self.logger.warning("No UTM parameters provided")
            return []
        
        try:
            stmt = select(User).where(and_(*conditions))
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            self.logger.error(f"Error getting users by UTM: {e}")
            return []