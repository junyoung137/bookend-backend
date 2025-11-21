"""
Item Repository for Bookend Recommendation System.

Handles all item-related database operations:
1. Item catalog management
2. Item feature retrieval
3. Popularity and trending tracking
4. Category and type management
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import Session, joinedload

from ..models import Item, ItemFeature, Interaction
from .base_repository import BaseRepository, RecordNotFoundError

logger = logging.getLogger(__name__)


class ItemRepository(BaseRepository[Item]):
    """
    Repository for Item model with specialized queries.
    
    Follows principles:
    - Single Responsibility: Only item-related queries
    - One Source of Truth: All item queries go through this class
    - Error Handling: Proper exception handling and logging
    """
    
    def get_repository_name(self) -> str:
        """Get repository name for logging."""
        return "ItemRepository"
    
    # ========================================
    # Single Item Queries
    # ========================================
    
    def get_by_code(self, item_code: str) -> Optional[Item]:
        """
        Get item by item_code (unique identifier).
        
        Args:
            item_code: Unique item identifier (e.g., "paraphrase_tool")
        
        Returns:
            Item instance or None
        
        Example:
            >>> item = repo.get_by_code("paraphrase_tool")
        """
        if not item_code:
            self.logger.warning("item_code is empty")
            return None
        
        try:
            return self.get_by_field("item_code", item_code)
        except Exception as e:
            self.logger.error(f"Error fetching item by code: {e}")
            return None
    
    def get_with_features(self, item_id: int) -> Optional[Item]:
        """
        Get item with preloaded features (eager loading).
        
        Args:
            item_id: Item ID
        
        Returns:
            Item instance with features loaded
        
        Example:
            >>> item = repo.get_with_features(123)
            >>> if item and item.features:
            ...     print(f"Popularity: {item.features.popularity_score}")
        """
        if not item_id or item_id <= 0:
            self.logger.warning(f"Invalid item_id: {item_id}")
            return None
        
        try:
            stmt = select(Item).options(
                joinedload(Item.features)
            ).where(Item.id == item_id)
            
            result = self.session.execute(stmt)
            return result.unique().scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error fetching item with features: {e}")
            return None
    
    # ========================================
    # Item Creation and Upsert
    # ========================================
    
    def get_or_create_by_code(
        self,
        item_code: str,
        item_name: str,
        item_type: str,
        **kwargs
    ) -> Tuple[Optional[Item], bool]:
        """
        Get existing item or create new one by item_code.
        
        One Source of Truth: This is the only way to create items.
        
        Args:
            item_code: Unique item identifier
            item_name: Human-readable item name
            item_type: Item type (e.g., "editor_feature", "tool")
            **kwargs: Additional item attributes
        
        Returns:
            Tuple of (Item instance, created flag)
            Returns (None, False) on error
        
        Example:
            >>> item, created = repo.get_or_create_by_code(
            ...     item_code="paraphrase_tool",
            ...     item_name="Paraphrasing Tool",
            ...     item_type="editor_feature",
            ...     category="editor"
            ... )
        """
        if not item_code or not item_name or not item_type:
            self.logger.error("item_code, item_name, and item_type are required")
            return None, False
        
        try:
            # Try to get existing item
            item = self.get_by_code(item_code)
            if item:
                self.logger.debug(f"Item already exists: {item_code}")
                return item, False
            
            # Create new item
            item = self.create(
                item_code=item_code,
                item_name=item_name,
                item_type=item_type,
                **kwargs
            )
            if item:
                self.logger.info(f"Created new item: {item_code}")
                return item, True
            else:
                return None, False
                
        except Exception as e:
            self.logger.error(f"Error in get_or_create_by_code: {e}")
            return None, False
    
    # ========================================
    # Item Filters
    # ========================================
    
    def get_active_items(
        self,
        limit: Optional[int] = None
    ) -> List[Item]:
        """
        Get all active items.
        
        Args:
            limit: Maximum number of items
        
        Returns:
            List of active items ordered by usage
        """
        try:
            stmt = select(Item).where(
                Item.is_active == True
            ).order_by(desc(Item.total_usage_count))
            
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            items = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(items)} active items")
            return items
            
        except Exception as e:
            self.logger.error(f"Error fetching active items: {e}")
            return []
    
    def get_items_by_category(
        self,
        category: str,
        active_only: bool = True,
        limit: Optional[int] = None
    ) -> List[Item]:
        """
        Get items by category.
        
        Args:
            category: Category name (e.g., "editor", "tools")
            active_only: Only return active items
            limit: Maximum number of items
        
        Returns:
            List of items in category
        """
        if not category:
            self.logger.warning("category is empty")
            return []
        
        try:
            conditions = [Item.category == category]
            if active_only:
                conditions.append(Item.is_active == True)
            
            stmt = select(Item).where(
                and_(*conditions)
            ).order_by(desc(Item.total_usage_count))
            
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            items = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(items)} items in category '{category}'")
            return items
            
        except Exception as e:
            self.logger.error(f"Error fetching items by category: {e}")
            return []
    
    def get_items_by_type(
        self,
        item_type: str,
        active_only: bool = True,
        limit: Optional[int] = None
    ) -> List[Item]:
        """
        Get items by type.
        
        Args:
            item_type: Item type (e.g., "editor_feature")
            active_only: Only return active items
            limit: Maximum number of items
        
        Returns:
            List of items of specified type
        """
        if not item_type:
            self.logger.warning("item_type is empty")
            return []
        
        try:
            conditions = [Item.item_type == item_type]
            if active_only:
                conditions.append(Item.is_active == True)
            
            stmt = select(Item).where(
                and_(*conditions)
            ).order_by(desc(Item.total_usage_count))
            
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            
            result = self.session.execute(stmt)
            items = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(items)} items of type '{item_type}'")
            return items
            
        except Exception as e:
            self.logger.error(f"Error fetching items by type: {e}")
            return []
    
    # ========================================
    # Popularity and Trending
    # ========================================
    
    def get_popular_items(
        self,
        limit: int = 10,
        days: Optional[int] = None,
        active_only: bool = True
    ) -> List[Item]:
        """
        Get most popular items by usage count.
        
        Args:
            limit: Maximum number of items (default: 10)
            days: Consider only last N days (None = all time)
            active_only: Only return active items
        
        Returns:
            List of popular items ordered by usage
        
        Example:
            >>> popular = repo.get_popular_items(limit=10, days=7)
        """
        if limit <= 0:
            limit = 10
        
        try:
            if days and days > 0:
                # Calculate from recent interactions
                return self._get_popular_items_by_interactions(limit, days, active_only)
            else:
                # Use total_usage_count
                return self._get_popular_items_all_time(limit, active_only)
                
        except Exception as e:
            self.logger.error(f"Error fetching popular items: {e}")
            return []
    
    def _get_popular_items_all_time(self, limit: int, active_only: bool) -> List[Item]:
        """Helper: Get popular items by all-time usage count."""
        try:
            conditions = []
            if active_only:
                conditions.append(Item.is_active == True)
            
            stmt = select(Item)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            stmt = stmt.order_by(desc(Item.total_usage_count)).limit(limit)
            
            result = self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error in _get_popular_items_all_time: {e}")
            return []
    
    def _get_popular_items_by_interactions(
        self,
        limit: int,
        days: int,
        active_only: bool
    ) -> List[Item]:
        """Helper: Get popular items from recent interactions."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            conditions = [
                Interaction.event_time >= cutoff_date,
                Interaction.item_id.isnot(None)
            ]
            
            stmt = select(
                Item,
                func.count(Interaction.id).label('interaction_count')
            ).join(
                Interaction, Item.id == Interaction.item_id
            ).where(and_(*conditions))
            
            if active_only:
                stmt = stmt.where(Item.is_active == True)
            
            stmt = stmt.group_by(Item.id).order_by(
                desc('interaction_count')
            ).limit(limit)
            
            result = self.session.execute(stmt)
            return [row[0] for row in result.all()]
            
        except Exception as e:
            self.logger.error(f"Error in _get_popular_items_by_interactions: {e}")
            return []
    
    def get_trending_items(
        self,
        limit: int = 10,
        days: int = 7,
        active_only: bool = True
    ) -> List[Item]:
        """
        Get trending items (items with increasing usage).
        
        Uses trending_score from ItemFeature table.
        
        Args:
            limit: Maximum number of items (default: 10)
            days: Trending window in days (default: 7)
            active_only: Only return active items
        
        Returns:
            List of trending items
        """
        if limit <= 0:
            limit = 10
        if days <= 0:
            days = 7
        
        try:
            conditions = [ItemFeature.trending_score.isnot(None)]
            if active_only:
                conditions.append(Item.is_active == True)
            
            stmt = select(Item).join(
                ItemFeature, Item.id == ItemFeature.item_id
            ).where(
                and_(*conditions)
            ).order_by(
                desc(ItemFeature.trending_score)
            ).limit(limit)
            
            result = self.session.execute(stmt)
            items = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(items)} trending items")
            return items
            
        except Exception as e:
            self.logger.error(f"Error fetching trending items: {e}")
            return []
    
    # ========================================
    # Item Statistics
    # ========================================
    
    def get_item_stats(self, item_id: int) -> Dict[str, Any]:
        """
        Get comprehensive statistics for an item.
        
        Single Responsibility: Aggregates all item-related stats in one place.
        
        Args:
            item_id: Item ID
        
        Returns:
            Dictionary with item statistics
        """
        if not item_id or item_id <= 0:
            self.logger.warning(f"Invalid item_id: {item_id}")
            return {}
        
        try:
            item = self.get_with_features(item_id)
            if not item:
                self.logger.warning(f"Item not found: {item_id}")
                return {}
            
            # Get event distribution
            event_distribution = self._get_item_event_distribution(item_id)
            
            # Get recent usage
            recent_usage_7d = self._get_item_recent_usage(item_id, days=7)
            recent_usage_30d = self._get_item_recent_usage(item_id, days=30)
            
            return {
                "item_id": item_id,
                "item_code": item.item_code,
                "item_name": item.item_name,
                "item_type": item.item_type,
                "category": item.category,
                "subcategory": item.subcategory,
                "total_usage_count": item.total_usage_count,
                "unique_user_count": item.unique_user_count,
                "avg_rating": item.avg_rating,
                "is_active": item.is_active,
                "is_premium": item.is_premium,
                "event_distribution": event_distribution,
                "recent_usage": {
                    "last_7_days": recent_usage_7d,
                    "last_30_days": recent_usage_30d
                },
                "features": item.features.to_dict() if item.features else None,
                "tags": item.tags
            }
            
        except Exception as e:
            self.logger.error(f"Error getting item stats: {e}")
            return {}
    
    def _get_item_event_distribution(self, item_id: int) -> Dict[str, int]:
        """Helper: Get event type distribution for item."""
        try:
            stmt = select(
                Interaction.event_name,
                func.count().label('count')
            ).where(
                Interaction.item_id == item_id
            ).group_by(Interaction.event_name)
            
            result = self.session.execute(stmt)
            return {row.event_name: row.count for row in result.all()}
        except Exception as e:
            self.logger.error(f"Error getting event distribution: {e}")
            return {}
    
    def _get_item_recent_usage(self, item_id: int, days: int) -> int:
        """Helper: Get usage count for last N days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            stmt = select(func.count()).select_from(Interaction).where(
                and_(
                    Interaction.item_id == item_id,
                    Interaction.event_time >= cutoff_date
                )
            )
            return self.session.execute(stmt).scalar() or 0
        except Exception as e:
            self.logger.error(f"Error getting recent usage: {e}")
            return 0
    
    # ========================================
    # Item Updates
    # ========================================
    
    def increment_usage_count(
        self,
        item_id: int,
        user_id: int
    ) -> bool:
        """
        Increment item usage count and update unique user count.
        
        Args:
            item_id: Item ID
            user_id: User ID who used the item
        
        Returns:
            True if successful, False otherwise
        
        Example:
            >>> success = repo.increment_usage_count(item_id=123, user_id=456)
        """
        if not item_id or item_id <= 0 or not user_id or user_id <= 0:
            self.logger.warning(f"Invalid item_id or user_id: {item_id}, {user_id}")
            return False
        
        try:
            item = self.get_by_id(item_id)
            if not item:
                self.logger.warning(f"Item not found: {item_id}")
                return False
            
            # Increment total usage
            item.total_usage_count = (item.total_usage_count or 0) + 1
            
            # Check if this is a new unique user
            is_new_user = self._is_new_user_for_item(item_id, user_id)
            if is_new_user:
                item.unique_user_count = (item.unique_user_count or 0) + 1
            
            self.session.flush()
            self.logger.debug(f"Incremented usage count for item {item_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error incrementing usage count: {e}")
            return False
    
    def _is_new_user_for_item(self, item_id: int, user_id: int) -> bool:
        """Helper: Check if this is the first interaction between user and item."""
        try:
            stmt = select(func.count()).select_from(Interaction).where(
                and_(
                    Interaction.item_id == item_id,
                    Interaction.user_id == user_id
                )
            )
            count = self.session.execute(stmt).scalar() or 0
            return count == 1  # True if this is the first interaction
        except Exception as e:
            self.logger.error(f"Error checking new user: {e}")
            return False
    
    # ========================================
    # Item Recommendations
    # ========================================
    
    def get_similar_items(
        self,
        item_id: int,
        limit: int = 10,
        active_only: bool = True
    ) -> List[Item]:
        """
        Get similar items based on category and type.
        
        Args:
            item_id: Source item ID
            limit: Maximum number of similar items (default: 10)
            active_only: Only return active items
        
        Returns:
            List of similar items
        """
        if not item_id or item_id <= 0:
            self.logger.warning(f"Invalid item_id: {item_id}")
            return []
        
        if limit <= 0:
            limit = 10
        
        try:
            item = self.get_by_id(item_id)
            if not item:
                self.logger.warning(f"Item not found: {item_id}")
                return []
            
            conditions = [
                Item.id != item_id,
                or_(
                    Item.category == item.category,
                    Item.item_type == item.item_type
                )
            ]
            
            if active_only:
                conditions.append(Item.is_active == True)
            
            stmt = select(Item).where(
                and_(*conditions)
            ).order_by(desc(Item.total_usage_count)).limit(limit)
            
            result = self.session.execute(stmt)
            items = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(items)} similar items for item {item_id}")
            return items
            
        except Exception as e:
            self.logger.error(f"Error fetching similar items: {e}")
            return []
    
    # ========================================
    # Catalog Management
    # ========================================
    
    def get_category_distribution(self, active_only: bool = True) -> Dict[str, int]:
        """
        Get distribution of items across categories.
        
        Args:
            active_only: Only count active items
        
        Returns:
            Dictionary mapping category to item count
        """
        try:
            stmt = select(
                Item.category,
                func.count().label('count')
            )
            
            if active_only:
                stmt = stmt.where(Item.is_active == True)
            
            stmt = stmt.group_by(Item.category)
            
            result = self.session.execute(stmt)
            distribution = {
                row.category: row.count 
                for row in result.all() 
                if row.category
            }
            
            self.logger.debug(f"Category distribution: {distribution}")
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error getting category distribution: {e}")
            return {}
    
    def search_items(
        self,
        query: str,
        active_only: bool = True,
        limit: int = 20
    ) -> List[Item]:
        """
        Search items by name, code, or description.
        
        Args:
            query: Search query
            active_only: Only return active items
            limit: Maximum results (default: 20)
        
        Returns:
            List of matching items
        """
        if not query or not query.strip():
            self.logger.warning("Empty search query")
            return []
        
        try:
            conditions = [
                or_(
                    Item.item_name.ilike(f"%{query}%"),
                    Item.item_code.ilike(f"%{query}%"),
                    Item.description.ilike(f"%{query}%")
                )
            ]
            
            if active_only:
                conditions.append(Item.is_active == True)
            
            stmt = select(Item).where(
                and_(*conditions)
            ).order_by(desc(Item.total_usage_count)).limit(limit)
            
            result = self.session.execute(stmt)
            items = list(result.scalars().all())
            
            self.logger.debug(f"Found {len(items)} items matching '{query}'")
            return items
            
        except Exception as e:
            self.logger.error(f"Error searching items: {e}")
            return []