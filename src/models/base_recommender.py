from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =========================================================
# Recommendation Result Data Class
# =========================================================

@dataclass
class RecommendationResult:
    """
    Structured recommendation result with metadata.
    
    Attributes:
        item_id: Recommended item database ID
        item_code: Recommended item code
        item_name: Item display name
        score: Recommendation score (0.0 - 1.0)
        rank: Rank in recommendation list (1-indexed)
        reason: Explanation for recommendation
        metadata: Additional context (e.g., category, tags)
        timestamp: When recommendation was generated
    
    Example:
        >>> result = RecommendationResult(
        ...     item_id=123,
        ...     item_code="paraphrase_formal",
        ...     item_name="Formal Paraphrasing",
        ...     score=0.85,
        ...     rank=1,
        ...     reason="Matches your preferred tone",
        ...     metadata={"category": "paraphrasing"}
        ... )
    """
    item_id: int
    item_code: str
    item_name: str
    score: float
    rank: int
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "item_id": self.item_id,
            "item_code": self.item_code,
            "item_name": self.item_name,
            "score": round(self.score, 4),
            "rank": self.rank,
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# =========================================================
# Base Recommender Abstract Class
# =========================================================

class BaseRecommender(ABC):
    """
    Abstract base class for all recommender systems.
    
    Implements Template Method Pattern:
    - fit(): Train/prepare the model
    - recommend(): Generate recommendations
    - _validate_user(): Validate user exists
    - _filter_candidates(): Filter items by business rules
    - _score_items(): Score each candidate item
    - _rank_items(): Sort and rank items
    
    Subclasses must implement:
    - _compute_scores(): Core scoring logic
    - get_model_name(): Return model identifier
    
    Example:
        >>> class MyRecommender(BaseRecommender):
        ...     def _compute_scores(self, user_id, context):
        ...         # Custom scoring logic
        ...         return {item_id: score}
        ...     
        ...     def get_model_name(self):
        ...         return "my_recommender"
    """
    
    def __init__(self, session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base recommender.
        
        Args:
            session: Database session
            config: Model configuration dictionary
        """
        self.session = session
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_fitted = False
    
    # =========================================================
    # Abstract Methods (Must Implement)
    # =========================================================
    
    @abstractmethod
    def _compute_scores(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None,
        candidate_items: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute recommendation scores for candidate items.
        
        Args:
            user_id: User database ID
            context: Contextual features
            candidate_items: List of candidate item IDs (None = all items)
        
        Returns:
            Dictionary mapping item_id to score (0.0 - 1.0)
        
        Note:
            This is the core recommendation logic.
            Subclasses must implement their scoring strategy here.
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return model identifier for logging/tracking.
        
        Returns:
            Model name string
        """
        pass
    
    # =========================================================
    # Template Method (Common Workflow)
    # =========================================================
    
    def recommend(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
        exclude_items: Optional[List[int]] = None,
        include_reasons: bool = True
    ) -> List[RecommendationResult]:
        """
        Generate recommendations for a user.
        
        Template method implementing standard recommendation workflow:
        1. Validate user
        2. Get candidate items
        3. Compute scores
        4. Filter by minimum score
        5. Rank and limit
        6. Create results with reasons
        
        Args:
            user_id: User database ID
            context: Contextual features (time, device, location, etc.)
            limit: Maximum number of recommendations
            min_score: Minimum score threshold (0.0 - 1.0)
            exclude_items: Item IDs to exclude
            include_reasons: Whether to generate explanations
        
        Returns:
            List of RecommendationResult objects
        
        Example:
            >>> recommender = MyRecommender(session)
            >>> results = recommender.recommend(
            ...     user_id=123,
            ...     context={"time_of_day": "morning"},
            ...     limit=5
            ... )
        """
        try:
            self.logger.info(
                f"Generating recommendations: user={user_id}, "
                f"limit={limit}, model={self.get_model_name()}"
            )
            
            # Step 1: Validate user
            if not self._validate_user(user_id):
                self.logger.warning(f"User {user_id} not found or invalid")
                return []
            
            # Step 2: Get candidate items
            candidate_items = self._get_candidate_items(user_id, exclude_items)
            if not candidate_items:
                self.logger.warning(f"No candidate items for user {user_id}")
                return []
            
            self.logger.debug(f"Found {len(candidate_items)} candidate items")
            
            # Step 3: Compute scores
            scores = self._compute_scores(user_id, context, candidate_items)
            if not scores:
                self.logger.warning(f"No scores computed for user {user_id}")
                return []
            
            self.logger.debug(f"Computed scores for {len(scores)} items")
            
            # Step 4: Filter by minimum score
            filtered_scores = {
                item_id: score
                for item_id, score in scores.items()
                if score >= min_score
            }
            
            if not filtered_scores:
                self.logger.warning(
                    f"No items passed min_score={min_score} threshold"
                )
                return []
            
            self.logger.debug(
                f"{len(filtered_scores)} items passed min_score threshold"
            )
            
            # Step 5: Rank and limit
            ranked_items = self._rank_items(filtered_scores, limit)
            
            # Step 6: Create results with metadata
            results = self._create_results(
                ranked_items,
                user_id,
                context,
                include_reasons
            )
            
            self.logger.info(
                f"Generated {len(results)} recommendations for user {user_id}"
            )
            
            return results
        
        except Exception as e:
            self.logger.error(
                f"Failed to generate recommendations for user {user_id}: {e}",
                exc_info=True
            )
            return []
    
    # =========================================================
    # Helper Methods (Can Override)
    # =========================================================
    
    def _validate_user(self, user_id: int) -> bool:
        """
        Validate that user exists and is active.
        
        Args:
            user_id: User database ID
        
        Returns:
            True if valid, False otherwise
        """
        try:
            from src.database.models import User
            
            user = self.session.get(User, user_id)
            return user is not None
        
        except Exception as e:
            self.logger.error(f"User validation failed: {e}")
            return False
    
    def _get_candidate_items(
        self,
        user_id: int,
        exclude_items: Optional[List[int]] = None
    ) -> List[int]:
        """
        Get candidate items for recommendation.
        
        Args:
            user_id: User database ID
            exclude_items: Item IDs to exclude
        
        Returns:
            List of candidate item IDs
        """
        try:
            from src.database.models import Item
            
            query = self.session.query(Item.id).filter(
                Item.is_active == True
            )
            
            # Exclude specified items
            if exclude_items:
                query = query.filter(~Item.id.in_(exclude_items))
            
            candidate_ids = [item_id for (item_id,) in query.all()]
            
            return candidate_ids
        
        except Exception as e:
            self.logger.error(f"Failed to get candidate items: {e}")
            return []
    
    def _rank_items(
        self,
        scores: Dict[int, float],
        limit: int
    ) -> List[Tuple[int, float]]:
        """
        Rank items by score and apply limit.
        
        Args:
            scores: Dictionary mapping item_id to score
            limit: Maximum number of items to return
        
        Returns:
            List of (item_id, score) tuples, sorted by score descending
        """
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return ranked
    
    def _create_results(
        self,
        ranked_items: List[Tuple[int, float]],
        user_id: int,
        context: Optional[Dict[str, Any]],
        include_reasons: bool
    ) -> List[RecommendationResult]:
        """
        Create RecommendationResult objects from ranked items.
        
        Args:
            ranked_items: List of (item_id, score) tuples
            user_id: User database ID
            context: Contextual features
            include_reasons: Whether to generate reasons
        
        Returns:
            List of RecommendationResult objects
        """
        try:
            from src.database.models import Item
            
            results = []
            
            for rank, (item_id, score) in enumerate(ranked_items, start=1):
                # Convert numpy types to Python native types
                item_id = int(item_id)
                score = float(score)
                
                # Get item details
                item = self.session.get(Item, item_id)
                if not item:
                    self.logger.warning(f"Item {item_id} not found")
                    continue
                
                # Generate reason
                reason = self._generate_reason(
                    item,
                    score,
                    user_id,
                    context
                ) if include_reasons else "Recommended for you"
                
                # Create result
                result = RecommendationResult(
                    item_id=item.id,
                    item_code=item.item_code,
                    item_name=item.item_name,
                    score=score,
                    rank=rank,
                    reason=reason,
                    metadata={
                        "category": item.category,
                        "item_type": item.item_type,
                        "is_premium": item.is_premium,
                    }
                )
                
                results.append(result)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Failed to create results: {e}", exc_info=True)
            return []
    
    def _generate_reason(
        self,
        item: Any,
        score: float,
        user_id: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable explanation for recommendation.
        
        Args:
            item: Item object
            score: Recommendation score
            user_id: User database ID
            context: Contextual features
        
        Returns:
            Explanation string
        
        Note:
            Subclasses can override for model-specific explanations.
        """
        # Default generic reason
        if score >= 0.8:
            return f"Highly relevant to your preferences"
        elif score >= 0.6:
            return f"Matches your usage patterns"
        elif score >= 0.4:
            return f"Popular in {item.category}"
        else:
            return f"You might like this"
    
    # =========================================================
    # Training Interface (Optional)
    # =========================================================
    
    def fit(self, **kwargs) -> None:
        """
        Train or prepare the recommendation model.
        
        Args:
            **kwargs: Model-specific training parameters
        
        Note:
            Not all models require training (e.g., rule-based).
            Collaborative filtering models will implement this.
        """
        self.logger.info(f"Fitting {self.get_model_name()} model")
        self.is_fitted = True
    
    # =========================================================
    # Utility Methods
    # =========================================================
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default fallback.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        return self.config.get(key, default)


if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("=" * 70)
    print("BASE RECOMMENDER DEMO")
    print("=" * 70)
    
    # Create sample recommendation result
    result = RecommendationResult(
        item_id=1,
        item_code="paraphrase_formal",
        item_name="Formal Paraphrasing",
        score=0.87,
        rank=1,
        reason="Matches your preferred tone",
        metadata={"category": "paraphrasing", "is_premium": False}
    )
    
    print("\n📊 Sample Recommendation Result:")
    print(result)
    
    print("\n📋 As Dictionary:")
    import json
    print(json.dumps(result.to_dict(), indent=2))
    
    print("\n✅ Base recommender abstract class ready!")
    print("   Subclasses must implement:")
    print("   - _compute_scores()")
    print("   - get_model_name()")