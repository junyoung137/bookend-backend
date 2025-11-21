"""
Item-based Collaborative Filtering recommender.

Implements item-based CF algorithm:
1. Find similar items based on user interaction patterns
2. Recommend items similar to what user has interacted with
3. Support multiple similarity metrics (cosine, pearson, jaccard)
4. K-nearest neighbors approach

Principle: "Items similar to what you liked"
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix
from sqlalchemy.orm import Session

from src.models.base_recommender import BaseRecommender, RecommendationResult
from src.models.collaborative.interaction_matrix import InteractionMatrixBuilder
from src.models.utils.similarity import (
    cosine_similarity,
    pearson_correlation,
    SimilarityMetric,
    compute_similarity_matrix
)
from src.database.models import Item

logger = logging.getLogger(__name__)


class ItemCFRecommender(BaseRecommender):
    """
    Item-based Collaborative Filtering recommender.
    
    Finds similar items and recommends items similar to user's history.
    
    Configuration:
        - similarity_metric: Similarity metric to use (default: 'cosine')
        - k_neighbors: Number of similar items to consider (default: 50)
        - min_common_users: Minimum common users for similarity (default: 5)
        - normalize: Whether to normalize vectors (default: True)
    """
    
    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Item-CF recommender.
        
        Args:
            session: Database session
            config: Configuration dictionary
        """
        super().__init__(session, config)
        
        # Get configuration
        self.similarity_metric = SimilarityMetric(
            self.get_config("similarity_metric", "cosine")
        )
        self.k_neighbors = self.get_config("k_neighbors", 50)
        self.min_common_users = self.get_config("min_common_users", 5)
        self.normalize = self.get_config("normalize", True)
        
        # Matrix builder
        self.matrix_builder = InteractionMatrixBuilder(session, config)
        
        # Precomputed similarity matrix (cached)
        self.similarity_matrix: Optional[np.ndarray] = None
        self.similarity_computed_at: Optional[datetime] = None
    
    def fit(self, **kwargs) -> None:
        """
        Train the Item-CF model.
        
        Builds interaction matrix and precomputes item similarity matrix.
        
        Args:
            **kwargs: Additional arguments for matrix building
        """
        try:
            self.logger.info("Fitting Item-CF model")
            
            # Build interaction matrix
            self.logger.info("Building interaction matrix...")
            self.matrix_builder.build_matrix(
                weighting=kwargs.get("weighting", "binary"),
                min_interactions=kwargs.get("min_interactions", 2),
                lookback_days=kwargs.get("lookback_days", None),
                normalize=self.normalize
            )
            
            # Precompute item similarity matrix
            self.logger.info("Precomputing item similarity matrix...")
            self._precompute_similarity_matrix()
            
            self.is_fitted = True
            self.logger.info("Item-CF model fitted successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to fit Item-CF model: {e}", exc_info=True)
            raise
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return f"item_cf_{self.similarity_metric.value}"
    
    def _compute_scores(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None,
        candidate_items: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute recommendation scores using Item-CF.
        
        Args:
            user_id: User database ID
            context: Contextual features (not used in basic CF)
            candidate_items: List of candidate item IDs
        
        Returns:
            Dictionary mapping item_id to score
        """
        try:
            # Check if model is fitted
            if not self.is_fitted:
                self.logger.error("Model not fitted yet")
                return {}
            
            # Get user's interaction vector
            user_vector = self.matrix_builder.get_user_vector(user_id)
            
            if user_vector is None:
                self.logger.debug(f"User {user_id} not found in matrix")
                return {}
            
            # Get items user has interacted with
            user_items = self._get_user_items(user_vector)
            
            if not user_items:
                self.logger.debug(f"User {user_id} has no interactions")
                return {}
            
            # Compute scores for candidate items
            scores = self._compute_item_scores(
                user_items,
                user_vector,
                candidate_items
            )
            
            return scores
        
        except Exception as e:
            self.logger.error(f"Failed to compute scores: {e}", exc_info=True)
            return {}
    
    def _get_user_items(
        self,
        user_vector: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Get items the user has interacted with.
        
        Args:
            user_vector: User's interaction vector
        
        Returns:
            List of (item_id, interaction_strength) tuples
        """
        try:
            user_items = []
            
            for item_idx in np.where(user_vector > 0)[0]:
                item_id = self.matrix_builder.idx_to_item_id[item_idx]
                interaction_strength = user_vector[item_idx]
                user_items.append((item_id, interaction_strength))
            
            return user_items
        
        except Exception as e:
            self.logger.error(f"Failed to get user items: {e}")
            return []
    
    def _compute_item_scores(
        self,
        user_items: List[Tuple[int, float]],
        user_vector: np.ndarray,
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """
        Compute scores for candidate items based on user's history.
        
        Args:
            user_items: List of (item_id, interaction_strength) tuples
            user_vector: User's interaction vector
            candidate_items: List of candidate item IDs (None = all items)
        
        Returns:
            Dictionary mapping item_id to score
        """
        try:
            scores = {}
            user_item_ids = set(item_id for item_id, _ in user_items)
            
            # For each item in user's history
            for item_id, interaction_strength in user_items:
                # Find similar items
                similar_items = self._find_similar_items(item_id)
                
                # Aggregate scores
                for similar_item_id, similarity in similar_items:
                    # Skip items user already interacted with
                    if similar_item_id in user_item_ids:
                        continue
                    
                    # Skip if not in candidate items
                    if candidate_items and similar_item_id not in candidate_items:
                        continue
                    
                    # Weighted score: interaction_strength * item_similarity
                    weighted_score = interaction_strength * similarity
                    
                    if similar_item_id not in scores:
                        scores[similar_item_id] = 0.0
                    
                    scores[similar_item_id] += weighted_score
            
            # Normalize scores
            if scores:
                # Normalize by number of user items
                n_user_items = len(user_items)
                if n_user_items > 0:
                    scores = {
                        item_id: score / n_user_items
                        for item_id, score in scores.items()
                    }
            
            return scores
        
        except Exception as e:
            self.logger.error(f"Failed to compute item scores: {e}")
            return {}
    
    def _find_similar_items(
        self,
        item_id: int
    ) -> List[Tuple[int, float]]:
        """
        Find K most similar items to the target item.
        
        Args:
            item_id: Target item database ID
        
        Returns:
            List of (similar_item_id, similarity_score) tuples
        """
        try:
            # Get item index in matrix
            item_idx = self.matrix_builder.item_id_to_idx.get(item_id)
            
            if item_idx is None:
                return []
            
            # Use precomputed similarity matrix
            if self.similarity_matrix is None:
                self.logger.warning("Similarity matrix not precomputed")
                return []
            
            similarities = self.similarity_matrix[item_idx, :]
            
            # Get top-K similar items (excluding self)
            similar_items = []
            
            for other_idx in np.argsort(-similarities)[:self.k_neighbors + 1]:
                if other_idx == item_idx:
                    continue  # Skip self
                
                other_item_id = self.matrix_builder.idx_to_item_id[other_idx]
                similarity = similarities[other_idx]
                
                if similarity > 0:
                    # Check minimum common users
                    if self._has_min_common_users(item_idx, other_idx):
                        similar_items.append((other_item_id, similarity))
                
                if len(similar_items) >= self.k_neighbors:
                    break
            
            return similar_items
        
        except Exception as e:
            self.logger.error(f"Failed to find similar items: {e}")
            return []
    
    def _has_min_common_users(
        self,
        item_idx: int,
        other_item_idx: int
    ) -> bool:
        """
        Check if two items have minimum number of common users.
        
        Args:
            item_idx: Target item's matrix index
            other_item_idx: Other item's matrix index
        
        Returns:
            True if minimum common users criterion is met
        """
        try:
            matrix = self.matrix_builder.matrix
            
            item_vector = matrix[:, item_idx].toarray().flatten()
            other_vector = matrix[:, other_item_idx].toarray().flatten()
            
            # Count common users (both non-zero)
            common_users = np.sum((item_vector > 0) & (other_vector > 0))
            
            return common_users >= self.min_common_users
        
        except Exception as e:
            self.logger.error(f"Failed to check common users: {e}")
            return False
    
    def _precompute_similarity_matrix(self) -> None:
        """
        Precompute pairwise item similarity matrix.
        
        Note: Item-CF typically has fewer items than users,
        making precomputation more feasible.
        """
        try:
            matrix = self.matrix_builder.matrix
            n_items = matrix.shape[1]
            
            if n_items == 0:
                self.logger.warning("No items in matrix")
                return
            
            self.logger.info(f"Precomputing {n_items}x{n_items} item similarity matrix...")
            
            # Transpose matrix to get item-user matrix
            item_matrix = matrix.T.toarray()
            
            # Compute pairwise similarities
            self.similarity_matrix = compute_similarity_matrix(
                item_matrix,
                metric=self.similarity_metric
            )
            
            self.similarity_computed_at = datetime.now()
            
            self.logger.info("Item similarity matrix precomputed successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to precompute similarity matrix: {e}")
            self.similarity_matrix = None
    
    def get_item_similarities(
        self,
        item_id: int,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get most similar items to a given item.
        
        Public method for exploring item relationships.
        
        Args:
            item_id: Target item database ID
            top_k: Number of similar items to return
        
        Returns:
            List of (similar_item_id, similarity_score) tuples
        
        Example:
            >>> recommender = ItemCFRecommender(session)
            >>> recommender.fit()
            >>> similar = recommender.get_item_similarities(item_id=5, top_k=5)
        """
        try:
            if not self.is_fitted:
                self.logger.error("Model not fitted yet")
                return []
            
            # Temporarily override k_neighbors for this query
            original_k = self.k_neighbors
            self.k_neighbors = top_k
            
            similar_items = self._find_similar_items(item_id)
            
            # Restore original k_neighbors
            self.k_neighbors = original_k
            
            return similar_items
        
        except Exception as e:
            self.logger.error(f"Failed to get item similarities: {e}")
            return []
    
    def _generate_reason(
        self,
        item: Any,
        score: float,
        user_id: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate explanation for Item-CF recommendation.
        
        Args:
            item: Item object
            score: Recommendation score
            user_id: User database ID
            context: Contextual features
        
        Returns:
            Explanation string
        """
        try:
            # Get user's interaction history
            user_vector = self.matrix_builder.get_user_vector(user_id)
            
            if user_vector is not None:
                user_items = self._get_user_items(user_vector)
                
                if user_items:
                    # Find which user item is most similar to recommended item
                    max_similarity = 0.0
                    most_similar_item_id = None
                    
                    for user_item_id, _ in user_items[:5]:  # Check top 5 user items
                        similar_items = self._find_similar_items(user_item_id)
                        
                        for similar_id, similarity in similar_items:
                            if similar_id == item.id and similarity > max_similarity:
                                max_similarity = similarity
                                most_similar_item_id = user_item_id
                    
                    if most_similar_item_id:
                        similar_item = self.session.get(Item, most_similar_item_id)
                        if similar_item:
                            return f"Similar to '{similar_item.item_name}' which you used"
            
            # Fallback
            return "Similar to items you've used"
        
        except Exception as e:
            self.logger.error(f"Failed to generate reason: {e}")
            return "Recommended based on item similarity"


if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres
    
    setup_logging()
    
    pg = get_postgres()
    
    with pg.transaction() as session:
        # Create recommender
        config = {
            "similarity_metric": "cosine",
            "k_neighbors": 20,
            "min_common_users": 3,
            "normalize": True
        }
        
        recommender = ItemCFRecommender(session, config)
        
        # Fit model
        print("\n" + "="*70)
        print("🔧 FITTING ITEM-CF MODEL")
        print("="*70)
        
        recommender.fit(
            weighting="count",
            min_interactions=2,
            lookback_days=90
        )
        
        # Get matrix info
        info = recommender.matrix_builder.get_matrix_info()
        print("\n📊 Matrix Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test recommendation
        if recommender.matrix_builder.n_users > 0:
            test_user_id = recommender.matrix_builder.idx_to_user_id[0]
            
            print(f"\n🎯 Generating recommendations for user {test_user_id}")
            
            recommendations = recommender.recommend(
                user_id=test_user_id,
                limit=5,
                min_score=0.0
            )
            
            print(f"\n📋 Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                print(f"  {rec.rank}. {rec.item_name}")
                print(f"     Score: {rec.score:.4f}")
                print(f"     Reason: {rec.reason}")
        
        # Test item similarity
        if recommender.matrix_builder.n_items > 0:
            test_item_id = recommender.matrix_builder.idx_to_item_id[0]
            test_item = session.get(Item, test_item_id)
            
            if test_item:
                print(f"\n🔗 Similar items to '{test_item.item_name}':")
                
                similar_items = recommender.get_item_similarities(
                    item_id=test_item_id,
                    top_k=5
                )
                
                for similar_id, similarity in similar_items:
                    similar_item = session.get(Item, similar_id)
                    if similar_item:
                        print(f"  - {similar_item.item_name}: {similarity:.4f}")