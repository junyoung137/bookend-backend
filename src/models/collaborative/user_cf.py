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


class UserCFRecommender(BaseRecommender):
    """
    User-based Collaborative Filtering recommender.
    
    Finds similar users and recommends items those users interacted with.
    
    Configuration:
        - similarity_metric: Similarity metric to use (default: 'cosine')
        - k_neighbors: Number of similar users to consider (default: 20)
        - min_common_items: Minimum common items for similarity (default: 3)
        - normalize: Whether to normalize vectors (default: True)
    """
    
    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize User-CF recommender.
        
        Args:
            session: Database session
            config: Configuration dictionary
        """
        super().__init__(session, config)
        
        # Get configuration
        self.similarity_metric = SimilarityMetric(
            self.get_config("similarity_metric", "cosine")
        )
        self.k_neighbors = self.get_config("k_neighbors", 20)
        self.min_common_items = self.get_config("min_common_items", 3)
        self.normalize = self.get_config("normalize", True)
        
        # Matrix builder
        self.matrix_builder = InteractionMatrixBuilder(session, config)
        
        # Precomputed similarity matrix (cached)
        self.similarity_matrix: Optional[np.ndarray] = None
        self.similarity_computed_at: Optional[datetime] = None
    
    def fit(self, **kwargs) -> None:
        """
        Train the User-CF model.
        
        Builds interaction matrix and optionally precomputes similarity matrix.
        
        Args:
            **kwargs: Additional arguments for matrix building
        """
        try:
            self.logger.info("Fitting User-CF model")
            
            # Build interaction matrix
            self.logger.info("Building interaction matrix...")
            self.matrix_builder.build_matrix(
                weighting=kwargs.get("weighting", "binary"),
                min_interactions=kwargs.get("min_interactions", 2),
                lookback_days=kwargs.get("lookback_days", None),
                normalize=self.normalize
            )
            
            # Precompute similarity matrix if requested
            if kwargs.get("precompute_similarity", False):
                self.logger.info("Precomputing user similarity matrix...")
                self._precompute_similarity_matrix()
            
            self.is_fitted = True
            self.logger.info("User-CF model fitted successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to fit User-CF model: {e}", exc_info=True)
            raise
    
    def get_model_name(self) -> str:
        """Return model identifier."""
        return f"user_cf_{self.similarity_metric.value}"
    
    def _compute_scores(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None,
        candidate_items: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute recommendation scores using User-CF.
        
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
            
            # Find similar users
            similar_users = self._find_similar_users(user_id, user_vector)
            
            if not similar_users:
                self.logger.debug(f"No similar users found for user {user_id}")
                return {}
            
            # Compute scores for candidate items
            scores = self._compute_item_scores(
                user_vector,
                similar_users,
                candidate_items
            )
            
            return scores
        
        except Exception as e:
            self.logger.error(f"Failed to compute scores: {e}", exc_info=True)
            return {}
    
    def _find_similar_users(
        self,
        user_id: int,
        user_vector: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Find K most similar users to the target user.
        
        Args:
            user_id: Target user database ID
            user_vector: Target user's interaction vector
        
        Returns:
            List of (similar_user_id, similarity_score) tuples
        """
        try:
            # Get user index in matrix
            user_idx = self.matrix_builder.user_id_to_idx.get(user_id)
            
            if user_idx is None:
                return []
            
            # Use precomputed similarity matrix if available
            if self.similarity_matrix is not None:
                similarities = self.similarity_matrix[user_idx, :]
            else:
                # Compute similarities on-the-fly
                similarities = self._compute_user_similarities(user_vector)
            
            # Get top-K similar users (excluding self)
            similar_users = []
            
            for other_idx in np.argsort(-similarities)[:self.k_neighbors + 1]:
                if other_idx == user_idx:
                    continue  # Skip self
                
                other_user_id = self.matrix_builder.idx_to_user_id[other_idx]
                similarity = similarities[other_idx]
                
                if similarity > 0:
                    # Check minimum common items
                    if self._has_min_common_items(user_vector, other_idx):
                        similar_users.append((other_user_id, similarity))
                
                if len(similar_users) >= self.k_neighbors:
                    break
            
            self.logger.debug(
                f"Found {len(similar_users)} similar users for user {user_id}"
            )
            
            return similar_users
        
        except Exception as e:
            self.logger.error(f"Failed to find similar users: {e}")
            return []
    
    def _compute_user_similarities(
        self,
        user_vector: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between target user and all other users.
        
        Args:
            user_vector: Target user's interaction vector
        
        Returns:
            Array of similarity scores
        """
        try:
            matrix = self.matrix_builder.matrix
            n_users = matrix.shape[0]
            
            similarities = np.zeros(n_users)
            
            for idx in range(n_users):
                other_vector = matrix[idx, :].toarray().flatten()
                
                if self.similarity_metric == SimilarityMetric.COSINE:
                    sim = cosine_similarity(user_vector, other_vector)
                elif self.similarity_metric == SimilarityMetric.PEARSON:
                    sim = pearson_correlation(
                        user_vector,
                        other_vector,
                        min_overlap=self.min_common_items
                    )
                else:
                    # Default to cosine
                    sim = cosine_similarity(user_vector, other_vector)
                
                similarities[idx] = sim
            
            return similarities
        
        except Exception as e:
            self.logger.error(f"Failed to compute similarities: {e}")
            return np.zeros(n_users)
    
    def _has_min_common_items(
        self,
        user_vector: np.ndarray,
        other_user_idx: int
    ) -> bool:
        """
        Check if two users have minimum number of common items.
        
        Args:
            user_vector: Target user's vector
            other_user_idx: Other user's matrix index
        
        Returns:
            True if minimum common items criterion is met
        """
        try:
            other_vector = self.matrix_builder.matrix[other_user_idx, :].toarray().flatten()
            
            # Count common items (both non-zero)
            common_items = np.sum((user_vector > 0) & (other_vector > 0))
            
            return common_items >= self.min_common_items
        
        except Exception as e:
            self.logger.error(f"Failed to check common items: {e}")
            return False
    
    def _compute_item_scores(
        self,
        user_vector: np.ndarray,
        similar_users: List[Tuple[int, float]],
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """
        Compute scores for candidate items based on similar users.
        
        Args:
            user_vector: Target user's interaction vector
            similar_users: List of (user_id, similarity) tuples
            candidate_items: List of candidate item IDs (None = all items)
        
        Returns:
            Dictionary mapping item_id to score
        """
        try:
            scores = {}
            
            # Get items already interacted with by target user
            user_items = set(np.where(user_vector > 0)[0])
            
            # Aggregate scores from similar users
            for similar_user_id, similarity in similar_users:
                similar_user_vector = self.matrix_builder.get_user_vector(similar_user_id)
                
                if similar_user_vector is None:
                    continue
                
                # Get items this similar user interacted with
                for item_idx in np.where(similar_user_vector > 0)[0]:
                    # Skip items user already interacted with
                    if item_idx in user_items:
                        continue
                    
                    item_id = self.matrix_builder.idx_to_item_id[item_idx]
                    
                    # Skip if not in candidate items
                    if candidate_items and item_id not in candidate_items:
                        continue
                    
                    # Weighted score: interaction_strength * user_similarity
                    interaction_strength = similar_user_vector[item_idx]
                    weighted_score = interaction_strength * similarity
                    
                    if item_id not in scores:
                        scores[item_id] = 0.0
                    
                    scores[item_id] += weighted_score
            
            # Normalize scores by sum of similarities
            if scores:
                total_similarity = sum(sim for _, sim in similar_users)
                if total_similarity > 0:
                    scores = {
                        item_id: score / total_similarity
                        for item_id, score in scores.items()
                    }
            
            return scores
        
        except Exception as e:
            self.logger.error(f"Failed to compute item scores: {e}")
            return {}
    
    def _precompute_similarity_matrix(self) -> None:
        """
        Precompute pairwise user similarity matrix.
        
        Warning: Memory-intensive for large user bases.
        Only use if n_users is reasonable (<10,000).
        """
        try:
            matrix = self.matrix_builder.matrix
            n_users = matrix.shape[0]
            
            if n_users > 10000:
                self.logger.warning(
                    f"Large user base ({n_users}), skipping precomputation"
                )
                return
            
            self.logger.info(f"Precomputing {n_users}x{n_users} similarity matrix...")
            
            # Convert to dense for similarity computation
            matrix_dense = matrix.toarray()
            
            # Compute pairwise similarities
            self.similarity_matrix = compute_similarity_matrix(
                matrix_dense,
                metric=self.similarity_metric
            )
            
            self.similarity_computed_at = datetime.now()
            
            self.logger.info("Similarity matrix precomputed successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to precompute similarity matrix: {e}")
            self.similarity_matrix = None
    
    def _generate_reason(
        self,
        item: Any,
        score: float,
        user_id: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate explanation for User-CF recommendation.
        
        Args:
            item: Item object
            score: Recommendation score
            user_id: User database ID
            context: Contextual features
        
        Returns:
            Explanation string
        """
        try:
            # Get user's similar users count
            user_vector = self.matrix_builder.get_user_vector(user_id)
            
            if user_vector is not None:
                similar_users = self._find_similar_users(user_id, user_vector)
                n_similar = len(similar_users)
                
                if n_similar > 0:
                    return f"Recommended by {n_similar} users with similar preferences"
            
            # Fallback
            return "Popular among users like you"
        
        except Exception as e:
            self.logger.error(f"Failed to generate reason: {e}")
            return "Recommended based on user similarity"


if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres
    
    setup_logging()
    
    pg = get_postgres()
    
    with pg.transaction() as session:
        # Create recommender
        config = {
            "similarity_metric": "cosine",
            "k_neighbors": 10,
            "min_common_items": 2,
            "normalize": True
        }
        
        recommender = UserCFRecommender(session, config)
        
        # Fit model
        print("\n" + "="*70)
        print("🔧 FITTING USER-CF MODEL")
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