"""
MMR (Maximal Marginal Relevance) utility module for hybrid recommender systems.

This module provides:
- Greedy MMR selection algorithm
- Embedding-based similarity (cosine)
- Category fallback similarity
- Session-level caching for efficiency
- Robust error handling and numerical stability

Used by: HybridRecommender

Improvements over original:
- OrderedDict return type for guaranteed order
- Configurable embedding attribute names
- Enhanced error handling
- Better logging
"""

from typing import Dict, List, Optional
from collections import OrderedDict
import numpy as np
import logging
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class MMR:
    """
    Maximal Marginal Relevance (MMR) re-ranking for recommendation diversity.

    Parameters:
        session (Session): SQLAlchemy session for DB access
        lambda_ (float): Trade-off between relevance and diversity (0~1)
                        0.0 = pure diversity, 1.0 = pure relevance
        max_items (Optional[int]): Optional cap on number of returned items
        embedding_attrs (List[str]): Attribute names to check for embeddings
                                     (checked in order)
    
    Example:
        >>> mmr = MMR(session, lambda_=0.5, max_items=10)
        >>> scores = {1: 0.9, 2: 0.85, 3: 0.8}
        >>> reranked = mmr.rerank(scores)
    """

    def __init__(
        self,
        session: Session,
        lambda_: float = 0.5,
        max_items: Optional[int] = None,
        embedding_attrs: Optional[List[str]] = None
    ):
        """
        Initialize MMR selector.
        
        Args:
            session: Database session
            lambda_: Relevance-diversity trade-off (0-1)
            max_items: Maximum items to return
            embedding_attrs: List of embedding attribute names to try
        """
        self.session = session
        self.lambda_ = float(np.clip(lambda_, 0.0, 1.0))
        self.max_items = max_items
        self.embedding_attrs = embedding_attrs or ["embedding", "feature_vector"]
        self._embedding_cache: Dict[int, Optional[np.ndarray]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Validate lambda
        if not 0.0 <= lambda_ <= 1.0:
            self.logger.warning(
                f"lambda_ {lambda_} outside [0,1], clipped to {self.lambda_}"
            )

    # ====================================================
    # Public API
    # ====================================================
    def rerank(self, scores: Dict[int, float]) -> OrderedDict[int, float]:
        """
        Apply MMR re-ranking to item scores.

        Args:
            scores: Dictionary {item_id -> relevance score}

        Returns:
            OrderedDict of item_id -> original score (re-ordered by MMR)
            
        Note:
            The returned scores are the ORIGINAL relevance scores,
            but the ORDER is determined by MMR.
        """
        if not scores:
            self.logger.debug("Empty scores provided to MMR")
            return OrderedDict()
        
        if len(scores) == 1:
            return OrderedDict(scores)

        try:
            # Sort items by original relevance
            items_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            candidates = [iid for iid, _ in items_sorted]
            selected: List[int] = []
            selected_scores: OrderedDict[int, float] = OrderedDict()

            # Greedy selection loop
            while candidates:
                best_item, best_mmr_score = None, -np.inf
                
                for iid in candidates:
                    relevance = scores[iid]
                    
                    if not selected:
                        # First item: pure relevance
                        mmr_score = self.lambda_ * relevance
                    else:
                        # Subsequent items: relevance - diversity penalty
                        max_sim = self._compute_max_similarity(iid, selected)
                        mmr_score = (
                            self.lambda_ * relevance - 
                            (1 - self.lambda_) * max_sim
                        )

                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_item = iid

                if best_item is None:
                    self.logger.warning("MMR selection stalled, returning partial results")
                    break

                selected.append(best_item)
                selected_scores[best_item] = scores[best_item]  # Original score
                candidates.remove(best_item)

                if self.max_items and len(selected) >= self.max_items:
                    break

            self.logger.debug(
                f"MMR selected {len(selected)}/{len(scores)} items "
                f"(lambda={self.lambda_:.2f})"
            )
            
            return selected_scores

        except Exception as e:
            self.logger.error(f"MMR rerank failed: {e}", exc_info=True)
            # Fallback: return top-k by relevance
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if self.max_items:
                sorted_items = sorted_items[:self.max_items]
            return OrderedDict(sorted_items)

    # ====================================================
    # Similarity utilities
    # ====================================================
    def _compute_max_similarity(self, item_id: int, selected_ids: List[int]) -> float:
        """
        Compute maximum similarity between item and selected items.
        
        Args:
            item_id: Target item ID
            selected_ids: List of already selected item IDs
        
        Returns:
            Maximum similarity score in [0, 1]
        """
        try:
            if not selected_ids:
                return 0.0
            
            similarities = [
                self._similarity(item_id, sid) 
                for sid in selected_ids
            ]
            
            return max(similarities, default=0.0)
        
        except Exception as e:
            self.logger.error(
                f"Max similarity computation failed for item {item_id}: {e}"
            )
            return 0.0

    def _similarity(self, a: int, b: int) -> float:
        """
        Compute similarity between two items.
        
        Strategy:
        1. Try embedding-based cosine similarity
        2. Fallback to category equality
        3. Return 0.0 if all fails
        
        Args:
            a: First item ID
            b: Second item ID
        
        Returns:
            Similarity score in [0, 1]
        """
        try:
            # Try embedding similarity
            emb_a = self._get_embedding(a)
            emb_b = self._get_embedding(b)

            if emb_a is not None and emb_b is not None:
                if emb_a.size > 0 and emb_b.size > 0:
                    sim = self._cosine_similarity(emb_a, emb_b)
                    if sim is not None:
                        return sim

            # Fallback: category equality
            return self._category_similarity(a, b)

        except Exception as e:
            self.logger.warning(
                f"Similarity computation failed for ({a}, {b}): {e}"
            )
            return 0.0

    def _cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray
    ) -> Optional[float]:
        """
        Compute robust cosine similarity between two vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
        
        Returns:
            Similarity in [0, 1] or None if computation fails
        """
        try:
            # Ensure same dimensions
            dim = min(vec_a.size, vec_b.size)
            va, vb = vec_a[:dim], vec_b[:dim]
            
            # Compute norms
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            # Cosine similarity
            cosine = float(np.dot(va, vb) / (norm_a * norm_b))
            
            # Map from [-1, 1] to [0, 1]
            similarity = (cosine + 1.0) / 2.0
            
            # Clamp to valid range
            return float(np.clip(similarity, 0.0, 1.0))
        
        except Exception as e:
            self.logger.debug(f"Cosine similarity computation failed: {e}")
            return None

    def _category_similarity(self, a: int, b: int) -> float:
        """
        Compute category-based similarity (fallback).
        
        Args:
            a: First item ID
            b: Second item ID
        
        Returns:
            1.0 if same category, 0.0 otherwise
        """
        try:
            from src.database.models import Item
            
            item_a = self.session.get(Item, a)
            item_b = self.session.get(Item, b)
            
            if not item_a or not item_b:
                return 0.0
            
            cat_a = getattr(item_a, "category", None)
            cat_b = getattr(item_b, "category", None)
            
            if cat_a and cat_b and cat_a == cat_b:
                return 1.0
            
            return 0.0
        
        except Exception as e:
            self.logger.debug(f"Category similarity failed: {e}")
            return 0.0

    def _get_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """
        Get embedding vector for item (with caching).
        
        Args:
            item_id: Item database ID
        
        Returns:
            Numpy array embedding or None
        """
        # Check cache first
        if item_id in self._embedding_cache:
            return self._embedding_cache[item_id]

        embedding = None
        
        try:
            from src.database.models import ItemFeature, Item
            
            # Try ItemFeature table first
            feat = self.session.query(ItemFeature).filter(
                ItemFeature.item_id == item_id
            ).first()
            
            if feat:
                embedding = self._extract_embedding_from_object(feat)
            
            # Try Item table if not found
            if embedding is None:
                item = self.session.get(Item, item_id)
                if item:
                    embedding = self._extract_embedding_from_object(item)
        
        except Exception as e:
            self.logger.debug(f"Embedding fetch failed for item {item_id}: {e}")
        
        # Cache result (even if None)
        self._embedding_cache[item_id] = embedding
        return embedding

    def _extract_embedding_from_object(self, obj: object) -> Optional[np.ndarray]:
        """
        Extract embedding from database object.
        
        Tries multiple attribute names in order.
        
        Args:
            obj: Database object (Item or ItemFeature)
        
        Returns:
            Numpy array or None
        """
        for attr_name in self.embedding_attrs:
            if hasattr(obj, attr_name):
                value = getattr(obj, attr_name)
                if value is not None:
                    try:
                        arr = np.asarray(value, dtype=float)
                        if arr.size > 0:
                            return arr
                    except Exception as e:
                        self.logger.debug(
                            f"Failed to convert {attr_name} to array: {e}"
                        )
        
        return None

    def clear_cache(self) -> None:
        """Clear embedding cache (useful for testing or memory management)."""
        self._embedding_cache.clear()
        self.logger.debug("MMR embedding cache cleared")

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache info
        """
        return {
            "cached_embeddings": len(self._embedding_cache),
            "cache_hits": sum(1 for v in self._embedding_cache.values() if v is not None),
            "cache_misses": sum(1 for v in self._embedding_cache.values() if v is None)
        }


if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres
    
    setup_logging()
    
    pg = get_postgres()
    
    with pg.transaction() as session:
        print("\n" + "="*70)
        print("🎯 MMR MODULE DEMO")
        print("="*70)
        
        # Create MMR selector
        mmr = MMR(session, lambda_=0.5, max_items=5)
        
        # Mock scores
        scores = {
            1: 0.95,
            2: 0.90,
            3: 0.88,
            4: 0.85,
            5: 0.82,
            6: 0.80,
            7: 0.78,
            8: 0.75,
        }
        
        print(f"\n📊 Original Scores:")
        for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  Item {item_id}: {score:.3f}")
        
        # Apply MMR
        reranked = mmr.rerank(scores)
        
        print(f"\n✅ MMR Re-ranked (λ={mmr.lambda_}):")
        for i, (item_id, score) in enumerate(reranked.items(), 1):
            print(f"  {i}. Item {item_id}: {score:.3f}")
        
        # Cache info
        cache_info = mmr.get_cache_info()
        print(f"\n📦 Cache Info:")
        for key, value in cache_info.items():
            print(f"  {key}: {value}")