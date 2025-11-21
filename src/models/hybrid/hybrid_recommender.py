from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from src.models.base_recommender import BaseRecommender, RecommendationResult
from src.models.collaborative.user_cf import UserCFRecommender
from src.models.collaborative.item_cf import ItemCFRecommender
from src.features.context_features import ContextFeatureExtractor
from src.database.models import Item, User

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender combining multiple strategies.

    Combines:
    - User-CF: Recommendations from similar users
    - Item-CF: Recommendations from similar items
    - Context: Time/device/location adjustments
    - Recency: Boost for recently popular items

    Configuration:
        - user_cf_weight: Weight for User-CF (default: 0.3)
        - item_cf_weight: Weight for Item-CF (default: 0.3)
        - context_weight: Weight for context features (default: 0.2)
        - recency_weight: Weight for recency boost (default: 0.2)
        - enable_mmr: Enable diversity via MMR (default: True)
        - mmr_lambda: MMR diversity parameter (default: 0.5)
        - recency_cache_ttl: Recency cache TTL seconds (default: 3600)
    """

    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hybrid recommender.

        Args:
            session: Database session
            config: Configuration dictionary
        """
        super().__init__(session, config)

        # Get weights from config
        self.user_cf_weight = float(self.get_config("user_cf_weight", 0.3))
        self.item_cf_weight = float(self.get_config("item_cf_weight", 0.3))
        self.context_weight = float(self.get_config("context_weight", 0.2))
        self.recency_weight = float(self.get_config("recency_weight", 0.2))

        # Validate weights sum to 1.0
        total_weight = (
            self.user_cf_weight +
            self.item_cf_weight +
            self.context_weight +
            self.recency_weight
        )

        if not np.isclose(total_weight, 1.0):
            self.logger.warning(
                f"Weights sum to {total_weight:.4f}, normalizing to 1.0"
            )
            self._normalize_weights()

        # MMR settings
        self.enable_mmr = bool(self.get_config("enable_mmr", True))
        self.mmr_lambda = float(self.get_config("mmr_lambda", 0.5))

        # Recency cache TTL (seconds)
        self.recency_cache_ttl = int(self.get_config("recency_cache_ttl", 3600))

        # Initialize component recommenders
        self.user_cf = UserCFRecommender(session, config)
        self.item_cf = ItemCFRecommender(session, config)

        # Context feature extractor
        self.context_extractor = ContextFeatureExtractor()

        # Cache for recency scores
        self._recency_scores_cache: Optional[Dict[int, float]] = None
        self._recency_cache_time: Optional[datetime] = None

    def fit(self, **kwargs) -> None:
        """
        Train all component models.

        Args:
            **kwargs: Arguments passed to component models
        """
        try:
            self.logger.info("Fitting Hybrid Recommender")

            # Fit User-CF
            if self.user_cf_weight > 0:
                self.logger.info("Fitting User-CF component...")
                self.user_cf.fit(**kwargs)

            # Fit Item-CF
            if self.item_cf_weight > 0:
                self.logger.info("Fitting Item-CF component...")
                self.item_cf.fit(**kwargs)

            # Precompute recency scores
            if self.recency_weight > 0:
                self.logger.info("Computing recency scores...")
                self._compute_recency_scores()

            self.is_fitted = True
            self.logger.info("Hybrid Recommender fitted successfully")

        except Exception as e:
            self.logger.error(f"Failed to fit Hybrid Recommender: {e}", exc_info=True)
            raise

    def get_model_name(self) -> str:
        """Return model identifier."""
        return "hybrid_recommender"

    def _compute_scores(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None,
        candidate_items: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute hybrid recommendation scores.

        Args:
            user_id: User database ID
            context: Contextual features
            candidate_items: List of candidate item IDs

        Returns:
            Dictionary mapping item_id to hybrid score
        """
        try:
            if not self.is_fitted:
                self.logger.error("Model not fitted yet")
                return {}

            hybrid_scores: Dict[int, float] = {}

            # 1. Get User-CF scores
            if self.user_cf_weight > 0:
                user_cf_scores = self._get_user_cf_scores(
                    user_id,
                    candidate_items
                )
                self._merge_scores(
                    hybrid_scores,
                    user_cf_scores,
                    self.user_cf_weight
                )

            # 2. Get Item-CF scores
            if self.item_cf_weight > 0:
                item_cf_scores = self._get_item_cf_scores(
                    user_id,
                    candidate_items
                )
                self._merge_scores(
                    hybrid_scores,
                    item_cf_scores,
                    self.item_cf_weight
                )

            # 3. Apply context adjustment
            if self.context_weight > 0 and context:
                context_scores = self._get_context_scores(
                    user_id,
                    context,
                    candidate_items
                )
                self._merge_scores(
                    hybrid_scores,
                    context_scores,
                    self.context_weight
                )

            # 4. Apply recency boost
            if self.recency_weight > 0:
                recency_scores = self._get_recency_scores(candidate_items)
                self._merge_scores(
                    hybrid_scores,
                    recency_scores,
                    self.recency_weight
                )

            # 5. Apply MMR for diversity (optional)
            if self.enable_mmr and len(hybrid_scores) > 1:
                hybrid_scores = self._apply_mmr(
                    hybrid_scores,
                    user_id
                )

            return hybrid_scores

        except Exception as e:
            self.logger.error(f"Failed to compute hybrid scores: {e}", exc_info=True)
            return {}

    # -------------------------
    # Component score getters
    # -------------------------

    def _get_user_cf_scores(
        self,
        user_id: int,
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """Get scores from User-CF component."""
        try:
            scores = self.user_cf._compute_scores(
                user_id,
                context=None,
                candidate_items=candidate_items
            )
            scores = self._normalize_scores(scores)
            self.logger.debug(f"User-CF returned {len(scores)} scores")
            return scores
        except Exception as e:
            self.logger.error(f"User-CF scoring failed: {e}")
            return {}

    def _get_item_cf_scores(
        self,
        user_id: int,
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """Get scores from Item-CF component."""
        try:
            scores = self.item_cf._compute_scores(
                user_id,
                context=None,
                candidate_items=candidate_items
            )
            scores = self._normalize_scores(scores)
            self.logger.debug(f"Item-CF returned {len(scores)} scores")
            return scores
        except Exception as e:
            self.logger.error(f"Item-CF scoring failed: {e}")
            return {}

    # -------------------------
    # Context scoring
    # -------------------------

    def _get_context_scores(
        self,
        user_id: int,
        context: Dict[str, Any],
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """
        Compute context-based score adjustments.

        Returns:
            Dict[item_id -> score]
        """
        try:
            temporal_context = self.context_extractor.extract_temporal_context(
                context.get('timestamp')
            )

            device_context = self.context_extractor.extract_device_context(
                context.get('browser'),
                context.get('os'),
                context.get('device_id')
            )

            # Build simple numeric context vector
            context_vector = self._build_context_vector(temporal_context, device_context)

            # Get items to score
            if candidate_items:
                items_to_score = candidate_items
            else:
                items = self.session.query(Item).filter(Item.is_active == True).all()
                items_to_score = [item.id for item in items]

            context_scores: Dict[int, float] = {}

            for item_id in items_to_score:
                item = self.session.get(Item, item_id)
                if not item:
                    continue

                score = self._compute_context_score(
                    item,
                    temporal_context,
                    device_context,
                    context_vector
                )

                context_scores[item_id] = float(score)

            context_scores = self._normalize_scores(context_scores)
            self.logger.debug(f"Context scoring returned {len(context_scores)} scores")
            return context_scores

        except Exception as e:
            self.logger.error(f"Context scoring failed: {e}", exc_info=True)
            return {}

    def _build_context_vector(self, temporal_context: Dict[str, Any], device_context: Dict[str, Any]) -> np.ndarray:
        """
        Turn a small set of context features into a numeric vector.
        This is intentionally simple and meant to be lightweight:
        - time_of_day one-hot (5)
        - is_business_hours (1)
        - is_mobile/is_desktop/is_tablet (3)
        -> total length = 9
        """
        try:
            # time_of_day ordering matches ContextFeatureExtractor TimeOfDay enum values
            time_buckets = ["early_morning", "morning", "afternoon", "evening", "night"]
            tod = temporal_context.get("time_of_day")
            tod_vec = [1.0 if tod == t else 0.0 for t in time_buckets]

            business = [1.0 if temporal_context.get("is_business_hours") else 0.0]

            dev_mobile = 1.0 if device_context.get("is_mobile") else 0.0
            dev_desktop = 1.0 if device_context.get("is_desktop") else 0.0
            dev_tablet = 1.0 if device_context.get("is_tablet") else 0.0

            vec = np.array(tod_vec + business + [dev_mobile, dev_desktop, dev_tablet], dtype=float)
            # normalize vector to unit length to allow cosine comparisons
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception:
            # fallback to zeros
            return np.zeros(9, dtype=float)

    def _compute_context_score(
        self,
        item: Item,
        temporal_context: Dict[str, Any],
        device_context: Dict[str, Any],
        context_vector: np.ndarray
    ) -> float:
        """
        Compute context score for a single item.
        - Prefer embedding-based cosine similarity if item has a numeric feature vector/embedding.
        - Otherwise fallback to rule-based heuristics (previous behavior).
        """
        try:
            # Try embedding/feature vector approach first
            item_embedding = None
            # look for common attribute or related ItemFeature with embedding
            try:
                # If item has attribute 'feature_vector' or 'embedding' already accessible
                if hasattr(item, "feature_vector") and item.feature_vector is not None:
                    item_embedding = np.asarray(item.feature_vector, dtype=float)
                elif hasattr(item, "embedding") and item.embedding is not None:
                    item_embedding = np.asarray(item.embedding, dtype=float)
                else:
                    # Try ItemFeature table (if exists) to fetch embedding field (optional)
                    from src.database.models import ItemFeature
                    feat = self.session.query(ItemFeature).filter(ItemFeature.item_id == item.id).first()
                    if feat is not None and getattr(feat, "embedding", None) is not None:
                        item_embedding = np.asarray(feat.embedding, dtype=float)
            except Exception:
                item_embedding = None

            if item_embedding is not None and context_vector is not None and item_embedding.size > 0:
                # reduce/expand item_embedding to match context vector size if necessary using simple projection:
                # If lengths differ, compute cosine on min-dim prefix to be robust.
                try:
                    min_len = min(context_vector.size, item_embedding.size)
                    a = context_vector[:min_len]
                    b = item_embedding[:min_len]
                    denom = (np.linalg.norm(a) * np.linalg.norm(b))
                    if denom > 0:
                        cos = float(np.dot(a, b) / denom)
                        # Map cosine [-1,1] -> [0,1]
                        return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))
                except Exception:
                    # fallback to heuristic
                    pass

            # Fallback: previous rule-based scoring kept, starting from base 0.5
            score = 0.5

            # Time-of-day adjustment
            time_of_day = temporal_context.get('time_of_day')
            if time_of_day == 'morning' and getattr(item, "category", "") == 'paraphrasing':
                score += 0.2
            elif time_of_day == 'evening' and getattr(item, "category", "") == 'creative':
                score += 0.1

            # Business hours adjustment
            if temporal_context.get('is_business_hours') and getattr(item, "is_premium", False):
                score += 0.1

            # Device type adjustment
            if device_context.get('is_mobile') and getattr(item, "item_type", "") == 'quick_tool':
                score += 0.15

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Context score compute error for item {getattr(item,'id', None)}: {e}")
            return 0.0

    # -------------------------
    # Recency scoring + cache
    # -------------------------

    def _get_recency_scores(
        self,
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """
        Get recency-based scores. Uses TTL-based in-memory cache.
        """
        try:
            # Use cached scores if fresh
            if self._recency_scores_cache and self._recency_cache_time:
                age = (datetime.now() - self._recency_cache_time).total_seconds()
                if age < self.recency_cache_ttl:
                    if candidate_items:
                        return {iid: self._recency_scores_cache.get(iid, 0.0) for iid in candidate_items}
                    return self._recency_scores_cache.copy()

            # Recompute and return subset if asked
            self._compute_recency_scores()
            if candidate_items:
                return {iid: self._recency_scores_cache.get(iid, 0.0) for iid in candidate_items}
            return self._recency_scores_cache.copy() if self._recency_scores_cache else {}

        except Exception as e:
            self.logger.error(f"Recency scoring failed: {e}", exc_info=True)
            return {}

    def _compute_recency_scores(self) -> None:
        """Precompute recency scores for all items."""
        try:
            from src.database.models import ItemFeature

            item_features = self.session.query(ItemFeature).all()
            if not item_features:
                self._recency_scores_cache = {}
                self._recency_cache_time = datetime.now()
                return

            scores: Dict[int, float] = {}
            for feat in item_features:
                trending = getattr(feat, "trending_score", 0.0) or 0.0
                freshness = getattr(feat, "freshness_score", 0.0) or 0.0
                recency_score = 0.6 * float(trending) + 0.4 * float(freshness)
                scores[int(feat.item_id)] = recency_score

            # Normalize and cache
            scores = self._normalize_scores(scores)
            self._recency_scores_cache = scores
            self._recency_cache_time = datetime.now()
            self.logger.debug(f"Computed recency scores for {len(scores)} items")

        except Exception as e:
            self.logger.error(f"Failed to compute recency scores: {e}", exc_info=True)
            self._recency_scores_cache = {}
            self._recency_cache_time = datetime.now()

    # -------------------------
    # Scoring utilities
    # -------------------------

    def _merge_scores(
        self,
        target_scores: Dict[int, float],
        source_scores: Dict[int, float],
        weight: float
    ) -> None:
        """
        Merge source scores into target scores with weight.
        """
        for item_id, score in source_scores.items():
            if item_id not in target_scores:
                target_scores[item_id] = 0.0
            target_scores[item_id] += weight * score

    def _normalize_scores(
        self,
        scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Normalize scores to [0, 1] range.
        """
        if not scores:
            return {}
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {iid: 0.5 for iid in scores}
        return {iid: (v - mn) / (mx - mn) for iid, v in scores.items()}

    # -------------------------
    # MMR (Maximal Marginal Relevance)
    # -------------------------

    def _apply_mmr(self, scores: Dict[int, float], user_id: int) -> Dict[int, float]:
        """
        Re-rank items using MMR. Returns new dict preserving score values for selected items.
        Implementation:
        - Greedy selection: pick item that maximizes lambda * relevance - (1-lambda) * max_similarity_to_selected
        - item similarity uses embeddings when available, otherwise category equality fallback.
        """
        try:
            lambda_ = float(self.mmr_lambda)
            if lambda_ < 0 or lambda_ > 1:
                lambda_ = 0.5

            # Prepare items sorted by original score (desc)
            items_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            cand_ids = [iid for iid, _ in items_sorted]

            selected: List[int] = []
            selected_scores: Dict[int, float] = {}

            # Precompute a lightweight similarity matrix (on-the-fly)
            # We'll cache item embeddings when available to avoid repeated DB hits
            emb_cache: Dict[int, Optional[np.ndarray]] = {}
            def get_emb(iid: int) -> Optional[np.ndarray]:
                if iid in emb_cache:
                    return emb_cache[iid]
                try:
                    # Try ItemFeature.embedding
                    from src.database.models import ItemFeature
                    feat = self.session.query(ItemFeature).filter(ItemFeature.item_id == iid).first()
                    if feat and getattr(feat, "embedding", None) is not None:
                        arr = np.asarray(feat.embedding, dtype=float)
                        emb_cache[iid] = arr
                        return arr
                    # Try Item model attribute fallback
                    it = self.session.get(Item, iid)
                    if it and getattr(it, "feature_vector", None) is not None:
                        arr = np.asarray(it.feature_vector, dtype=float)
                        emb_cache[iid] = arr
                        return arr
                except Exception:
                    pass
                emb_cache[iid] = None
                return None

            def item_similarity(a: int, b: int) -> float:
                # Try embeddings
                emb_a = get_emb(a)
                emb_b = get_emb(b)
                if emb_a is not None and emb_b is not None and emb_a.size > 0 and emb_b.size > 0:
                    min_len = min(emb_a.size, emb_b.size)
                    if min_len == 0:
                        return 0.0
                    aa = emb_a[:min_len]
                    bb = emb_b[:min_len]
                    denom = (np.linalg.norm(aa) * np.linalg.norm(bb))
                    if denom > 0:
                        sim = float(np.dot(aa, bb) / denom)
                        return float(np.clip(sim, -1.0, 1.0))
                # Fallback: category equality
                try:
                    it_a = self.session.get(Item, a)
                    it_b = self.session.get(Item, b)
                    if it_a and it_b and getattr(it_a, "category", None) and getattr(it_b, "category", None):
                        return 1.0 if it_a.category == it_b.category else 0.0
                except Exception:
                    pass
                return 0.0

            # Greedy MMR selection
            remaining = cand_ids.copy()
            while remaining:
                best_item = None
                best_score = -np.inf
                for iid in remaining:
                    relevance = scores.get(iid, 0.0)
                    if not selected:
                        mmr_val = lambda_ * relevance
                    else:
                        max_sim = max([item_similarity(iid, s) for s in selected]) if selected else 0.0
                        # Convert similarity [-1,1] to [0,1] if needed
                        max_sim = (max_sim + 1.0) / 2.0 if max_sim < 0 or max_sim > 1 else max_sim
                        mmr_val = lambda_ * relevance - (1.0 - lambda_) * max_sim
                    if mmr_val > best_score:
                        best_score = mmr_val
                        best_item = iid

                if best_item is None:
                    break
                selected.append(best_item)
                selected_scores[best_item] = scores.get(best_item, 0.0)
                remaining.remove(best_item)

            self.logger.debug(f"MMR selected {len(selected_scores)} items (lambda={lambda_})")
            # Return dict preserving original scores but in MMR order (dict ordering in py3.7+ keeps insertion order)
            ordered = {iid: selected_scores[iid] for iid in selected}
            return ordered

        except Exception as e:
            self.logger.error(f"MMR application failed: {e}", exc_info=True)
            return scores

    # -------------------------
    # Weight normalization
    # -------------------------

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = (
            self.user_cf_weight +
            self.item_cf_weight +
            self.context_weight +
            self.recency_weight
        )
        if total > 0:
            self.user_cf_weight /= total
            self.item_cf_weight /= total
            self.context_weight /= total
            self.recency_weight /= total

    # -------------------------
    # Explainability
    # -------------------------

    def _generate_reason(
        self,
        item: Any,
        score: float,
        user_id: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate explanation for hybrid recommendation.
        """
        reasons = []
        if self.user_cf_weight > 0:
            reasons.append("similar users")
        if self.item_cf_weight > 0:
            reasons.append("your preferences")
        if self.context_weight > 0 and context:
            tod = context.get("time_of_day") or context.get("timestamp")
            if tod:
                reasons.append(f"good for {tod}")
        if self.recency_weight > 0:
            reasons.append("trending")
        if reasons:
            return f"Recommended based on: {', '.join(reasons)}"
        return "Recommended for you"


if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres

    setup_logging()
    pg = get_postgres()

    with pg.transaction() as session:
        config = {
            "user_cf_weight": 0.3,
            "item_cf_weight": 0.3,
            "context_weight": 0.2,
            "recency_weight": 0.2,
            "enable_mmr": True,
            "mmr_lambda": 0.5,
            "recency_cache_ttl": 3600
        }
        recommender = HybridRecommender(session, config)
        recommender.fit(
            weighting="count",
            min_interactions=2,
            lookback_days=90
        )

        if recommender.user_cf.matrix_builder.n_users > 0:
            test_user_id = recommender.user_cf.matrix_builder.idx_to_user_id[0]
            context = {
                'timestamp': datetime.now(),
                'browser': 'Chrome',
                'os': 'Windows',
                'time_of_day': 'morning'
            }
            print(f"\n🎯 Generating recommendations for user {test_user_id}")
            recommendations = recommender.recommend(
                user_id=test_user_id,
                context=context,
                limit=5,
                min_score=0.0
            )
            print(f"\n📋 Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                print(f"  {rec.rank}. {rec.item_name}")
                print(f"     Score: {rec.score:.4f}")
                print(f"     Reason: {rec.reason}")
