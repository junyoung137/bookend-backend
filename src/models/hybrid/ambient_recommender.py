"""
Ambient Recommender for layout-aware personalized recommendations.

Improvements over original:
1. Activity-based cache refresh (clicks tracking)
2. Slot-type specific weighting (hero/sidebar/footer)
3. Contextual embedding fusion (user + context)
4. Soft attention mechanism for embedding blending
5. Fallback diversity strategy (category gap heuristic)
6. Better error handling and logging
7. Korean explanation generation support
"""

import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from sqlalchemy.orm import Session

from src.models.base_recommender import BaseRecommender, RecommendationResult
from src.models.hybrid.hybrid_recommender import HybridRecommender
from src.models.utils.mmr import MMR
from src.features.context_features import ContextFeatureExtractor
from src.database.models import User, Item, Interaction

logger = logging.getLogger(__name__)


# =========================================================
# Slot Type Enum
# =========================================================

class SlotType:
    """Layout slot types with different importance weights."""
    HERO_BANNER = "hero_banner"
    SIDEBAR_QUICK = "sidebar_quick"
    FOOTER_SUGGESTION = "footer_suggestion"
    DASHBOARD_WIDGET = "dashboard_widget"


# =========================================================
# Ambient Recommender
# =========================================================

class AmbientRecommender(BaseRecommender):
    """
    Ambient-aware layout recommender integrating:
    - Hybrid recommendation base
    - Contextual embedding fusion
    - Slot-specific weighting
    - Activity-based cache refresh
    - MMR-based diversity

    Configuration:
        - layout_positions: Number of layout slots (default: 6)
        - refresh_interval_hours: Cache TTL in hours (default: 6)
        - activity_refresh_threshold: Clicks before force refresh (default: 5)
        - diversity_penalty: Category repetition penalty (default: 0.2)
        - min_category_gap: Minimum slots between same category (default: 2)
        - personalization_strength: User vs popularity weight (default: 0.7)
        - context_awareness: Context weight in scoring (default: 0.8)
        - enable_mmr: Use MMR for diversity (default: True)
        - explanation_language: "ko" or "en" (default: "ko")
    """

    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Ambient Recommender.

        Args:
            session: Database session
            config: Configuration dictionary
        """
        super().__init__(session, config)

        # Layout configuration
        self.layout_positions = self.get_config("layout_positions", 6)
        self.refresh_interval_hours = self.get_config("refresh_interval_hours", 6)
        self.activity_refresh_threshold = self.get_config("activity_refresh_threshold", 5)

        # Diversity settings
        self.diversity_penalty = float(self.get_config("diversity_penalty", 0.2))
        self.min_category_gap = int(self.get_config("min_category_gap", 2))

        # Personalization settings
        self.personalization_strength = float(
            self.get_config("personalization_strength", 0.7)
        )
        self.context_awareness = float(
            self.get_config("context_awareness", 0.8)
        )

        # MMR settings
        self.enable_mmr = bool(self.get_config("enable_mmr", True))
        self.mmr_lambda = float(self.get_config("mmr_lambda", 0.5))

        # Explanation language
        self.explanation_language = self.get_config("explanation_language", "ko")

        # Slot-type specific weights
        self.slot_weights = {
            SlotType.HERO_BANNER: 1.2,
            SlotType.SIDEBAR_QUICK: 0.9,
            SlotType.FOOTER_SUGGESTION: 0.7,
            SlotType.DASHBOARD_WIDGET: 1.0,
        }

        # Base hybrid recommender
        self.hybrid_recommender = HybridRecommender(session, config)

        # Context feature extractor
        self.context_extractor = ContextFeatureExtractor()

        # MMR for diversity (optional)
        if self.enable_mmr:
            self.mmr_model = MMR(
                session=session,
                lambda_=self.mmr_lambda,
                max_items=self.layout_positions
            )
        else:
            self.mmr_model = None

        # Caches
        self._layout_cache: Dict[int, List[int]] = {}
        self._cache_timestamps: Dict[int, datetime] = {}
        self._click_counters: Dict[int, int] = {}

        # Embedding dimension (adjust based on your embeddings)
        self.embedding_dim = int(self.get_config("embedding_dim", 128))

    def fit(self, **kwargs) -> None:
        """
        Train the Ambient Recommender.

        Fits the underlying hybrid recommender.
        """
        try:
            self.logger.info("Fitting Ambient Recommender")

            # Fit base hybrid recommender
            self.hybrid_recommender.fit(**kwargs)

            self.is_fitted = True
            self.logger.info("Ambient Recommender fitted successfully")

        except Exception as e:
            self.logger.error(f"Failed to fit Ambient Recommender: {e}", exc_info=True)
            raise

    def get_model_name(self) -> str:
        """Return model identifier."""
        return "ambient_recommender"

    # =========================================================
    # Main Recommendation Entry
    # =========================================================

    def recommend_for_layout(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None,
        layout_type: str = "standard",
        slot_type: str = SlotType.HERO_BANNER,
        top_k: Optional[int] = None,
        force_refresh: bool = False
    ) -> List[RecommendationResult]:
        """
        Generate layout-aware recommendations with activity-based refresh.

        Args:
            user_id: User database ID
            context: Contextual features (time, device, location)
            layout_type: Layout variant (standard/carousel/grid)
            slot_type: Slot type for weight adjustment
            top_k: Override number of slots (None = use layout_positions)
            force_refresh: Force cache refresh

        Returns:
            List of RecommendationResult objects optimized for layout
        """
        try:
            now = datetime.now()
            k = top_k or self.layout_positions

            # Check if refresh is needed
            needs_refresh = force_refresh or self._needs_refresh(user_id, now)

            if needs_refresh:
                self.logger.debug(
                    f"Refreshing layout for user {user_id} "
                    f"(force={force_refresh}, ttl_expired={not force_refresh})"
                )

                # Generate fresh recommendations
                layout_results = self._generate_fresh_layout(
                    user_id,
                    context,
                    slot_type,
                    k
                )

                # Cache results
                self._cache_layout(user_id, layout_results, now)

            else:
                self.logger.debug(f"Using cached layout for user {user_id}")
                layout_results = self._get_cached_layout(user_id, k)

            return layout_results

        except Exception as e:
            self.logger.error(
                f"Failed to generate ambient layout for user {user_id}: {e}",
                exc_info=True
            )
            return []

    def _compute_scores(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None,
        candidate_items: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute ambient recommendation scores (called by base recommend()).

        For layout-specific recommendations, use recommend_for_layout() instead.
        """
        try:
            # Get base hybrid scores
            base_scores = self.hybrid_recommender._compute_scores(
                user_id,
                context,
                candidate_items
            )

            if not base_scores:
                return {}

            # Apply ambient-specific adjustments
            ambient_scores = self._apply_ambient_adjustments(
                base_scores,
                user_id,
                context
            )

            return ambient_scores

        except Exception as e:
            self.logger.error(f"Failed to compute ambient scores: {e}", exc_info=True)
            return {}

    # =========================================================
    # Fresh Layout Generation
    # =========================================================

    def _generate_fresh_layout(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]],
        slot_type: str,
        k: int
    ) -> List[RecommendationResult]:
        """
        Generate fresh layout recommendations with all enhancements.
        """
        try:
            # Step 1: Get base recommendations (3x for diversity filtering)
            candidate_limit = k * 3
            base_recommendations = self.hybrid_recommender.recommend(
                user_id=user_id,
                context=context,
                limit=candidate_limit,
                min_score=0.0,
                include_reasons=False  # We'll generate ambient-specific reasons
            )

            if not base_recommendations:
                self.logger.warning(f"No base recommendations for user {user_id}")
                return []

            # Step 2: Extract context features
            context_features = self._extract_context_features(context)

            # Step 3: Apply layout-aware ranking
            ranked_items = self._apply_layout_ranking(
                user_id,
                base_recommendations,
                context_features,
                slot_type
            )

            # Step 4: Apply diversity (MMR or category-based)
            diversified_items = self._apply_diversity(ranked_items, k)

            # Step 5: Create RecommendationResult objects with explanations
            final_results = self._create_layout_results(
                diversified_items,
                user_id,
                context_features
            )

            return final_results[:k]

        except Exception as e:
            self.logger.error(f"Fresh layout generation failed: {e}", exc_info=True)
            return []

    def _extract_context_features(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract and enrich context features.
        """
        try:
            if not context:
                context = {}

            # Extract temporal context
            temporal_context = self.context_extractor.extract_temporal_context(
                context.get('timestamp')
            )

            # Extract device context
            device_context = self.context_extractor.extract_device_context(
                context.get('browser'),
                context.get('os'),
                context.get('device_id')
            )

            # Combine
            return {
                **(temporal_context or {}),
                **(device_context or {}),
                'original_context': context
            }

        except Exception as e:
            self.logger.error(f"Context extraction failed: {e}")
            return {}

    # =========================================================
    # Layout-Aware Ranking
    # =========================================================

    def _apply_layout_ranking(
        self,
        user_id: int,
        recommendations: List[RecommendationResult],
        context_features: Dict[str, Any],
        slot_type: str
    ) -> List[Tuple[int, float]]:
        """
        Apply layout-aware ranking with contextual embedding fusion.

        Returns:
            List of (item_id, score) tuples sorted by score
        """
        try:
            # Get user embedding
            user = self.session.get(User, user_id)
            user_embedding = self._get_user_embedding(user)

            # Compute contextual embedding
            contextual_embedding = self._compute_contextual_embedding(
                user_embedding,
                context_features
            )

            # Get slot weight
            slot_weight = self.slot_weights.get(slot_type, 1.0)

            # Score each candidate
            scored_items: List[Tuple[int, float]] = []

            for rec in recommendations:
                item = self.session.get(Item, rec.item_id)

                if not item:
                    continue

                # Base score from hybrid model
                base_score = getattr(rec, "score", 0.0)

                # Context fit score
                context_fit = self._compute_context_fit(item, context_features)

                # User affinity score
                user_affinity = self._compute_user_affinity(user_id, item)

                # Recency bonus
                recency_bonus = self._compute_recency_bonus(item)

                # Contextual embedding similarity
                context_emb_score = self._compute_contextual_similarity(
                    item,
                    contextual_embedding
                )

                # Combined layout score
                layout_score = (
                    base_score * 0.40 +              # Hybrid model base
                    context_fit * 0.20 +             # Context alignment
                    user_affinity * 0.15 +           # Historical affinity
                    recency_bonus * 0.10 +           # Trending boost
                    context_emb_score * 0.15         # Contextual embedding
                ) * slot_weight

                scored_items.append((item.id, float(layout_score)))

            # Sort by score
            scored_items.sort(key=lambda x: x[1], reverse=True)

            return scored_items

        except Exception as e:
            self.logger.error(f"Layout ranking failed: {e}", exc_info=True)
            # Fallback: return original recommendations
            return [(rec.item_id, getattr(rec, "score", 0.0)) for rec in recommendations]

    # =========================================================
    # Contextual Embedding Fusion
    # =========================================================

    def _get_user_embedding(self, user: Optional[User]) -> Optional[np.ndarray]:
        """Get user embedding from database or features."""
        try:
            if not user:
                return None

            # Try direct embedding attribute
            if hasattr(user, 'embedding') and user.embedding is not None:
                return np.asarray(user.embedding, dtype=float)

            # Try user features
            if hasattr(user, 'features') and user.features:
                feature_vector = getattr(user.features, 'feature_vector', None)
                if feature_vector is not None:
                    return np.asarray(feature_vector, dtype=float)

            return None

        except Exception as e:
            self.logger.error(f"User embedding retrieval failed: {e}")
            return None

    def _compute_contextual_embedding(
        self,
        user_embedding: Optional[np.ndarray],
        context_features: Dict[str, Any]
    ) -> np.ndarray:
        """
        Fuse user embedding with context using soft attention mechanism.

        Formula: contextual_emb = α * user_emb + (1-α) * context_emb
        where α = 0.7 (user weight), 0.3 (context weight)
        """
        try:
            # Encode context to vector
            context_vector = self._encode_context_vector(context_features)

            # If no user embedding, return context only
            if user_embedding is None or user_embedding.size == 0:
                return context_vector

            # Ensure same dimensions (pad or truncate if needed)
            target_dim = min(user_embedding.size, context_vector.size, self.embedding_dim)

            if target_dim == 0:
                return np.zeros(self.embedding_dim, dtype=float)

            user_vec = user_embedding[:target_dim]
            ctx_vec = context_vector[:target_dim]

            # Soft attention fusion
            alpha = 0.7  # User weight
            fused = alpha * user_vec + (1 - alpha) * ctx_vec

            # Pad fused to embedding_dim
            if fused.size < self.embedding_dim:
                padded = np.zeros(self.embedding_dim, dtype=float)
                padded[:fused.size] = fused
                fused = padded

            return fused

        except Exception as e:
            self.logger.error(f"Contextual embedding fusion failed: {e}")
            return np.zeros(self.embedding_dim, dtype=float)

    def _encode_context_vector(
        self,
        context_features: Dict[str, Any]
    ) -> np.ndarray:
        """
        Encode context features to dense vector.

        Features encoded:
        - time_of_day one-hot (5 dims)
        - day_of_week one-hot (7 dims)
        - is_business_hours (1 dim)
        - device_type one-hot (4 dims: mobile/desktop/tablet/unknown)
        - is_weekend (1 dim)

        Total before padding: 18 dims (padded/truncated to embedding_dim)
        """
        try:
            vector_parts: List[float] = []

            # Time of day (5 categories)
            time_buckets = ["early_morning", "morning", "afternoon", "evening", "night"]
            tod = context_features.get("time_of_day", "morning")
            tod_vec = [1.0 if tod == t else 0.0 for t in time_buckets]
            vector_parts.extend(tod_vec)

            # Day of week (7 days, 0=Monday)
            dow = int(context_features.get("day_of_week", 0) or 0)
            dow_vec = [1.0 if i == dow else 0.0 for i in range(7)]
            vector_parts.extend(dow_vec)

            # Business hours
            vector_parts.append(1.0 if context_features.get("is_business_hours") else 0.0)

            # Device type
            device_types = ["mobile", "desktop", "tablet", "unknown"]
            device = context_features.get("device_type", "unknown")
            device_vec = [1.0 if device == dt else 0.0 for dt in device_types]
            vector_parts.extend(device_vec)

            # Weekend
            vector_parts.append(1.0 if context_features.get("is_weekend") else 0.0)

            # Convert to numpy array
            vector = np.array(vector_parts, dtype=float)

            # Pad or truncate to embedding_dim
            if vector.size < self.embedding_dim:
                padded = np.zeros(self.embedding_dim, dtype=float)
                padded[:vector.size] = vector
                vector = padded
            elif vector.size > self.embedding_dim:
                vector = vector[:self.embedding_dim]

            # L2 normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            return vector

        except Exception as e:
            self.logger.error(f"Context encoding failed: {e}")
            return np.zeros(self.embedding_dim, dtype=float)

    def _compute_contextual_similarity(
        self,
        item: Item,
        contextual_embedding: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between item and contextual embedding.
        """
        try:
            # Get item embedding
            item_embedding = None

            if hasattr(item, 'embedding') and item.embedding is not None:
                item_embedding = np.asarray(item.embedding, dtype=float)
            elif hasattr(item, 'features') and item.features:
                feature_vector = getattr(item.features, 'feature_vector', None)
                if feature_vector is not None:
                    item_embedding = np.asarray(feature_vector, dtype=float)

            if item_embedding is None or item_embedding.size == 0:
                return 0.0

            if contextual_embedding is None or contextual_embedding.size == 0:
                return 0.0

            # Ensure same dimensions
            min_dim = min(item_embedding.size, contextual_embedding.size)
            if min_dim == 0:
                return 0.0

            item_vec = item_embedding[:min_dim]
            ctx_vec = contextual_embedding[:min_dim]

            # Cosine similarity
            dot_product = float(np.dot(item_vec, ctx_vec))
            norm_product = float(np.linalg.norm(item_vec) * np.linalg.norm(ctx_vec))

            if norm_product == 0:
                return 0.0

            similarity = dot_product / norm_product

            # Map cosine [-1,1] -> [0,1]
            return float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Contextual similarity failed: {e}")
            return 0.0

    # =========================================================
    # Scoring Components
    # =========================================================

    def _compute_context_fit(
        self,
        item: Item,
        context_features: Dict[str, Any]
    ) -> float:
        """
        Compute how well item fits current context.
        """
        try:
            fit_score = 0.5  # Base score

            # Time of day fit
            time_of_day = context_features.get('time_of_day')
            category = getattr(item, 'category', None)

            if time_of_day and category:
                # Example business rules (customize based on your data)
                if time_of_day == 'morning' and category == 'paraphrasing':
                    fit_score += 0.25
                elif time_of_day in ['afternoon', 'evening'] and category == 'creative':
                    fit_score += 0.20
                elif time_of_day == 'night' and category == 'grammar':
                    fit_score += 0.15

            # Business hours fit
            if context_features.get('is_business_hours'):
                if getattr(item, 'is_premium', False):
                    fit_score += 0.10

            # Device fit
            device_type = context_features.get('device_type')
            item_type = getattr(item, 'item_type', None)

            if device_type == 'mobile' and item_type == 'quick_tool':
                fit_score += 0.15
            elif device_type == 'desktop' and item_type == 'advanced_tool':
                fit_score += 0.15

            return float(np.clip(fit_score, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Context fit computation failed: {e}")
            return 0.5

    def _compute_user_affinity(
        self,
        user_id: int,
        item: Item
    ) -> float:
        """
        Compute user's historical affinity to this item.
        """
        try:
            # Count interactions
            interaction_count = self.session.query(Interaction).filter(
                Interaction.user_id == user_id,
                Interaction.item_id == item.id
            ).count()

            # Normalize (diminishing returns after 10 interactions)
            affinity = min(interaction_count / 10.0, 1.0) * 0.5 + 0.5

            return float(affinity)

        except Exception as e:
            self.logger.error(f"User affinity computation failed: {e}")
            return 0.5

    def _compute_recency_bonus(self, item: Item) -> float:
        """
        Compute recency/trending bonus.
        """
        try:
            # Check item features
            if hasattr(item, 'features') and item.features:
                trending = getattr(item.features, 'trending_score', 0.0) or 0.0
                freshness = getattr(item.features, 'freshness_score', 0.0) or 0.0

                recency = 0.6 * trending + 0.4 * freshness
                return float(np.clip(recency, 0.0, 1.0))

            # Fallback: use creation date
            if hasattr(item, 'created_at') and item.created_at:
                days_old = (datetime.now() - item.created_at).days
                recency = max(0.0, 1.0 - 0.05 * days_old)
                return float(np.clip(recency, 0.0, 1.0))

            return 0.0

        except Exception as e:
            self.logger.error(f"Recency bonus computation failed: {e}")
            return 0.0

    # =========================================================
    # Diversity Application
    # =========================================================

    def _apply_diversity(
        self,
        ranked_items: List[Tuple[int, float]],
        k: int
    ) -> List[int]:
        """
        Apply diversity strategy (MMR or category-based fallback).
        """
        try:
            if self.enable_mmr and self.mmr_model:
                return self._apply_mmr_diversity(ranked_items, k)
            else:
                return self._apply_category_diversity(ranked_items, k)

        except Exception as e:
            self.logger.error(f"Diversity application failed: {e}", exc_info=True)
            # Fallback: return top-k by score
            return [item_id for item_id, _ in ranked_items[:k]]

    def _apply_mmr_diversity(
        self,
        ranked_items: List[Tuple[int, float]],
        k: int
    ) -> List[int]:
        """
        Apply MMR-based diversity.
        """
        try:
            # Convert to score dict for MMR
            scores = {item_id: score for item_id, score in ranked_items}

            # Apply MMR
            diversified = self.mmr_model.rerank(scores)

            # Return top-k
            return list(diversified.keys())[:k]

        except Exception as e:
            self.logger.warning(f"MMR diversity failed, using fallback: {e}")
            return self._apply_category_diversity(ranked_items, k)

    def _apply_category_diversity(
        self,
        ranked_items: List[Tuple[int, float]],
        k: int
    ) -> List[int]:
        """
        Apply category-based diversity (fallback strategy).
        """
        try:
            selected: List[int] = []
            category_positions: Dict[str, List[int]] = defaultdict(list)

            for item_id, score in ranked_items:
                item = self.session.get(Item, item_id)

                if not item:
                    continue

                category = getattr(item, 'category', 'unknown')

                # Check if category violates gap constraint
                if category in category_positions:
                    last_positions = category_positions[category]
                    current_position = len(selected)

                    if last_positions:
                        gap = current_position - max(last_positions)

                        if gap < self.min_category_gap:
                            # Skip this item (too close to same category)
                            continue

                # Add item
                selected.append(item_id)
                category_positions[category].append(len(selected) - 1)

                # Stop when we have enough
                if len(selected) >= k:
                    break

            # If not enough, fill with remaining items
            if len(selected) < k:
                remaining = [
                    item_id for item_id, _ in ranked_items
                    if item_id not in selected
                ]
                selected.extend(remaining[:k - len(selected)])

            self.logger.debug(
                f"Category diversity: {len(category_positions)} categories "
                f"across {len(selected)} items"
            )

            return selected

        except Exception as e:
            self.logger.error(f"Category diversity failed: {e}", exc_info=True)
            # Final fallback: top-k by score
            return [item_id for item_id, _ in ranked_items[:k]]

    # =========================================================
    # Result Creation
    # =========================================================

    def _create_layout_results(
        self,
        item_ids: List[int],
        user_id: int,
        context_features: Dict[str, Any]
    ) -> List[RecommendationResult]:
        """
        Create RecommendationResult objects with ambient-specific explanations.
        """
        try:
            results: List[RecommendationResult] = []

            for rank, item_id in enumerate(item_ids, 1):
                item = self.session.get(Item, item_id)

                if not item:
                    continue

                # Generate explanation
                reason = self._generate_ambient_reason(
                    user_id,
                    item,
                    context_features
                )

                # Create result
                result = RecommendationResult(
                    item_id=item.id,
                    item_code=getattr(item, "item_code", None),
                    item_name=getattr(item, "item_name", None),
                    score=1.0 - (rank - 1) * 0.1,  # Approximate score
                    rank=rank,
                    reason=reason,
                    metadata={
                        "category": getattr(item, 'category', None),
                        "item_type": getattr(item, 'item_type', None),
                        "is_premium": getattr(item, 'is_premium', False),
                        "layout_optimized": True
                    }
                )

                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Result creation failed: {e}", exc_info=True)
            return []

    def _generate_ambient_reason(
        self,
        user_id: int,
        item: Item,
        context_features: Dict[str, Any]
    ) -> str:

        try:
            reasons: List[str] = []
            lang = self.explanation_language

            # Category-based reason
            category = getattr(item, 'category', None)
            if category:
                if lang == "ko":
                    reasons.append(f"‘{category}’의 감성을 담은 이야기예요.")
                else:
                    reasons.append(f"A story reflecting the mood of '{category}'.")

            # Time-of-day reason
            tod = context_features.get("time_of_day")
            if tod:
                if lang == "ko":
                    tod_map = {
                        "morning": "맑은 아침에 살짝 기운을 북돋아주는 콘텐츠예요 ",
                        "afternoon": "차분한 오후에 가볍게 즐기기 좋은 추천이에요 ",
                        "evening": "편안한 저녁에 마음을 부드럽게 감싸줄 거예요 ",
                        "night": "조용한 밤, 집중이나 휴식에 잘 어울려요 ",
                        "early_morning": "이른 새벽, 고요한 시간과 어울리는 콘텐츠예요 "
                    }
                    reasons.append(tod_map.get(tod, "지금 시간에 어울리는 추천이에요."))
                else:
                    tod_map = {
                        "morning": "Light and uplifting for a clear morning ",
                        "afternoon": "Gentle companion for a calm afternoon ",
                        "evening": "Softly unwinds your evening ",
                        "night": "Quiet night, suited for focus or relaxation ",
                        "early_morning": "Blends into the calm of early dawn "
                    }
                    reasons.append(tod_map.get(tod, "Recommended for this moment."))

            # Device-based reason (mobile 제거 → desktop/mac/tablet)
            device_type = context_features.get("device_type")
            if device_type:
                if lang == "ko":
                    if device_type == "desktop":
                        reasons.append("넓은 화면에서 여유롭게 감상할 수 있어요 ")
                    elif device_type == "mac":
                        reasons.append("Mac 환경에서 섬세하게 즐길 수 있어요 ")
                    elif device_type == "tablet":
                        reasons.append("편안한 터치 환경에 잘 맞아요 ")
                else:
                    if device_type == "desktop":
                        reasons.append("Enjoy comfortably on a wide desktop view ")
                    elif device_type == "mac":
                        reasons.append("Crafted for a detailed Mac experience ")
                    elif device_type == "tablet":
                        reasons.append("Perfect for a relaxed touch on tablet ")

            # Popularity reason
            popularity = getattr(item.features, 'popularity_score', 0.0) if hasattr(item, 'features') and item.features else 0.0
            if popularity > 0.7:
                if lang == "ko":
                    reasons.append("많은 이들이 살짝 들러본 인기 있는 이야기예요 ")
                else:
                    reasons.append("A lightly popular choice among readers ")

            # Personalization / history reason
            try:
                interaction_count = self.session.query(Interaction).filter(
                    Interaction.user_id == user_id,
                    Interaction.item_id == item.id
                ).count()
                if interaction_count > 0:
                    if lang == "ko":
                        reasons.append("이전에 관심을 가졌던 분위기와 닮아 있어요 ")
                    else:
                        reasons.append("Feels familiar — akin to stories you've liked before ")
            except Exception:
                pass  # DB 오류로 인해 실패하더라도 무시

            # Recency reason
            days_old = (datetime.now() - item.created_at).days if hasattr(item, 'created_at') and item.created_at else 999
            if days_old <= 7:
                if lang == "ko":
                    reasons.append("최근에 선보인 새 이야기예요 ")
                else:
                    reasons.append("A freshly introduced story ")

            # Default fallback
            if reasons:
                return " ".join(reasons)
            return "지금 이 순간, 북앤드가 추천드리는 이야기예요 " if lang == "ko" else "A gentle recommendation from Bookend "

        except Exception as e:
            self.logger.error(f"Reason generation failed: {e}", exc_info=True)
            return "지금 이 순간, 북앤드가 추천드리는 이야기예요 " if lang == "ko" else "A gentle recommendation from Bookend "


    # =========================================================
    # Cache Management
    # =========================================================

    def _needs_refresh(self, user_id: int, now: datetime) -> bool:
        """
        Determine whether layout needs refresh.
        """
        try:
            last_refresh = self._cache_timestamps.get(user_id)
            click_count = self._click_counters.get(user_id, 0)

            if last_refresh is None:
                return True
            if now - last_refresh > timedelta(hours=self.refresh_interval_hours):
                return True
            if click_count >= self.activity_refresh_threshold:
                self.logger.debug(f"Refreshing due to activity (click_count={click_count}) for user {user_id}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Cache refresh decision failed: {e}", exc_info=True)
            return True

    def _cache_layout(self, user_id: int, results: List[RecommendationResult], now: datetime) -> None:
        """Cache layout recommendations."""
        try:
            self._layout_cache[user_id] = [r.item_id for r in results]
            self._cache_timestamps[user_id] = now
            self._click_counters[user_id] = 0
            self.logger.debug(f"Cached layout for user {user_id} with {len(results)} items")
        except Exception as e:
            self.logger.error(f"Cache storage failed: {e}", exc_info=True)

    def _get_cached_layout(self, user_id: int, k: int) -> List[RecommendationResult]:
        """Return cached layout results as RecommendationResult list (approximate)."""
        try:
            item_ids = self._layout_cache.get(user_id, [])[:k]
            results: List[RecommendationResult] = []

            for rank, item_id in enumerate(item_ids, 1):
                item = self.session.get(Item, item_id)
                if not item:
                    continue
                reason = self._generate_ambient_reason(user_id, item, {})
                results.append(
                    RecommendationResult(
                        item_id=item.id,
                        item_code=getattr(item, "item_code", None),
                        item_name=getattr(item, "item_name", None),
                        score=1.0 - (rank - 1) * 0.1,
                        rank=rank,
                        reason=reason,
                        metadata={
                            "category": getattr(item, 'category', None),
                            "item_type": getattr(item, 'item_type', None),
                            "is_premium": getattr(item, 'is_premium', False),
                            "layout_optimized": True
                        }
                    )
                )
            return results
        except Exception as e:
            self.logger.error(f"Get cached layout failed: {e}", exc_info=True)
            return []

    def register_click(self, user_id: int) -> None:
        """
        Register user interaction for dynamic refresh trigger.
        """
        try:
            self._click_counters[user_id] = self._click_counters.get(user_id, 0) + 1
            self.logger.debug(f"Registered click for user {user_id}, total={self._click_counters[user_id]}")
        except Exception as e:
            self.logger.error(f"Register click failed: {e}")

    def clear_cache(self, user_id: Optional[int] = None) -> None:
        """
        Clear layout cache for a user or all users.
        """
        try:
            if user_id is not None:
                self._layout_cache.pop(user_id, None)
                self._cache_timestamps.pop(user_id, None)
                self._click_counters.pop(user_id, None)
                self.logger.debug(f"Cleared cache for user {user_id}")
            else:
                self._layout_cache.clear()
                self._cache_timestamps.clear()
                self._click_counters.clear()
                self.logger.debug("Cleared all layout caches")
        except Exception as e:
            self.logger.error(f"Clear cache failed: {e}", exc_info=True)


# =========================================================
# Demo / Smoke test (module run)
# =========================================================
if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres

    setup_logging()
    pg = get_postgres()

    with pg.transaction() as session:
        print("\n" + "=" * 70)
        print("🏠 AMBIENT RECOMMENDER - FULL DEMO")
        print("=" * 70)

        # Example config (could be loaded from YAML)
        config = {
            "layout_positions": 6,
            "refresh_interval_hours": 6,
            "activity_refresh_threshold": 5,
            "diversity_penalty": 0.2,
            "min_category_gap": 2,
            "personalization_strength": 0.7,
            "context_awareness": 0.8,
            "enable_mmr": True,
            "mmr_lambda": 0.5,
            "embedding_dim": 128,
            "explanation_language": "ko",
        }

        recommender = AmbientRecommender(session, config)

        # Fit base models (if required)
        print("\n🔧 Fitting recommender (this may be a no-op if underlying models already fitted)...")
        try:
            recommender.fit(
                weighting="count",
                min_interactions=2,
                lookback_days=90
            )
        except Exception as e:
            print(f"Warning: fit may have failed or not required in demo: {e}")

        # Try to select a test user from hybrid matrix if available; otherwise pick first user in DB
        test_user_id = None
        try:
            # Try to detect from hybrid_recommender internals if present
            mb = getattr(recommender.hybrid_recommender, "user_cf", None)
            if mb and hasattr(mb, "matrix_builder") and getattr(mb.matrix_builder, "n_users", 0) > 0:
                test_user_id = mb.matrix_builder.idx_to_user_id[0]
        except Exception:
            test_user_id = None

        if test_user_id is None:
            try:
                u = session.query(User).first()
                test_user_id = u.id if u else None
            except Exception:
                test_user_id = None

        if test_user_id is None:
            print("⚠️ No user available in DB for demo. Exiting demo.")
        else:
            context = {
                'timestamp': datetime.now(),
                'browser': 'Chrome',
                'os': 'Windows',
                'device_id': 'demo-device-1'
            }

            print(f"\n🎯 Generating ambient layout for user {test_user_id}")
            layout = recommender.recommend_for_layout(
                user_id=test_user_id,
                context=context,
                slot_type=SlotType.HERO_BANNER,
                top_k=6,
                force_refresh=True
            )

            print(f"\n📱 Layout Recommendations ({len(layout)} positions):")
            for rec in layout:
                print(f"\n  Position {rec.rank}: {rec.item_name} (id={rec.item_id})")
                print(f"     Score: {rec.score:.4f}")
                print(f"     Category: {rec.metadata.get('category')}")
                print(f"     Reason: {rec.reason}")

            # Show cache info
            print("\n💾 Cache Info:")
            print(f"  Cached users: {len(recommender._layout_cache)}")
            if test_user_id in recommender._cache_timestamps:
                print(f"  User {test_user_id} cached at: {recommender._cache_timestamps[test_user_id]}")
            print("\n✅ Ambient demo complete.")
