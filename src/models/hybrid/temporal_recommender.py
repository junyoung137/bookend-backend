"""
Temporal Flow Recommender for time-pattern based recommendations.

Integrates:
1. User temporal patterns (hour/day preferences)
2. Item temporal patterns (peak usage times)
3. Current time context (real-time matching)
4. Recency decay (time-since-last-interaction)
5. Temporal trend analysis (usage trends over time)

Principles:
- Single Responsibility: Focus on temporal aspects only
- Error Handling: Graceful degradation when patterns unavailable
- Performance: Cached pattern analysis with generator queries
- Explainability: Time-aware recommendation reasons
"""

from typing import Dict, Any, Optional, List, Generator
import logging
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from src.models.base_recommender import BaseRecommender
from src.models.hybrid.hybrid_recommender import HybridRecommender
from src.models.hybrid.temporal_flow_analyzer import TemporalFlowAnalyzer
from src.features.context_features import ContextFeatureExtractor
from src.database.models import Item, Interaction

logger = logging.getLogger(__name__)


class TemporalRecommender(BaseRecommender):
    """
    Time-aware recommender using temporal patterns.
    
    Configuration:
        - temporal_weight: Weight for temporal matching (default: 0.4)
        - recency_weight: Weight for recency decay (default: 0.3)
        - base_weight: Weight for base hybrid scores (default: 0.3)
        - lookback_days: Days to analyze patterns (default: 90)
        - recency_decay_hours: Half-life for recency decay (default: 24)
        - peak_hour_boost: Boost multiplier for peak hours (default: 1.3)
        - enable_trend_analysis: Enable temporal trend detection (default: True)
        - explanation_language: "ko" or "en" (default: "ko")
        - max_items_to_score: Maximum items to score for performance (default: 1000)
    """

    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        super().__init__(session, config)

        # Weights
        self.temporal_weight = float(self.get_config("temporal_weight", 0.4))
        self.recency_weight = float(self.get_config("recency_weight", 0.3))
        self.base_weight = float(self.get_config("base_weight", 0.3))

        # Validate and normalize weights
        self._validate_and_normalize_weights()

        # Temporal settings
        self.lookback_days = int(self.get_config("lookback_days", 90))
        self.recency_decay_hours = float(self.get_config("recency_decay_hours", 24))
        self.peak_hour_boost = float(self.get_config("peak_hour_boost", 1.3))
        self.enable_trend_analysis = bool(self.get_config("enable_trend_analysis", True))
        
        # Performance settings
        self.max_items_to_score = int(self.get_config("max_items_to_score", 1000))

        # Explanation language
        self.explanation_language = self.get_config("explanation_language", "ko")

        # Components (lazy initialization for better performance)
        self.base_recommender = HybridRecommender(session, config)
        self.temporal_analyzer = TemporalFlowAnalyzer(session, lookback_days=self.lookback_days)
        self.context_extractor = ContextFeatureExtractor()
        
        # Cache for time-of-day reason mappings (avoid repeated dict creation)
        self._tod_reasons_cache = self._build_tod_reasons_cache()

    def fit(self, **kwargs) -> None:
        """Train temporal recommender and validate configuration."""
        try:
            self.logger.info("Fitting Temporal Recommender")
            
            # Validate configuration before fitting
            self._validate_configuration()
            
            # Fit base recommender
            self.base_recommender.fit(**kwargs)
            
            self.is_fitted = True
            self.logger.info(
                f"Temporal Recommender fitted successfully "
                f"(weights: temporal={self.temporal_weight:.2f}, "
                f"recency={self.recency_weight:.2f}, base={self.base_weight:.2f})"
            )
        except Exception as e:
            self.logger.error(f"Failed to fit Temporal Recommender: {e}", exc_info=True)
            raise

    def get_model_name(self) -> str:
        return "temporal_recommender"

    # =========================================================
    # Core Scoring Logic
    # =========================================================

    def _compute_scores(
        self, 
        user_id: int, 
        context: Optional[Dict[str, Any]] = None,
        candidate_items: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute temporal recommendation scores with performance optimizations.
        
        Args:
            user_id: User database ID
            context: Contextual features (must include timestamp)
            candidate_items: List of candidate item IDs
        
        Returns:
            Dictionary mapping item_id to temporal score
        """
        if not self.is_fitted:
            self.logger.error("Model not fitted yet")
            return {}

        try:
            temporal_scores: Dict[int, float] = {}
            
            # Extract context once (avoid repeated calls)
            current_context = self._extract_current_context(context)
            
            # Get user pattern once (cached by analyzer)
            user_pattern = self.temporal_analyzer.get_user_temporal_pattern(user_id)

            # 1. Base scores (if weight > 0)
            if self.base_weight > 0:
                base_scores = self._get_base_scores(user_id, context, candidate_items)
                self._merge_scores(temporal_scores, base_scores, self.base_weight)

            # 2. Temporal match scores (if weight > 0)
            if self.temporal_weight > 0:
                temporal_match_scores = self._compute_temporal_match_scores(
                    user_id, user_pattern, current_context, candidate_items
                )
                self._merge_scores(temporal_scores, temporal_match_scores, self.temporal_weight)

            # 3. Recency decay scores (if weight > 0)
            if self.recency_weight > 0:
                recency_scores = self._compute_recency_scores(user_id, candidate_items)
                self._merge_scores(temporal_scores, recency_scores, self.recency_weight)

            # 4. Peak hour boost (optional)
            if self.peak_hour_boost > 1.0 and temporal_scores:
                temporal_scores = self._apply_peak_hour_boost(
                    temporal_scores, user_pattern, current_context
                )

            return temporal_scores

        except Exception as e:
            self.logger.error(f"Failed to compute temporal scores: {e}", exc_info=True)
            return {}

    # =========================================================
    # Component Score Computation
    # =========================================================

    def _get_base_scores(
        self, 
        user_id: int, 
        context: Optional[Dict[str, Any]],
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """Get and normalize base hybrid recommendation scores."""
        try:
            scores = self.base_recommender._compute_scores(user_id, context, candidate_items)
            normalized = self._normalize_scores(scores)
            self.logger.debug(f"Base scores: {len(normalized)} items")
            return normalized
        except Exception as e:
            self.logger.error(f"Base scoring failed: {e}")
            return {}

    def _compute_temporal_match_scores(
        self, 
        user_id: int, 
        user_pattern: Dict[str, Any],
        current_context: Dict[str, Any],
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """
        Compute temporal matching scores with generator for memory efficiency.
        
        Args:
            user_id: User database ID
            user_pattern: User's temporal pattern
            current_context: Current time context
            candidate_items: List of candidate item IDs
        
        Returns:
            Dictionary mapping item_id to temporal match score
        """
        try:
            # Use generator for memory efficiency
            if candidate_items:
                items_to_score = iter(candidate_items)
            else:
                # Limit query with max_items_to_score for performance
                items_query = self.session.query(Item.id).filter(
                    Item.is_active == True
                ).limit(self.max_items_to_score)
                items_to_score = (item_id for (item_id,) in items_query)

            match_scores: Dict[int, float] = {}
            
            # Process items in batches for better cache utilization
            for item_id in items_to_score:
                item_pattern = self.temporal_analyzer.get_item_temporal_pattern(item_id)
                affinity = self.temporal_analyzer.compute_temporal_affinity(
                    user_pattern, item_pattern, current_context
                )
                match_scores[item_id] = float(affinity)

            normalized = self._normalize_scores(match_scores)
            self.logger.debug(f"Temporal match scores: {len(normalized)} items")
            return normalized

        except Exception as e:
            self.logger.error(f"Temporal match scoring failed: {e}", exc_info=True)
            return {}

    def _compute_recency_scores(
        self, 
        user_id: int, 
        candidate_items: Optional[List[int]]
    ) -> Dict[int, float]:
        """
        Compute recency-based scores with exponential decay.
        
        Uses single query with optimized filtering for performance.
        
        Args:
            user_id: User database ID
            candidate_items: List of candidate item IDs
        
        Returns:
            Dictionary mapping item_id to recency score
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            
            # Build query with optional item filtering
            query = self.session.query(Interaction).filter(
                Interaction.user_id == user_id,
                Interaction.event_time >= cutoff_date
            )
            
            # Add item filter if candidates provided
            if candidate_items:
                query = query.filter(Interaction.item_id.in_(candidate_items))
            
            # Execute query
            interactions = query.order_by(Interaction.event_time.desc()).all()

            if not interactions:
                self.logger.debug(f"No recent interactions for user {user_id}")
                return {}

            # Compute decay scores
            recency_scores: Dict[int, float] = {}
            now = datetime.now()

            for interaction in interactions:
                item_id = interaction.item_id
                hours_ago = (now - interaction.event_time).total_seconds() / 3600
                decay_score = np.exp(-hours_ago / self.recency_decay_hours)
                
                # Keep highest score per item (most recent interaction)
                recency_scores[item_id] = max(
                    recency_scores.get(item_id, 0.0), 
                    float(decay_score)
                )

            normalized = self._normalize_scores(recency_scores)
            self.logger.debug(f"Recency scores: {len(normalized)} items")
            return normalized
            
        except Exception as e:
            self.logger.error(f"Recency scoring failed: {e}", exc_info=True)
            return {}

    # =========================================================
    # Peak Hour Boost
    # =========================================================

    def _apply_peak_hour_boost(
        self, 
        scores: Dict[int, float], 
        user_pattern: Dict[str, Any],
        current_context: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Boost scores for items matching user's peak activity hours.
        
        Args:
            scores: Current scores
            user_pattern: User's temporal pattern
            current_context: Current time context
        
        Returns:
            Boosted scores (or original if no boost applies)
        """
        try:
            current_hour = current_context.get('hour')
            peak_hour = user_pattern.get('peak_hour')

            # Early return if boost not applicable
            if current_hour is None or peak_hour is None:
                return scores

            # Calculate circular hour distance
            hour_distance = min(
                abs(current_hour - peak_hour), 
                24 - abs(current_hour - peak_hour)
            )
            
            # Apply boost only within 2-hour window
            if hour_distance > 2:
                return scores
            
            # Calculate boost factor with distance decay
            boost_factor = max(
                1.0,  # Minimum boost is 1.0 (no change)
                self.peak_hour_boost * (1.0 - hour_distance / 3.0)
            )
            
            boosted_scores = {
                item_id: score * boost_factor 
                for item_id, score in scores.items()
            }
            
            self.logger.debug(
                f"Applied peak hour boost: factor={boost_factor:.2f}, "
                f"current={current_hour}, peak={peak_hour}"
            )
            
            return boosted_scores

        except Exception as e:
            self.logger.error(f"Peak hour boost failed: {e}", exc_info=True)
            return scores

    # =========================================================
    # Context Extraction
    # =========================================================

    def _extract_current_context(
        self, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract and enrich current time context with safe defaults.
        
        Args:
            context: Input context (may contain timestamp)
        
        Returns:
            Enriched context with temporal features
        """
        try:
            # Use provided timestamp or default to now
            timestamp = (
                context.get('timestamp') 
                if context and 'timestamp' in context 
                else datetime.now()
            )
            
            # Extract temporal context
            temporal_context = self.context_extractor.extract_temporal_context(timestamp)
            
            return temporal_context or {}
            
        except Exception as e:
            self.logger.error(f"Context extraction failed: {e}", exc_info=True)
            # Return safe default context
            return {
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'time_of_day': 'afternoon',
            }

    # =========================================================
    # Scoring Utilities
    # =========================================================

    def _merge_scores(
        self, 
        target_scores: Dict[int, float], 
        source_scores: Dict[int, float], 
        weight: float
    ) -> None:
        """
        Merge source scores into target with weight (in-place operation).
        
        Args:
            target_scores: Target dictionary to update
            source_scores: Source scores to merge
            weight: Weight to apply to source scores
        """
        for item_id, score in source_scores.items():
            target_scores[item_id] = target_scores.get(item_id, 0.0) + weight * score

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Normalize scores to [0, 1] range with safe defaults.
        
        Args:
            scores: Raw scores
        
        Returns:
            Normalized scores in [0, 1]
        """
        if not scores:
            return {}
        
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        
        # Handle constant scores
        if max_val == min_val:
            return {item_id: 1.0 for item_id in scores}
        
        # Min-max normalization
        return {
            item_id: (score - min_val) / (max_val - min_val) 
            for item_id, score in scores.items()
        }

    def _validate_and_normalize_weights(self) -> None:
        """Validate and normalize weights to sum to 1.0."""
        total = self.temporal_weight + self.recency_weight + self.base_weight
        
        if total <= 0:
            raise ValueError(
                f"Invalid weights: all weights are zero or negative "
                f"(temporal={self.temporal_weight}, recency={self.recency_weight}, "
                f"base={self.base_weight})"
            )
        
        if not np.isclose(total, 1.0):
            self.logger.warning(f"Weights sum to {total:.3f}, normalizing to 1.0")
            self.temporal_weight /= total
            self.recency_weight /= total
            self.base_weight /= total

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        if self.lookback_days <= 0:
            raise ValueError(f"lookback_days must be positive, got {self.lookback_days}")
        
        if self.recency_decay_hours <= 0:
            raise ValueError(f"recency_decay_hours must be positive, got {self.recency_decay_hours}")
        
        if self.peak_hour_boost < 1.0:
            raise ValueError(f"peak_hour_boost must be >= 1.0, got {self.peak_hour_boost}")

    # =========================================================
    # Explainability
    # =========================================================

    def _build_tod_reasons_cache(self) -> Dict[str, Dict[str, str]]:
        """Build cached time-of-day reason mappings for both languages."""
        return {
            "ko": {
                "morning": "상쾌한 아침에 어울리는 선택",
                "afternoon": "여유로운 오후 시간에 딱",
                "evening": "편안한 저녁 시간에 좋아요",
                "night": "조용한 밤에 집중하기 좋아요",
                "early_morning": "이른 새벽, 고요한 시간과 어울려요"
            },
            "en": {
                "morning": "Perfect for a fresh morning",
                "afternoon": "Great for a relaxed afternoon",
                "evening": "Ideal for a calm evening",
                "night": "Suited for a quiet night",
                "early_morning": "Fits the stillness of early dawn"
            }
        }

    def _generate_reason(
        self, 
        item: Any, 
        score: float, 
        user_id: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate time-aware explanation for recommendation.
        
        Args:
            item: Item object
            score: Recommendation score
            user_id: User database ID
            context: Contextual features
        
        Returns:
            Localized explanation string
        """
        try:
            reasons: List[str] = []
            lang = self.explanation_language
            
            # Extract current context
            current_context = self._extract_current_context(context)
            current_hour = current_context.get('hour')
            time_of_day = current_context.get('time_of_day')

            # Get user pattern (cached)
            user_pattern = self.temporal_analyzer.get_user_temporal_pattern(user_id)
            peak_hour = user_pattern.get('peak_hour')

            # 1. Peak hour match reason
            if current_hour is not None and peak_hour is not None:
                hour_distance = min(
                    abs(current_hour - peak_hour), 
                    24 - abs(current_hour - peak_hour)
                )
                
                if hour_distance <= 1:
                    reasons.append(
                        "지금이 가장 활동적인 시간" if lang == "ko" 
                        else "This is your most active time"
                    )

            # 2. Time-of-day match reason (using cached mappings)
            if time_of_day:
                tod_map = self._tod_reasons_cache.get(lang, self._tod_reasons_cache["en"])
                reasons.append(
                    tod_map.get(time_of_day, "지금 이 순간에 잘 맞아요" if lang == "ko" 
                                else "Well-suited for this moment")
                )

            # 3. Recency reason (if applicable)
            try:
                recent_interaction = self.session.query(Interaction).filter(
                    Interaction.user_id == user_id,
                    Interaction.item_id == item.id
                ).order_by(Interaction.event_time.desc()).first()
                
                if recent_interaction:
                    hours_ago = (
                        datetime.now() - recent_interaction.event_time
                    ).total_seconds() / 3600
                    
                    if hours_ago < 24:
                        reasons.append(
                            "최근 관심 보인 항목" if lang == "ko" 
                            else "Recently caught your interest"
                        )
            except Exception:
                pass  # Silently skip recency reason if query fails

            # Default fallback reason
            if not reasons:
                reasons.append(
                    "지금 이 시간에 추천" if lang == "ko" 
                    else "Recommended for this time"
                )

            return " • ".join(reasons)  # Use bullet separator for multiple reasons

        except Exception as e:
            self.logger.error(f"Reason generation failed: {e}", exc_info=True)
            return "추천" if self.explanation_language == "ko" else "Recommended"


# =========================================================
# Demo / Smoke Test
# =========================================================
if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres
    from src.database.models import User
    
    setup_logging()
    
    pg = get_postgres()
    
    with pg.transaction() as session:
        print("\n" + "="*70)
        print("⏰ TEMPORAL RECOMMENDER DEMO")
        print("="*70)
        
        config = {
            "temporal_weight": 0.4,
            "recency_weight": 0.3,
            "base_weight": 0.3,
            "lookback_days": 90,
            "recency_decay_hours": 24,
            "peak_hour_boost": 1.3,
            "enable_trend_analysis": True,
            "explanation_language": "ko",
            "max_items_to_score": 1000
        }
        
        recommender = TemporalRecommender(session, config)
        
        print("\n🔧 Fitting recommender...")
        try:
            recommender.fit(
                weighting="count",
                min_interactions=2,
                lookback_days=90
            )
        except Exception as e:
            print(f"⚠️  Fit warnings: {e}")
        
        # Get test user
        user = session.query(User).join(Interaction).first()
        
        if user:
            print(f"\n👤 Test user: {user.distinct_id}")
            
            # Create context with current time
            context = {
                'timestamp': datetime.now(),
                'browser': 'Chrome',
                'os': 'Windows'
            }
            
            print(f"\n🎯 Generating temporal recommendations...")
            recommendations = recommender.recommend(
                user_id=user.id,
                context=context,
                limit=5,
                min_score=0.0
            )
            
            print(f"\n📋 Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                print(f"\n  {rec.rank}. {rec.item_name}")
                print(f"     Score: {rec.score:.4f}")
                print(f"     Reason: {rec.reason}")
                print(f"     Category: {rec.metadata.get('category')}")
        
        else:
            print("\n⚠️  No users with interactions found for demo")
        
        print("\n" + "="*70)
        print("✅ Temporal Recommender Demo Complete")
        print("="*70)