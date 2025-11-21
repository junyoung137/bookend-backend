# src/api/model_loader.py
"""
Trained Model Loader for API

학습된 Hybrid v2 모델을 로드하고 API에서 사용할 수 있도록 래핑합니다.
Singleton 패턴으로 한 번만 로드하여 메모리 효율성을 높입니다.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    학습된 Hybrid v2 Recommender 모델 로더
    
    Features:
    - Singleton pattern (앱 당 한 번만 로드)
    - Lazy loading (실제 요청 시 로드)
    - Error handling with fallback
    """
    
    def __init__(self, model_path: str = "data/models/hybrid_v2_rebalanced.pkl"):
        """
        Initialize model loader
        
        Args:
            model_path: Path to pickled model file
        """
        self.model_path = Path(model_path)
        self._model = None
        self._is_loaded = False
        
        logger.info(f"ModelLoader initialized with path: {self.model_path}")
    
    def load(self):
        """
        Load model from pickle file
        
        Returns:
            Loaded model instance
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if self._is_loaded and self._model is not None:
            logger.debug("Model already loaded, returning cached instance")
            return self._model
        
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Please train the model first using: "
                    f"python src/models/hybrid/hybrid_v2_rebalanced.py"
                )
            
            logger.info(f"Loading model from {self.model_path}...")
            
            with open(self.model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            self._is_loaded = True
            
            # Model validation
            if not hasattr(self._model, 'recommend'):
                raise ValueError("Loaded model doesn't have 'recommend' method")
            
            logger.info("✅ Model loaded successfully!")
            logger.info(f"   Users: {len(self._model.user_ids) if hasattr(self._model, 'user_ids') else 'N/A'}")
            logger.info(f"   Items: {len(self._model.item_ids) if hasattr(self._model, 'item_ids') else 'N/A'}")
            
            return self._model
        
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User database ID
            k: Number of recommendations
            exclude_interacted: Exclude already interacted items
        
        Returns:
            List of recommendation dicts with keys:
            - item_id: int
            - score: float
            - reasons: List[str]
        
        Raises:
            RuntimeError: If model not loaded
        """
        if not self._is_loaded or self._model is None:
            logger.warning("Model not loaded, attempting to load...")
            self.load()
        
        try:
            recommendations = self._model.recommend(
                user_id=user_id,
                k=k,
                exclude_interacted=exclude_interacted
            )
            
            logger.debug(
                f"Generated {len(recommendations)} recommendations "
                f"for user {user_id}"
            )
            
            return recommendations
        
        except Exception as e:
            logger.error(
                f"Recommendation failed for user {user_id}: {e}",
                exc_info=True
            )
            raise
    
    def is_user_known(self, user_id: int) -> bool:
        """
        Check if user is in training data
        
        Args:
            user_id: User database ID
        
        Returns:
            True if user is known, False otherwise
        """
        if not self._is_loaded or self._model is None:
            self.load()
        
        if hasattr(self._model, 'user_ids'):
            return user_id in self._model.user_ids
        
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and statistics
        
        Returns:
            Dictionary with model info
        """
        if not self._is_loaded or self._model is None:
            self.load()
        
        info = {
            "model_type": "Hybrid v2 Rebalanced",
            "model_path": str(self.model_path),
            "is_loaded": self._is_loaded,
        }
        
        if hasattr(self._model, 'user_ids'):
            info["num_users"] = len(self._model.user_ids)
        
        if hasattr(self._model, 'item_ids'):
            info["num_items"] = len(self._model.item_ids)
        
        if hasattr(self._model, 'popularity_weight'):
            info["weights"] = {
                "popularity": self._model.popularity_weight,
                "user_cf": self._model.user_cf_weight,
                "item_cf": self._model.item_cf_weight,
                "diversity": self._model.diversity_weight,
            }
        
        return info
    
    def reload(self):
        """
        Force reload model from disk
        
        Useful for:
        - Model updates
        - Error recovery
        """
        logger.info("Force reloading model...")
        self._model = None
        self._is_loaded = False
        return self.load()


# =========================================================
# Singleton Instance
# =========================================================

_model_loader_instance: Optional[ModelLoader] = None


@lru_cache()
def get_model_loader(
    model_path: str = "data/models/hybrid_v2_rebalanced.pkl"
) -> ModelLoader:
    """
    Get singleton ModelLoader instance
    
    Args:
        model_path: Path to model file
    
    Returns:
        ModelLoader instance
    
    Usage:
        >>> loader = get_model_loader()
        >>> recommendations = loader.recommend(user_id=123, k=10)
    """
    global _model_loader_instance
    
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoader(model_path=model_path)
    
    return _model_loader_instance


# =========================================================
# Testing
# =========================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("🧪 Testing Model Loader")
    print("="*70)
    
    # Test 1: Load model
    print("\n1️⃣ Loading model...")
    loader = get_model_loader()
    
    try:
        loader.load()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        exit(1)
    
    # Test 2: Get model info
    print("\n2️⃣ Model information:")
    info = loader.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test 3: Test recommendation (user_id=1 예시)
    print("\n3️⃣ Testing recommendation for user_id=1...")
    try:
        recs = loader.recommend(user_id=1, k=5)
        print(f"✅ Generated {len(recs)} recommendations")
        for i, rec in enumerate(recs[:3], 1):
            print(f"   {i}. Item {rec['item_id']} (score: {rec['score']:.4f})")
    except Exception as e:
        print(f"⚠️  Recommendation failed: {e}")
    
    print("\n" + "="*70)
    print("✅ Model Loader Test Complete")
    print("="*70 + "\n")