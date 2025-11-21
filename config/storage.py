"""
Unified Storage Interface for Bookend Recommendation System.

Provides high-level abstraction over MinIO and Redis:
1. Model artifact management (save/load with versioning)
2. Feature store operations (with Redis caching)
3. Interim data storage (pipeline artifacts)
4. Automatic caching strategy
5. Health monitoring

Principles:
- Single Interface: One API for all storage operations
- Smart Caching: Automatic Redis cache for frequent access
- Versioning: Built-in model version management
- Error Recovery: Graceful degradation on storage failures
"""

from typing import Optional, Any, Dict, List, Tuple
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import pickle
import json

from config.settings import get_settings
from config.minio_config import get_minio
from config.redis_config import get_redis

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Unified storage manager for models, features, and interim data.
    
    Coordinates between MinIO (persistent) and Redis (cache) storage.
    Provides intelligent caching strategies and version management.
    """
    
    def __init__(self):
        """Initialize storage manager with MinIO and Redis connections."""
        self.settings = get_settings()
        self.minio = get_minio()
        self.redis = get_redis()
        
        logger.info("Storage manager initialized")
    
    # =========================================================
    # Model Artifact Management
    # =========================================================
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ) -> bool:
        """
        Save model artifact to MinIO with optional caching.
        
        Args:
            model: Model object (will be pickled)
            model_name: Model identifier (e.g., "ambient_recommender")
            version: Version string (e.g., "v1.0.0" or timestamp)
            metadata: Optional metadata dictionary
            cache: Whether to cache in Redis
        
        Returns:
            bool: True if successful
        
        Example:
            >>> storage = get_storage()
            >>> storage.save_model(
            ...     model_object,
            ...     "ambient_recommender",
            ...     "v1.0.0",
            ...     metadata={"accuracy": 0.85, "trained_at": "2025-01-15"}
            ... )
        """
        try:
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Generate object key
            object_key = self._get_model_key(model_name, version)
            
            # Prepare metadata
            full_metadata = {
                "model_name": model_name,
                "version": version,
                "saved_at": datetime.utcnow().isoformat(),
                "size_bytes": len(model_bytes),
                "checksum": hashlib.md5(model_bytes).hexdigest()
            }
            
            if metadata:
                full_metadata.update(metadata)
            
            # Save to MinIO
            success = self.minio.upload_data(
                bucket_name=self.settings.minio.models_bucket,
                object_name=object_key,
                data=model_bytes,
                content_type="application/octet-stream",
                metadata={k: str(v) for k, v in full_metadata.items()}
            )
            
            if not success:
                logger.error(f"Failed to save model {model_name} v{version} to MinIO")
                return False
            
            # Save metadata separately as JSON
            metadata_key = self._get_model_metadata_key(model_name, version)
            metadata_json = json.dumps(full_metadata).encode('utf-8')
            
            self.minio.upload_data(
                bucket_name=self.settings.minio.models_bucket,
                object_name=metadata_key,
                data=metadata_json,
                content_type="application/json"
            )
            
            # Cache in Redis if requested
            if cache:
                cache_key = f"model:{model_name}:{version}"
                self.redis.set(
                    key=cache_key,
                    value=model,
                    ttl=86400,  # 24 hours
                    serialize="pickle"
                )
                logger.debug(f"Cached model in Redis: {cache_key}")
            
            logger.info(
                f"Saved model {model_name} v{version} "
                f"({len(model_bytes)} bytes)"
            )
            return True
        
        except Exception as e:
            logger.error(
                f"Failed to save model {model_name} v{version}: {e}",
                exc_info=True
            )
            return False
    
    def load_model(
        self,
        model_name: str,
        version: str = "latest",
        use_cache: bool = True
    ) -> Optional[Any]:
        """
        Load model artifact from cache or MinIO.
        
        Args:
            model_name: Model identifier
            version: Version string or "latest"
            use_cache: Whether to check Redis cache first
        
        Returns:
            Model object, or None if not found
        
        Example:
            >>> storage = get_storage()
            >>> model = storage.load_model("ambient_recommender", "v1.0.0")
            >>> if model:
            ...     predictions = model.predict(data)
        """
        try:
            # Resolve version if "latest"
            if version == "latest":
                version = self._get_latest_version(model_name)
                if not version:
                    logger.warning(f"No versions found for model {model_name}")
                    return None
            
            # Try Redis cache first
            if use_cache:
                cache_key = f"model:{model_name}:{version}"
                cached_model = self.redis.get(
                    key=cache_key,
                    serialize="pickle"
                )
                
                if cached_model is not None:
                    logger.debug(f"Loaded model from cache: {cache_key}")
                    return cached_model
            
            # Load from MinIO
            object_key = self._get_model_key(model_name, version)
            
            model_bytes = self.minio.download_data(
                bucket_name=self.settings.minio.models_bucket,
                object_name=object_key
            )
            
            if model_bytes is None:
                logger.warning(
                    f"Model {model_name} v{version} not found in MinIO"
                )
                return None
            
            # Deserialize
            model = pickle.loads(model_bytes)
            
            # Cache for future use
            if use_cache:
                cache_key = f"model:{model_name}:{version}"
                self.redis.set(
                    key=cache_key,
                    value=model,
                    ttl=86400,
                    serialize="pickle"
                )
            
            logger.info(f"Loaded model {model_name} v{version} from MinIO")
            return model
        
        except Exception as e:
            logger.error(
                f"Failed to load model {model_name} v{version}: {e}",
                exc_info=True
            )
            return None
    
    def get_model_metadata(
        self,
        model_name: str,
        version: str = "latest"
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a model version.
        
        Args:
            model_name: Model identifier
            version: Version string or "latest"
        
        Returns:
            Metadata dictionary, or None if not found
        
        Example:
            >>> storage = get_storage()
            >>> metadata = storage.get_model_metadata("ambient_recommender", "v1.0.0")
            >>> print(f"Trained at: {metadata['trained_at']}")
        """
        try:
            # Resolve version if "latest"
            if version == "latest":
                version = self._get_latest_version(model_name)
                if not version:
                    return None
            
            # Try cache first
            cache_key = f"model_metadata:{model_name}:{version}"
            cached_metadata = self.redis.get(cache_key)
            
            if cached_metadata:
                return cached_metadata
            
            # Load from MinIO
            metadata_key = self._get_model_metadata_key(model_name, version)
            
            metadata_bytes = self.minio.download_data(
                bucket_name=self.settings.minio.models_bucket,
                object_name=metadata_key
            )
            
            if metadata_bytes is None:
                logger.warning(f"Metadata not found for {model_name} v{version}")
                return None
            
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Cache metadata
            self.redis.set(cache_key, metadata, ttl=3600)
            
            return metadata
        
        except Exception as e:
            logger.error(
                f"Failed to get metadata for {model_name} v{version}: {e}",
                exc_info=True
            )
            return None
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: Model identifier
        
        Returns:
            List of version strings (sorted, newest first)
        
        Example:
            >>> storage = get_storage()
            >>> versions = storage.list_model_versions("ambient_recommender")
            >>> print(f"Available versions: {versions}")
        """
        try:
            prefix = f"models/{model_name}/"
            
            objects = self.minio.list_objects(
                bucket_name=self.settings.minio.models_bucket,
                prefix=prefix,
                recursive=False
            )
            
            # Extract versions from object names
            versions = []
            for obj_name in objects:
                if obj_name.endswith(".pkl"):
                    # Extract version from path: models/NAME/VERSION/model.pkl
                    parts = obj_name.split('/')
                    if len(parts) >= 3:
                        version = parts[2]
                        versions.append(version)
            
            # Sort versions (newest first, assuming semantic versioning or timestamps)
            versions = sorted(set(versions), reverse=True)
            
            logger.debug(f"Found {len(versions)} versions for {model_name}")
            return versions
        
        except Exception as e:
            logger.error(
                f"Failed to list versions for {model_name}: {e}",
                exc_info=True
            )
            return []
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """
        Delete a model version from storage.
        
        Args:
            model_name: Model identifier
            version: Version to delete
        
        Returns:
            bool: True if successful
        
        Example:
            >>> storage = get_storage()
            >>> storage.delete_model("ambient_recommender", "v0.9.0")
        """
        try:
            # Delete from MinIO
            object_key = self._get_model_key(model_name, version)
            metadata_key = self._get_model_metadata_key(model_name, version)
            
            success = self.minio.delete_object(
                bucket_name=self.settings.minio.models_bucket,
                object_name=object_key
            )
            
            self.minio.delete_object(
                bucket_name=self.settings.minio.models_bucket,
                object_name=metadata_key
            )
            
            # Delete from cache
            cache_key = f"model:{model_name}:{version}"
            self.redis.delete(cache_key)
            
            cache_metadata_key = f"model_metadata:{model_name}:{version}"
            self.redis.delete(cache_metadata_key)
            
            if success:
                logger.info(f"Deleted model {model_name} v{version}")
            
            return success
        
        except Exception as e:
            logger.error(
                f"Failed to delete model {model_name} v{version}: {e}",
                exc_info=True
            )
            return False
    
    # =========================================================
    # Feature Store Operations
    # =========================================================
    
    def save_features(
        self,
        feature_name: str,
        data: Any,
        format: str = "parquet",
        cache: bool = True,
        ttl: int = 3600
    ) -> bool:
        """
        Save feature data to MinIO with optional caching.
        
        Args:
            feature_name: Feature identifier (e.g., "user_features_20250115")
            data: Data to save (DataFrame, dict, etc.)
            format: Storage format ("parquet", "json", "pickle")
            cache: Whether to cache in Redis
            ttl: Cache TTL in seconds
        
        Returns:
            bool: True if successful
        
        Example:
            >>> storage = get_storage()
            >>> storage.save_features(
            ...     "user_features_20250115",
            ...     user_df,
            ...     format="parquet"
            ... )
        """
        try:
            # Serialize based on format
            if format == "parquet":
                # Assume pandas DataFrame
                import pandas as pd
                from io import BytesIO
                
                buffer = BytesIO()
                data.to_parquet(buffer, index=False)
                data_bytes = buffer.getvalue()
                content_type = "application/octet-stream"
            
            elif format == "json":
                if hasattr(data, 'to_json'):
                    # DataFrame
                    data_str = data.to_json(orient='records')
                else:
                    data_str = json.dumps(data)
                
                data_bytes = data_str.encode('utf-8')
                content_type = "application/json"
            
            elif format == "pickle":
                data_bytes = pickle.dumps(data)
                content_type = "application/octet-stream"
            
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            # Save to MinIO
            object_key = f"features/{feature_name}.{format}"
            
            success = self.minio.upload_data(
                bucket_name=self.settings.minio.features_bucket,
                object_name=object_key,
                data=data_bytes,
                content_type=content_type
            )
            
            if not success:
                return False
            
            # Cache if requested and data is small enough
            if cache and len(data_bytes) < 1024 * 1024:  # < 1MB
                cache_key = f"features:{feature_name}"
                self.redis.set(
                    key=cache_key,
                    value=data,
                    ttl=ttl,
                    serialize="pickle"
                )
            
            logger.info(f"Saved features {feature_name} ({len(data_bytes)} bytes)")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save features {feature_name}: {e}", exc_info=True)
            return False
    
    def load_features(
        self,
        feature_name: str,
        format: str = "parquet",
        use_cache: bool = True
    ) -> Optional[Any]:
        """
        Load feature data from cache or MinIO.
        
        Args:
            feature_name: Feature identifier
            format: Storage format
            use_cache: Whether to check Redis cache first
        
        Returns:
            Feature data, or None if not found
        
        Example:
            >>> storage = get_storage()
            >>> user_features = storage.load_features("user_features_20250115")
        """
        try:
            # Try cache first
            if use_cache:
                cache_key = f"features:{feature_name}"
                cached_data = self.redis.get(cache_key, serialize="pickle")
                
                if cached_data is not None:
                    logger.debug(f"Loaded features from cache: {feature_name}")
                    return cached_data
            
            # Load from MinIO
            object_key = f"features/{feature_name}.{format}"
            
            data_bytes = self.minio.download_data(
                bucket_name=self.settings.minio.features_bucket,
                object_name=object_key
            )
            
            if data_bytes is None:
                logger.warning(f"Features not found: {feature_name}")
                return None
            
            # Deserialize based on format
            if format == "parquet":
                import pandas as pd
                from io import BytesIO
                
                buffer = BytesIO(data_bytes)
                data = pd.read_parquet(buffer)
            
            elif format == "json":
                data = json.loads(data_bytes.decode('utf-8'))
            
            elif format == "pickle":
                data = pickle.loads(data_bytes)
            
            else:
                logger.error(f"Unsupported format: {format}")
                return None
            
            # Cache for future use if small enough
            if use_cache and len(data_bytes) < 1024 * 1024:
                cache_key = f"features:{feature_name}"
                self.redis.set(cache_key, data, ttl=3600, serialize="pickle")
            
            logger.info(f"Loaded features {feature_name} from MinIO")
            return data
        
        except Exception as e:
            logger.error(f"Failed to load features {feature_name}: {e}", exc_info=True)
            return None
    
    # =========================================================
    # Interim Data Management
    # =========================================================
    
    def save_interim(
        self,
        data_name: str,
        data: Any,
        stage: str = "processed"
    ) -> bool:
        """
        Save interim pipeline data.
        
        Args:
            data_name: Data identifier
            data: Data to save
            stage: Pipeline stage ("raw", "interim", "processed")
        
        Returns:
            bool: True if successful
        
        Example:
            >>> storage = get_storage()
            >>> storage.save_interim("cleaned_interactions", df, stage="interim")
        """
        try:
            data_bytes = pickle.dumps(data)
            object_key = f"{stage}/{data_name}.pkl"
            
            success = self.minio.upload_data(
                bucket_name=self.settings.minio.interim_bucket,
                object_name=object_key,
                data=data_bytes
            )
            
            if success:
                logger.info(f"Saved interim data: {stage}/{data_name}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to save interim data {data_name}: {e}", exc_info=True)
            return False
    
    def load_interim(
        self,
        data_name: str,
        stage: str = "processed"
    ) -> Optional[Any]:
        """
        Load interim pipeline data.
        
        Args:
            data_name: Data identifier
            stage: Pipeline stage
        
        Returns:
            Data object, or None if not found
        
        Example:
            >>> storage = get_storage()
            >>> df = storage.load_interim("cleaned_interactions", stage="interim")
        """
        try:
            object_key = f"{stage}/{data_name}.pkl"
            
            data_bytes = self.minio.download_data(
                bucket_name=self.settings.minio.interim_bucket,
                object_name=object_key
            )
            
            if data_bytes is None:
                return None
            
            data = pickle.loads(data_bytes)
            logger.info(f"Loaded interim data: {stage}/{data_name}")
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to load interim data {data_name}: {e}", exc_info=True)
            return None
    
    # =========================================================
    # Health Check
    # =========================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all storage backends.
        
        Returns:
            dict: Health status for MinIO and Redis
        
        Example:
            >>> storage = get_storage()
            >>> status = storage.health_check()
            >>> if status['healthy']:
            ...     print("All storage systems operational")
        """
        minio_health = self.minio.health_check()
        redis_health = self.redis.health_check()
        
        result = {
            "healthy": minio_health["healthy"] and redis_health["healthy"],
            "minio": minio_health,
            "redis": redis_health
        }
        
        logger.info(
            f"Storage health check: "
            f"MinIO={minio_health['healthy']}, Redis={redis_health['healthy']}"
        )
        
        return result
    
    # =========================================================
    # Helper Methods
    # =========================================================
    
    def _get_model_key(self, model_name: str, version: str) -> str:
        """Generate MinIO object key for model."""
        return f"models/{model_name}/{version}/model.pkl"
    
    def _get_model_metadata_key(self, model_name: str, version: str) -> str:
        """Generate MinIO object key for model metadata."""
        return f"models/{model_name}/{version}/metadata.json"
    
    def _get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version for a model."""
        versions = self.list_model_versions(model_name)
        return versions[0] if versions else None


# =========================================================
# Singleton Instance
# =========================================================

_storage_instance: Optional[StorageManager] = None


def get_storage() -> StorageManager:
    """
    Get singleton storage manager instance.
    
    Returns:
        StorageManager: Singleton storage manager
    
    Example:
        >>> storage = get_storage()
        >>> storage.save_model(model, "ambient", "v1.0.0")
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageManager()
    return _storage_instance


if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "=" * 70)
    print("💾 STORAGE MANAGER TEST")
    print("=" * 70)
    
    # Initialize storage
    storage = get_storage()
    
    # Health check
    print("\n📊 Health Check:")
    health = storage.health_check()
    print(f"Overall Health: {health['healthy']}")
    print(f"MinIO: {health['minio']['healthy']}")
    print(f"Redis: {health['redis']['healthy']}")
    
    if health['healthy']:
        print("\n🧪 Testing Storage Operations:")
        
        # Test model save/load
        print("\n1️⃣ Model Operations:")
        test_model = {"type": "test", "params": {"alpha": 0.5}}
        
        saved = storage.save_model(
            test_model,
            "test_model",
            "v0.0.1",
            metadata={"test": True}
        )
        print(f"  ✓ Save model: {saved}")
        
        if saved:
            loaded = storage.load_model("test_model", "v0.0.1")
            print(f"  ✓ Load model: {loaded is not None}")
            
            versions = storage.list_model_versions("test_model")
            print(f"  ✓ List versions: {versions}")
            
            metadata = storage.get_model_metadata("test_model", "v0.0.1")
            print(f"  ✓ Get metadata: {metadata is not None}")
            
            deleted = storage.delete_model("test_model", "v0.0.1")
            print(f"  ✓ Delete model: {deleted}")
        
        # Test features save/load
        print("\n2️⃣ Feature Operations:")
        test_features = {"user_id": [1, 2, 3], "feature": [0.1, 0.2, 0.3]}
        
        saved = storage.save_features("test_features", test_features, format="json")
        print(f"  ✓ Save features: {saved}")
        
        if saved:
            loaded = storage.load_features("test_features", format="json")
            print(f"  ✓ Load features: {loaded is not None}")
    
    print("\n✅ Storage manager test complete")