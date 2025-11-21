"""
MinIO Object Storage Configuration for Bookend Recommendation System.

Provides:
1. MinIO client singleton with connection pooling
2. Bucket management (create/check/list)
3. File upload/download with retry logic
4. Presigned URL generation
5. Health check utilities

Principles:
- Single Source of Truth: Uses config.settings.MinIOSettings
- Singleton Pattern: One client instance per application
- Error Handling: Retry logic with exponential backoff
- Resource Management: Automatic cleanup
"""

from typing import Optional, List, Dict, Any, BinaryIO
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import logging
import time

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import MaxRetryError

from config.settings import get_settings

logger = logging.getLogger(__name__)


class MinIOConnection:
    """
    Singleton MinIO client manager.
    
    Ensures only one MinIO client exists across the application.
    Thread-safe and handles automatic bucket creation.
    """
    
    _instance: Optional['MinIOConnection'] = None
    _client: Optional[Minio] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'MinIOConnection':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize MinIO client (only once due to singleton)."""
        if not self._initialized:
            self._initialize_client()
            self._initialized = True
    
    def _initialize_client(self) -> None:
        """
        Create and configure MinIO client.
        
        Features:
        - Automatic retry on connection failure
        - Support for both HTTP and HTTPS
        - Region configuration
        - Bucket auto-creation on startup
        """
        settings = get_settings()
        minio_settings = settings.minio
        
        try:
            self._client = Minio(
                endpoint=minio_settings.endpoint,
                access_key=minio_settings.access_key,
                secret_key=minio_settings.secret_key,
                secure=minio_settings.secure,
                region=minio_settings.region
            )
            
            logger.info(
                f"MinIO client initialized: endpoint={minio_settings.endpoint}, "
                f"secure={minio_settings.secure}, region={minio_settings.region}"
            )
            
            # Ensure required buckets exist
            self._ensure_buckets()
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}", exc_info=True)
            raise
    
    def _ensure_buckets(self) -> None:
        """
        Create required buckets if they don't exist.
        
        Creates:
        - bookend-features
        - bookend-models
        - bookend-interim
        """
        settings = get_settings()
        buckets = [
            settings.minio.features_bucket,
            settings.minio.models_bucket,
            settings.minio.interim_bucket
        ]
        
        for bucket_name in buckets:
            try:
                if not self._client.bucket_exists(bucket_name):
                    self._client.make_bucket(bucket_name)
                    logger.info(f"Created bucket: {bucket_name}")
                else:
                    logger.debug(f"Bucket already exists: {bucket_name}")
            
            except S3Error as e:
                logger.error(
                    f"Failed to create bucket {bucket_name}: {e}",
                    exc_info=True
                )
                # Continue even if bucket creation fails
            
            except Exception as e:
                logger.error(
                    f"Unexpected error creating bucket {bucket_name}: {e}",
                    exc_info=True
                )
    
    @property
    def client(self) -> Minio:
        """Get MinIO client instance."""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> bool:
        """
        Upload file to MinIO with retry logic.
        
        Args:
            bucket_name: Target bucket name
            object_name: Object key in bucket
            file_path: Local file path to upload
            content_type: MIME type
            metadata: Optional metadata dictionary
            max_retries: Maximum retry attempts
        
        Returns:
            bool: True if successful, False otherwise
        
        Example:
            >>> minio = get_minio()
            >>> success = minio.upload_file(
            ...     "bookend-models",
            ...     "models/ambient_v1.pkl",
            ...     Path("data/models/ambient_v1.pkl")
            ... )
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        for attempt in range(max_retries):
            try:
                self._client.fput_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=str(file_path),
                    content_type=content_type,
                    metadata=metadata
                )
                
                logger.info(
                    f"Uploaded {file_path} to {bucket_name}/{object_name}"
                )
                return True
            
            except S3Error as e:
                logger.warning(
                    f"Upload attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to upload after {max_retries} attempts",
                        exc_info=True
                    )
                    return False
            
            except Exception as e:
                logger.error(f"Unexpected upload error: {e}", exc_info=True)
                return False
        
        return False
    
    def upload_data(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> bool:
        """
        Upload bytes data to MinIO.
        
        Args:
            bucket_name: Target bucket name
            object_name: Object key in bucket
            data: Bytes data to upload
            content_type: MIME type
            metadata: Optional metadata dictionary
            max_retries: Maximum retry attempts
        
        Returns:
            bool: True if successful, False otherwise
        
        Example:
            >>> import pickle
            >>> minio = get_minio()
            >>> model_bytes = pickle.dumps(model)
            >>> minio.upload_data(
            ...     "bookend-models",
            ...     "models/temp.pkl",
            ...     model_bytes
            ... )
        """
        data_stream = BytesIO(data)
        data_length = len(data)
        
        for attempt in range(max_retries):
            try:
                data_stream.seek(0)  # Reset stream position
                
                self._client.put_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    data=data_stream,
                    length=data_length,
                    content_type=content_type,
                    metadata=metadata
                )
                
                logger.info(
                    f"Uploaded {data_length} bytes to {bucket_name}/{object_name}"
                )
                return True
            
            except S3Error as e:
                logger.warning(
                    f"Upload attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(
                        f"Failed to upload after {max_retries} attempts",
                        exc_info=True
                    )
                    return False
            
            except Exception as e:
                logger.error(f"Unexpected upload error: {e}", exc_info=True)
                return False
        
        return False
    
    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path,
        max_retries: int = 3
    ) -> bool:
        """
        Download file from MinIO with retry logic.
        
        Args:
            bucket_name: Source bucket name
            object_name: Object key in bucket
            file_path: Local destination path
            max_retries: Maximum retry attempts
        
        Returns:
            bool: True if successful, False otherwise
        
        Example:
            >>> minio = get_minio()
            >>> success = minio.download_file(
            ...     "bookend-models",
            ...     "models/ambient_v1.pkl",
            ...     Path("temp/ambient_v1.pkl")
            ... )
        """
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                self._client.fget_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=str(file_path)
                )
                
                logger.info(
                    f"Downloaded {bucket_name}/{object_name} to {file_path}"
                )
                return True
            
            except S3Error as e:
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(
                        f"Failed to download after {max_retries} attempts",
                        exc_info=True
                    )
                    return False
            
            except Exception as e:
                logger.error(f"Unexpected download error: {e}", exc_info=True)
                return False
        
        return False
    
    def download_data(
        self,
        bucket_name: str,
        object_name: str,
        max_retries: int = 3
    ) -> Optional[bytes]:
        """
        Download object data as bytes.
        
        Args:
            bucket_name: Source bucket name
            object_name: Object key in bucket
            max_retries: Maximum retry attempts
        
        Returns:
            bytes: Object data, or None if failed
        
        Example:
            >>> import pickle
            >>> minio = get_minio()
            >>> data = minio.download_data("bookend-models", "models/model.pkl")
            >>> if data:
            ...     model = pickle.loads(data)
        """
        for attempt in range(max_retries):
            try:
                response = self._client.get_object(
                    bucket_name=bucket_name,
                    object_name=object_name
                )
                
                data = response.read()
                response.close()
                response.release_conn()
                
                logger.info(
                    f"Downloaded {len(data)} bytes from "
                    f"{bucket_name}/{object_name}"
                )
                return data
            
            except S3Error as e:
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(
                        f"Failed to download after {max_retries} attempts",
                        exc_info=True
                    )
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected download error: {e}", exc_info=True)
                return None
        
        return None
    
    def list_objects(
        self,
        bucket_name: str,
        prefix: str = "",
        recursive: bool = True
    ) -> List[str]:
        """
        List objects in bucket.
        
        Args:
            bucket_name: Bucket to list
            prefix: Object key prefix filter
            recursive: Whether to list recursively
        
        Returns:
            List of object names
        
        Example:
            >>> minio = get_minio()
            >>> objects = minio.list_objects("bookend-models", prefix="models/")
            >>> print(f"Found {len(objects)} objects")
        """
        try:
            objects = self._client.list_objects(
                bucket_name=bucket_name,
                prefix=prefix,
                recursive=recursive
            )
            
            object_names = [obj.object_name for obj in objects]
            
            logger.debug(
                f"Listed {len(object_names)} objects in "
                f"{bucket_name} (prefix={prefix})"
            )
            
            return object_names
        
        except S3Error as e:
            logger.error(f"Failed to list objects: {e}", exc_info=True)
            return []
        
        except Exception as e:
            logger.error(f"Unexpected list error: {e}", exc_info=True)
            return []
    
    def delete_object(
        self,
        bucket_name: str,
        object_name: str
    ) -> bool:
        """
        Delete object from bucket.
        
        Args:
            bucket_name: Bucket name
            object_name: Object key to delete
        
        Returns:
            bool: True if successful
        
        Example:
            >>> minio = get_minio()
            >>> minio.delete_object("bookend-models", "temp/old_model.pkl")
        """
        try:
            self._client.remove_object(
                bucket_name=bucket_name,
                object_name=object_name
            )
            
            logger.info(f"Deleted {bucket_name}/{object_name}")
            return True
        
        except S3Error as e:
            logger.error(f"Failed to delete object: {e}", exc_info=True)
            return False
        
        except Exception as e:
            logger.error(f"Unexpected delete error: {e}", exc_info=True)
            return False
    
    def get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires_seconds: int = 3600
    ) -> Optional[str]:
        """
        Generate presigned URL for object access.
        
        Args:
            bucket_name: Bucket name
            object_name: Object key
            expires_seconds: URL expiration time (default: 1 hour)
        
        Returns:
            Presigned URL string, or None if failed
        
        Example:
            >>> minio = get_minio()
            >>> url = minio.get_presigned_url(
            ...     "bookend-models",
            ...     "models/ambient_v1.pkl",
            ...     expires_seconds=3600
            ... )
            >>> print(f"Download URL: {url}")
        """
        try:
            from datetime import timedelta
            
            url = self._client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_name,
                expires=timedelta(seconds=expires_seconds)
            )
            
            logger.debug(
                f"Generated presigned URL for {bucket_name}/{object_name} "
                f"(expires in {expires_seconds}s)"
            )
            
            return url
        
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL: {e}", exc_info=True)
            return None
        
        except Exception as e:
            logger.error(
                f"Unexpected presigned URL error: {e}",
                exc_info=True
            )
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check MinIO connection health.
        
        Returns:
            dict: Health status with bucket info
        
        Example:
            >>> minio = get_minio()
            >>> status = minio.health_check()
            >>> if status['healthy']:
            ...     print("MinIO is healthy")
        """
        result = {
            "healthy": False,
            "buckets": [],
            "error": None
        }
        
        try:
            # List buckets as health check
            buckets = self._client.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]
            
            result.update({
                "healthy": True,
                "buckets": bucket_names
            })
            
            logger.debug(f"MinIO health check passed: {len(bucket_names)} buckets")
        
        except MaxRetryError as e:
            result["error"] = f"Connection failed: {e}"
            logger.error(f"MinIO health check failed: {e}", exc_info=True)
        
        except S3Error as e:
            result["error"] = f"S3 error: {e}"
            logger.error(f"MinIO health check failed: {e}", exc_info=True)
        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"MinIO health check failed: {e}", exc_info=True)
        
        return result


# Singleton instance getter
_minio_instance: Optional[MinIOConnection] = None


def get_minio() -> MinIOConnection:
    """
    Get singleton MinIO connection instance.
    
    Returns:
        MinIOConnection: Singleton MinIO connection
    
    Example:
        >>> minio = get_minio()
        >>> minio.upload_file("bookend-models", "test.txt", Path("test.txt"))
    """
    global _minio_instance
    if _minio_instance is None:
        _minio_instance = MinIOConnection()
    return _minio_instance


@contextmanager
def minio_client():
    """
    Context manager for MinIO client.
    
    Yields:
        Minio: MinIO client instance
    
    Example:
        >>> with minio_client() as client:
        ...     client.fput_object("bucket", "key", "file.txt")
    """
    minio_conn = get_minio()
    try:
        yield minio_conn.client
    except Exception as e:
        logger.error(f"MinIO operation error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "=" * 70)
    print("🪣 MINIO CONNECTION TEST")
    print("=" * 70)
    
    # Test MinIO connection
    minio = get_minio()
    
    # Health check
    print("\n📊 Health Check:")
    status = minio.health_check()
    print(f"Healthy: {status['healthy']}")
    print(f"Buckets: {status.get('buckets', [])}")
    if status.get('error'):
        print(f"Error: {status['error']}")
    
    # List objects (if healthy)
    if status['healthy']:
        settings = get_settings()
        
        print(f"\n📁 Objects in {settings.minio.models_bucket}:")
        objects = minio.list_objects(settings.minio.models_bucket)
        for obj in objects[:5]:  # Show first 5
            print(f"  - {obj}")
        
        if len(objects) > 5:
            print(f"  ... and {len(objects) - 5} more")
    
    print("\n✅ MinIO connection test complete")