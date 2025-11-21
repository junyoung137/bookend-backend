"""
Configuration loader utilities for Bookend Recommendation System.

Provides YAML configuration loading with:
1. Type validation
2. Environment variable interpolation
3. Default value fallback
4. Error handling

Principles:
- Single Responsibility: One loader per format
- Error Handling: Graceful degradation
- Type Safety: Schema validation
- Caching: Avoid redundant file reads
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

import yaml

# Import settings (Single Source of Truth for paths)
from config.settings import get_settings

logger = logging.getLogger(__name__)


# =========================================================
# YAML Configuration Loader
# =========================================================

def load_yaml_config(
    config_path: Path,
    validate_schema: bool = False,
    required_keys: Optional[list] = None
) -> Dict[str, Any]:
    """
    Load YAML configuration file with validation.
    
    Args:
        config_path: Path to YAML file
        validate_schema: Whether to validate required keys
        required_keys: List of required top-level keys
    
    Returns:
        Dictionary containing configuration
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If required keys are missing
    
    Example:
        >>> config = load_yaml_config(
        ...     Path("config/model_config.yaml"),
        ...     required_keys=["collaborative", "hybrid"]
        ... )
        >>> print(config['collaborative']['user_cf']['k_neighbors'])
        20
    """
    try:
        # Check if file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.debug(f"Loading YAML config from: {config_path}")
        
        # Load YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate it's a dictionary
        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid config format: expected dict, got {type(config)}"
            )
        
        # Validate required keys
        if validate_schema and required_keys:
            missing_keys = set(required_keys) - set(config.keys())
            if missing_keys:
                raise ValueError(
                    f"Missing required config keys: {missing_keys}"
                )
        
        logger.info(
            f"Successfully loaded config from {config_path} "
            f"({len(config)} top-level keys)"
        )
        
        return config
    
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}", exc_info=True)
        raise
    
    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        raise


# =========================================================
# Cached Model Config Loader
# =========================================================

@lru_cache(maxsize=1)
def load_model_config() -> Dict[str, Any]:
    """
    Load model configuration with caching.
    
    Uses settings.model_config_path as source of truth.
    Results are cached to avoid redundant file reads.
    
    Returns:
        Dictionary containing model configuration
    
    Example:
        >>> config = load_model_config()
        >>> user_cf_config = config['collaborative']['user_cf']
        >>> k_neighbors = user_cf_config['k_neighbors']
    """
    try:
        settings = get_settings()
        config_path = Path(settings.model_config_path)
        
        logger.info(f"Loading model config from: {config_path}")
        
        # Load with validation
        config = load_yaml_config(
            config_path,
            validate_schema=True,
            required_keys=['metadata', 'collaborative', 'hybrid']
        )
        
        # Log metadata
        if 'metadata' in config:
            metadata = config['metadata']
            logger.info(
                f"Model config loaded: version={metadata.get('version')}, "
                f"last_updated={metadata.get('last_updated')}"
            )
        
        return config
    
    except Exception as e:
        logger.error(f"Failed to load model config: {e}", exc_info=True)
        # Return minimal default config
        return _get_default_model_config()


# =========================================================
# Config Value Getter with Default
# =========================================================

def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """
    Get nested configuration value with dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "collaborative.user_cf.k_neighbors")
        default: Default value if key not found
        required: Whether to raise error if key missing
    
    Returns:
        Configuration value or default
    
    Raises:
        KeyError: If required=True and key not found
    
    Example:
        >>> config = load_model_config()
        >>> k = get_config_value(config, "collaborative.user_cf.k_neighbors", default=20)
        >>> print(k)
        20
    """
    try:
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(f"Cannot access key '{key}' in non-dict value")
        
        return value
    
    except KeyError as e:
        if required:
            raise KeyError(f"Required config key not found: {key_path}") from e
        
        logger.debug(f"Config key '{key_path}' not found, using default: {default}")
        return default


# =========================================================
# Config Section Getter
# =========================================================

def get_config_section(
    config: Dict[str, Any],
    section: str,
    required: bool = False
) -> Dict[str, Any]:
    """
    Get entire configuration section.
    
    Args:
        config: Configuration dictionary
        section: Section name (top-level key)
        required: Whether to raise error if section missing
    
    Returns:
        Section dictionary or empty dict
    
    Raises:
        KeyError: If required=True and section not found
    
    Example:
        >>> config = load_model_config()
        >>> user_cf_config = get_config_section(
        ...     config['collaborative'],
        ...     'user_cf',
        ...     required=True
        ... )
    """
    try:
        if section not in config:
            if required:
                raise KeyError(f"Required config section not found: {section}")
            
            logger.warning(f"Config section '{section}' not found, returning empty dict")
            return {}
        
        section_config = config[section]
        
        if not isinstance(section_config, dict):
            raise TypeError(
                f"Config section '{section}' is not a dict: {type(section_config)}"
            )
        
        return section_config
    
    except Exception as e:
        logger.error(f"Failed to get config section '{section}': {e}")
        if required:
            raise
        return {}


# =========================================================
# Default Config Generator
# =========================================================

def _get_default_model_config() -> Dict[str, Any]:
    """
    Get default model configuration as fallback.
    
    Returns:
        Minimal default configuration
    """
    logger.warning("Using default model configuration (config file load failed)")
    
    return {
        "metadata": {
            "version": "0.1.0",
            "last_updated": "2025-01-01",
            "author": "Bookend Team"
        },
        "collaborative": {
            "user_cf": {
                "similarity_metric": "cosine",
                "k_neighbors": 20,
                "min_common_items": 3,
                "normalize": True
            },
            "item_cf": {
                "similarity_metric": "cosine",
                "k_neighbors": 50,
                "min_common_users": 5,
                "normalize": True
            }
        },
        "hybrid": {
            "weights": {
                "user_cf": 0.3,
                "item_cf": 0.3,
                "recency": 0.2,
                "context": 0.2
            },
            "mmr": {
                "lambda": 0.5,
                "top_k": 10
            }
        },
        "features": {
            "user": [
                "total_paraphrases",
                "preferred_tone",
                "last_7d_count"
            ],
            "item": [
                "popularity_score",
                "quality_score",
                "category"
            ]
        },
        "evaluation": {
            "metrics": [
                "precision@5",
                "recall@10",
                "ndcg@10"
            ],
            "test_split": 0.2
        }
    }


# =========================================================
# Config Validator
# =========================================================

def validate_config(
    config: Dict[str, Any],
    schema: Dict[str, type]
) -> bool:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema dictionary mapping keys to expected types
    
    Returns:
        True if valid, False otherwise
    
    Example:
        >>> schema = {
        ...     "k_neighbors": int,
        ...     "similarity_metric": str,
        ...     "normalize": bool
        ... }
        >>> is_valid = validate_config(user_cf_config, schema)
    """
    try:
        for key, expected_type in schema.items():
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                return False
            
            value = config[key]
            if not isinstance(value, expected_type):
                logger.error(
                    f"Config key '{key}' has wrong type: "
                    f"expected {expected_type}, got {type(value)}"
                )
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"Config validation error: {e}")
        return False


if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging(environment="development", debug=True)
    
    print("=" * 70)
    print("CONFIG LOADER UTILITIES DEMO")
    print("=" * 70)
    
    # Load model config
    print("\n1️⃣ Loading Model Config:")
    try:
        config = load_model_config()
        print(f"✅ Loaded config with {len(config)} sections")
        print(f"   Sections: {list(config.keys())}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
    
    # Get nested value
    print("\n2️⃣ Getting Nested Config Value:")
    k_neighbors = get_config_value(
        config,
        "collaborative.user_cf.k_neighbors",
        default=20
    )
    print(f"   k_neighbors: {k_neighbors}")
    
    # Get section
    print("\n3️⃣ Getting Config Section:")
    user_cf_config = get_config_section(
        config.get('collaborative', {}),
        'user_cf'
    )
    print(f"   User CF Config: {user_cf_config}")
    
    # Validate config
    print("\n4️⃣ Validating Config:")
    schema = {
        "similarity_metric": str,
        "k_neighbors": int,
        "normalize": bool
    }
    is_valid = validate_config(user_cf_config, schema)
    print(f"   Validation result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    print("\n✅ Config loader demo completed!")