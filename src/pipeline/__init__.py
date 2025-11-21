# pipeline package
"""
Data pipeline package for Bookend Recommendation System.

This package provides ETL (Extract, Transform, Load) functionality:
- Extractors: Read raw data from various sources
- Validators: Ensure data quality and schema compliance
- Transformers: Clean and engineer features
- Loaders: Store processed data to database/storage
"""

from .extractors.base_extractor import BaseExtractor
from .extractors.file_extractor import FileExtractor, MultiFileExtractor
from .validators.schema_validator import SchemaValidator
from .validators.quality_validator import QualityValidator
from .loaders.postgres_loader import PostgresLoader

__all__ = [
    "BaseExtractor",
    "FileExtractor",
    "MultiFileExtractor",
    "SchemaValidator",
    "QualityValidator",
    "PostgresLoader",
]
