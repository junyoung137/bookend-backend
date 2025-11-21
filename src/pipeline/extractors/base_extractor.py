from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Base exception for extraction errors."""


class FileNotFoundError(ExtractionError):
    """Raised when source file is not found."""


class InvalidFormatError(ExtractionError):
    """Raised when file format is invalid."""


class BaseExtractor(ABC):
    """Abstract base class for all data extractors."""

    def __init__(self, source_path: Optional[Path] = None):
        self.source_path = source_path
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def extract(self) -> Any:
        """Extract data from source."""

    @abstractmethod
    def validate_source(self) -> bool:
        """Validate that source exists and is accessible."""

    def get_metadata(self) -> Dict[str, Any]:
        """Return basic metadata about the source."""
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "extractor_type": self.__class__.__name__,
        }

    def _log_extraction_start(self) -> None:
        self.logger.info(f"Starting extraction from {self.source_path}")

    def _log_extraction_complete(self, record_count: int) -> None:
        self.logger.info(f"Extraction complete: {record_count} records extracted")

    def _log_extraction_error(self, error: Exception) -> None:
        self.logger.error(f"Extraction failed: {error}", exc_info=True)


class BatchExtractor(BaseExtractor):
    """Base class for batch-based extractors."""

    def __init__(self, source_path: Optional[Path] = None, batch_size: int = 1000):
        super().__init__(source_path)
        self.batch_size = batch_size

    @abstractmethod
    def extract_batch(self, offset: int) -> Any:
        """Extract a single batch of data."""

    def extract(self) -> Any:
        self._log_extraction_start()
        all_data, offset = [], 0

        try:
            while True:
                batch = self.extract_batch(offset)
                if not batch:
                    break
                all_data.extend(batch)
                offset += self.batch_size
                self.logger.debug(f"Extracted batch offset={offset}, size={len(batch)}")
            self._log_extraction_complete(len(all_data))
            return all_data
        except Exception as e:
            self._log_extraction_error(e)
            raise ExtractionError(f"Batch extraction failed: {e}") from e


if __name__ == "__main__":
    print("BaseExtractor is an abstract class.")
    print("Inherit and implement extract() and validate_source().")
