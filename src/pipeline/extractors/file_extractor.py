import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from .base_extractor import BaseExtractor, ExtractionError, FileNotFoundError, InvalidFormatError

logger = logging.getLogger(__name__)


class FileExtractor(BaseExtractor):
    """Extract data from local files with automatic format detection and clean normalization."""

    SUPPORTED_FORMATS = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".csv": "csv",
        ".tsv": "tsv",
        ".parquet": "parquet",
        ".txt": "txt",
    }

    def __init__(self, source_path: Union[str, Path], file_format: Optional[str] = None, encoding: str = "utf-8"):
        source_path = Path(source_path)
        super().__init__(source_path)
        self.encoding = encoding
        self.file_format = file_format or self._detect_format()

    def _detect_format(self) -> str:
        suffix = self.source_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise InvalidFormatError(f"Unsupported file format: {suffix}")
        return self.SUPPORTED_FORMATS[suffix]

    def validate_source(self) -> bool:
        if not self.source_path or not self.source_path.exists():
            self.logger.error(f"File not found: {self.source_path}")
            return False
        if not self.source_path.is_file():
            self.logger.error(f"Not a file: {self.source_path}")
            return False
        return True

    def extract(self) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        if not self.validate_source():
            raise FileNotFoundError(f"Invalid source: {self.source_path}")

        self._log_extraction_start()
        try:
            loader = {
                "json": self._extract_json,
                "jsonl": self._extract_jsonl,
                "csv": self._extract_csv,
                "tsv": self._extract_tsv,
                "parquet": self._extract_parquet,
                "txt": self._extract_txt,
            }.get(self.file_format)

            if not loader:
                raise InvalidFormatError(f"Unsupported format: {self.file_format}")

            data = loader()
            data = self._clean_extracted_data(data)

            self._log_extraction_complete(len(data) if hasattr(data, "__len__") else 0)
            return data
        except Exception as e:
            self._log_extraction_error(e)
            raise ExtractionError(f"Failed to extract {self.source_path}: {e}") from e

    def _extract_json(self) -> List[Dict[str, Any]]:
        with open(self.source_path, "r", encoding=self.encoding) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def _extract_jsonl(self) -> List[Dict[str, Any]]:
        """✅ JSONL 전용: 각 라인마다 중복 키 제거하고 flatten"""
        records = []
        with open(self.source_path, "r", encoding=self.encoding) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_record = json.loads(line)
                    cleaned_record = self._flatten_and_deduplicate_record(raw_record)
                    records.append(cleaned_record)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON at line {i}: {e}")
        return records

    def _extract_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_path, encoding=self.encoding, low_memory=False)
        df.columns = df.columns.str.strip()
        return df

    def _extract_tsv(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_path, sep="\t", encoding=self.encoding, low_memory=False)
        df.columns = df.columns.str.strip()
        return df

    def _extract_parquet(self) -> pd.DataFrame:
        return pd.read_parquet(self.source_path)

    def _extract_txt(self) -> List[Dict[str, Any]]:
        try:
            return self._extract_jsonl()
        except Exception:
            pass
        with open(self.source_path, "r", encoding=self.encoding) as f:
            return [{"line_number": i + 1, "content": line.strip()} for i, line in enumerate(f) if line.strip()]

    def _flatten_and_deduplicate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ 개선된 로직:
        1. 최상위 필수 키 우선 저장
        2. properties 내부 키들을 flatten (중복 제외)
        3. 최종 결과에 중복 키가 없도록 보장
        """
        result = {}
        seen_keys = set()
        
        # Step 1: 최상위 필수 키 먼저 추가 (우선순위 높음)
        top_level_keys = ["distinct_id", "event_name", "insert_id", "device_id", "user_id", "time"]
        for key in top_level_keys:
            if key in record:
                result[key] = record[key]
                seen_keys.add(key)
        
        # Step 2: properties를 flatten (중복 키 제외)
        properties = record.get("properties", {})
        if isinstance(properties, dict):
            for prop_key, prop_value in properties.items():
                if prop_key not in seen_keys:  # 중복 방지
                    result[prop_key] = prop_value
                    seen_keys.add(prop_key)
        
        # Step 3: 나머지 최상위 키도 추가 (중복 제외)
        for key, value in record.items():
            if key != "properties" and key not in seen_keys:
                result[key] = value
                seen_keys.add(key)
        
        return result

    def _clean_extracted_data(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """✅ 최종 정리: DataFrame의 suffix 컬럼 제거"""
        if isinstance(data, pd.DataFrame):
            data.columns = data.columns.astype(str)
            # _mNNN 또는 .N suffix 제거
            data.columns = data.columns.str.replace(r'\.\d+$', '', regex=True)
            data.columns = data.columns.str.replace(r'_m\d+$', '', regex=True)
            # 중복 컬럼 제거 (첫 번째만 유지)
            data = data.loc[:, ~data.columns.duplicated()]
            return data

        # List[Dict]인 경우는 이미 _flatten_and_deduplicate_record에서 처리됨
        return data

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        if self.source_path.exists():
            stat = self.source_path.stat()
            metadata.update({
                "file_format": self.file_format,
                "encoding": self.encoding,
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "last_modified": stat.st_mtime,
            })
        return metadata


class MultiFileExtractor:
    """Combine multiple files into a single dataset"""

    def __init__(self, file_paths: List[Union[str, Path]], file_format: Optional[str] = None, encoding: str = "utf-8"):
        self.file_paths = [Path(p) for p in file_paths]
        self.file_format = file_format
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)

    def extract(self, combine: bool = True) -> Union[pd.DataFrame, List[Any]]:
        results = []
        for path in self.file_paths:
            try:
                extractor = FileExtractor(path, file_format=self.file_format, encoding=self.encoding)
                results.append(extractor.extract())
            except Exception as e:
                self.logger.error(f"Extraction failed for {path}: {e}")

        if not combine:
            return results

        if not results:
            return []

        first = results[0]
        if isinstance(first, pd.DataFrame):
            df = pd.concat(results, ignore_index=True)
            df.columns = df.columns.str.replace(r'_m\d+$', '', regex=True)
            df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
            df = df.loc[:, ~df.columns.duplicated()]
            return df

        if isinstance(first, list):
            combined = []
            for r in results:
                combined.extend(r)
            return combined

        return results