from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
import traceback

import pandas as pd

from src.pipeline.extractors.file_extractor import FileExtractor, MultiFileExtractor
from src.pipeline.validators.schema_validator import SchemaValidator, ValidationRule, DataType
from src.pipeline.validators.quality_validator import QualityValidator
from src.pipeline.transformers.user_transformer import UserTransformer
from src.pipeline.transformers.interaction_transformer import InteractionTransformer
from src.pipeline.loaders.postgres_loader import PostgresLoader
from src.database.models import User, Interaction
from config.settings import get_settings

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    EXTRACT = "extract"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    LOAD = "load"
    COMPLETE = "complete"
    FAILED = "failed"


class PipelineStatus:
    """Track pipeline execution status."""
    
    def __init__(self):
        self.stage = PipelineStage.EXTRACT
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.records_processed = 0
        self.records_loaded = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stage_durations: Dict[str, float] = {}
    
    def mark_stage_start(self, stage: PipelineStage):
        self.stage = stage
        self._stage_start_time = datetime.now()
        logger.info(f"Starting stage: {stage.value}")
    
    def mark_stage_end(self, stage: PipelineStage):
        duration = (datetime.now() - self._stage_start_time).total_seconds()
        self.stage_durations[stage.value] = duration
        logger.info(f"Completed stage: {stage.value} ({duration:.2f}s)")
    
    def add_error(self, error: str):
        self.errors.append(error)
        logger.error(error)
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
        logger.warning(warning)
    
    def finalize(self, success: bool):
        self.end_time = datetime.now()
        self.stage = PipelineStage.COMPLETE if success else PipelineStage.FAILED
        duration = (self.end_time - self.start_time).total_seconds()
        
        if success:
            logger.info(f"Pipeline completed successfully in {duration:.2f}s")
        else:
            logger.error(f"Pipeline failed after {duration:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            "status": self.stage.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "records_processed": self.records_processed,
            "records_loaded": self.records_loaded,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "stage_durations": self.stage_durations
        }


class PipelineOrchestrator:
    """
    Main pipeline orchestrator for ETL workflow.
    """
    
    def __init__(
        self,
        settings: Optional[Any] = None,
        enable_validation: bool = True,
        enable_quality_checks: bool = True,
        batch_size: int = 1000
    ):
        self.settings = settings or get_settings()
        self.enable_validation = enable_validation
        self.enable_quality_checks = enable_quality_checks
        self.batch_size = batch_size
        
        self.logger = logging.getLogger(__name__)
        self.status = PipelineStatus()
        
        self.user_transformer = UserTransformer()
        self.interaction_transformer = InteractionTransformer()
        self.loader = PostgresLoader()
        self.quality_validator = QualityValidator()
    
    # ==================== PIPELINES ====================

    def run_user_pipeline(self, input_file: Path, mode: str = "upsert") -> Dict[str, Any]:
        """User 파이프라인 실행"""
        self.status = PipelineStatus()
        try:
            self.status.mark_stage_start(PipelineStage.EXTRACT)
            raw_data = self._extract_user_data(input_file)
            self.status.records_processed = len(raw_data)
            self.status.mark_stage_end(PipelineStage.EXTRACT)
            
            if self.enable_validation:
                self.status.mark_stage_start(PipelineStage.VALIDATE)
                self._validate_user_data(raw_data)
                self.status.mark_stage_end(PipelineStage.VALIDATE)
            
            self.status.mark_stage_start(PipelineStage.TRANSFORM)
            transformed_df = self._transform_user_data(raw_data)
            if self.enable_quality_checks:
                self._run_quality_checks(transformed_df, "User")
            self.status.mark_stage_end(PipelineStage.TRANSFORM)
            
            self.status.mark_stage_start(PipelineStage.LOAD)
            loaded_count = self._load_user_data(transformed_df, mode)
            self.status.records_loaded = loaded_count
            self.status.mark_stage_end(PipelineStage.LOAD)
            
            self.status.finalize(success=True)
            return self.status.get_summary()
        
        except Exception as e:
            self.status.add_error(f"Pipeline failed: {str(e)}")
            self.status.add_error(traceback.format_exc())
            self.status.finalize(success=False)
            raise

    def run_interaction_pipeline(self, input_file: Path, mode: str = "insert") -> Dict[str, Any]:
        """✅ Interaction 파이프라인 실행 (User ID 매핑 포함)"""
        self.status = PipelineStatus()
        try:
            self.status.mark_stage_start(PipelineStage.EXTRACT)
            raw_data = self._extract_interaction_data(input_file)
            self.status.records_processed = len(raw_data)
            self.status.mark_stage_end(PipelineStage.EXTRACT)
            
            if self.enable_validation:
                self.status.mark_stage_start(PipelineStage.VALIDATE)
                self._validate_interaction_data(raw_data)
                self.status.mark_stage_end(PipelineStage.VALIDATE)
            
            self.status.mark_stage_start(PipelineStage.TRANSFORM)
            # ✅ load_user_mapping=True로 User ID 매핑 활성화
            transformed_df = self._transform_interaction_data(raw_data, load_user_mapping=True)
            
            # ✅ User ID 매핑 실패 통계 로깅
            if "user_id" in transformed_df.columns:
                total = len(transformed_df)
                mapped = transformed_df["user_id"].notna().sum()
                self.logger.info(f"User ID mapping: {mapped}/{total} interactions mapped")
                
                if mapped < total:
                    self.status.add_warning(
                        f"{total - mapped} interactions have no matching user and will be skipped"
                    )
            
            if self.enable_quality_checks:
                self._run_quality_checks(transformed_df, "Interaction")
            self.status.mark_stage_end(PipelineStage.TRANSFORM)
            
            self.status.mark_stage_start(PipelineStage.LOAD)
            loaded_count = self._load_interaction_data(transformed_df, mode)
            self.status.records_loaded = loaded_count
            self.status.mark_stage_end(PipelineStage.LOAD)
            
            self.status.finalize(success=True)
            return self.status.get_summary()
        
        except Exception as e:
            self.status.add_error(f"Pipeline failed: {str(e)}")
            self.status.add_error(traceback.format_exc())
            self.status.finalize(success=False)
            raise

    def run_full_pipeline(
        self,
        user_file: Path,
        interaction_file: Path,
        usage_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        self.logger.info("Starting full pipeline execution")
        results = {
            "user_pipeline": None,
            "interaction_pipeline": None,
            "usage_pipeline": None,
            "overall_status": "success"
        }
        try:
            # ✅ 1단계: User 파이프라인 먼저 실행 (필수)
            self.logger.info("=" * 60)
            self.logger.info("STEP 1: Running user pipeline...")
            self.logger.info("=" * 60)
            results["user_pipeline"] = self.run_user_pipeline(user_file, mode="upsert")
            
            # ✅ 2단계: Interaction 파이프라인 실행 (User ID 매핑 포함)
            self.logger.info("\n" + "=" * 60)
            self.logger.info("STEP 2: Running interaction pipeline with user mapping...")
            self.logger.info("=" * 60)
            results["interaction_pipeline"] = self.run_interaction_pipeline(interaction_file, mode="insert")
            
            # ✅ 3단계: Usage 파이프라인 (선택)
            if usage_file:
                self.logger.info("\n" + "=" * 60)
                self.logger.info("STEP 3: Running usage pipeline...")
                self.logger.info("=" * 60)
                results["usage_pipeline"] = self.run_interaction_pipeline(usage_file, mode="insert")
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("✅ Full pipeline completed successfully")
            self.logger.info("=" * 60)
        
        except Exception as e:
            results["overall_status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Full pipeline failed: {e}")
            raise
        
        return results

    # ==================== PRIVATE METHODS ====================

    def _extract_user_data(self, file_path: Path) -> List[Dict[str, Any]]:
        self.logger.info(f"Extracting user data from {file_path}")
        extractor = FileExtractor(file_path, file_format="jsonl")
        if not extractor.validate_source():
            raise FileNotFoundError(f"User data file not found: {file_path}")
        
        data = extractor.extract()
        self.logger.info(f"Extracted {len(data)} user records")
        return data

    def _extract_interaction_data(self, file_path: Path) -> List[Dict[str, Any]]:
        self.logger.info(f"Extracting interaction data from {file_path}")
        extractor = FileExtractor(file_path, file_format="jsonl")
        if not extractor.validate_source():
            raise FileNotFoundError(f"Interaction data file not found: {file_path}")
        data = extractor.extract()
        self.logger.info(f"Extracted {len(data)} interaction records")
        return data

    def _validate_user_data(self, data: List[Dict[str, Any]]) -> None:
        """Validate minimal user data structure (flattened version)."""
        self.logger.info("Validating user data schema")
        validator = SchemaValidator([
            ValidationRule("distinct_id", DataType.STRING, required=True),
        ])
        result = validator.validate(data)
        if not result.is_valid:
            error_msg = f"User data validation failed: {len(result.errors)} errors"
            self.status.add_error(error_msg)
            for error in result.errors[:10]:
                self.logger.error(f"  - {error}")
            raise ValueError(error_msg)
        self.logger.info(f"User data validation passed: {result.valid_records}/{result.total_records} valid")

    def _validate_interaction_data(self, data: List[Dict[str, Any]]) -> None:
        """✅ 수정: properties 필드는 flatten되므로 validation에서 제외"""
        self.logger.info("Validating interaction data schema")
        validator = SchemaValidator([
            ValidationRule("event_name", DataType.STRING, required=True),
            ValidationRule("distinct_id", DataType.STRING, required=True),
            ValidationRule("time", DataType.INTEGER, required=True),
        ])
        result = validator.validate(data)
        if not result.is_valid:
            error_msg = f"Interaction data validation failed: {len(result.errors)} errors"
            self.status.add_error(error_msg)
            for error in result.errors[:10]:
                self.logger.error(f"  - {error}")
            raise ValueError(error_msg)
        self.logger.info(f"Interaction data validation passed: {result.valid_records}/{result.total_records} valid")

    def _transform_user_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        self.logger.info("Transforming user data")
        df = self.user_transformer.transform(raw_data)
        self.logger.info(f"Transformed to {len(df)} rows, {len(df.columns)} columns")
        return df

    def _transform_interaction_data(
        self, 
        raw_data: List[Dict[str, Any]], 
        load_user_mapping: bool = True
    ) -> pd.DataFrame:
        """✅ User ID 매핑 옵션 추가"""
        self.logger.info("Transforming interaction data")
        df = self.interaction_transformer.transform(raw_data, load_user_mapping=load_user_mapping)
        self.logger.info(f"Transformed to {len(df)} rows, {len(df.columns)} columns")
        return df

    def _run_quality_checks(self, df: pd.DataFrame, data_type: str) -> None:
        self.logger.info(f"Running quality checks on {data_type} data")
        report = self.quality_validator.validate(df)
        for check in report.checks:
            if not check.passed:
                if check.severity == "error":
                    self.status.add_error(f"{data_type} quality: {check.message}")
                else:
                    self.status.add_warning(f"{data_type} quality: {check.message}")
        summary = report.get_summary()
        self.logger.info(
            f"Quality checks: {summary['passed_checks']}/{summary['total_checks']} passed"
        )

    def _load_user_data(self, df: pd.DataFrame, mode: str) -> int:
        self.logger.info(f"Loading {len(df)} user records (mode={mode})")
        count = self.loader.load_dataframe(
            User,
            df,
            mode=mode,
            conflict_columns=["distinct_id"],
            batch_size=self.batch_size,
            validate_schema=True
        )
        self.logger.info(f"Loaded {count} user records")
        return count

    def _load_interaction_data(self, df: pd.DataFrame, mode: str) -> int:
        """✅ skip_null_foreign_keys=True, skip_duplicates=True 추가"""
        self.logger.info(f"Loading {len(df)} interaction records (mode={mode})")
        count = self.loader.load_dataframe(
            Interaction,
            df,
            mode=mode,
            conflict_columns=["insert_id"] if mode == "upsert" else None,
            batch_size=self.batch_size,
            validate_schema=True,
            skip_null_foreign_keys=True,  # ✅ NULL user_id 자동 스킵
            skip_duplicates=True  # ✅ 중복 insert_id 자동 스킵
        )
        self.logger.info(f"Loaded {count} interaction records")
        return count


if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    BASE_DIR = Path("/home/kj1004mj/bookend-recommendation/data/raw")
    CLIENT_DATA = BASE_DIR / "client_data.json"
    EVENT_DATA  = BASE_DIR / "event_data.json"
    USAGE_DATA  = BASE_DIR / "usage_data.json"
    
    orchestrator = PipelineOrchestrator(
        enable_validation=True,
        enable_quality_checks=True,
        batch_size=1000
    )
    
    try:
        result = orchestrator.run_full_pipeline(
            user_file=CLIENT_DATA,
            interaction_file=EVENT_DATA,
            usage_file=USAGE_DATA
        )
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"\nOverall Status: {result['overall_status']}")
        
        if result.get('user_pipeline'):
            u = result['user_pipeline']
            print(f"\nUser Pipeline:")
            print(f"  - Processed: {u['records_processed']}")
            print(f"  - Loaded: {u['records_loaded']}")
            print(f"  - Duration: {u['duration_seconds']:.2f}s")
        
        if result.get('interaction_pipeline'):
            i = result['interaction_pipeline']
            print(f"\nInteraction Pipeline:")
            print(f"  - Processed: {i['records_processed']}")
            print(f"  - Loaded: {i['records_loaded']}")
            print(f"  - Duration: {i['duration_seconds']:.2f}s")
            if i.get('warning_count', 0) > 0:
                print(f"  - Warnings: {i['warning_count']}")
        
        if result.get('usage_pipeline'):
            g = result['usage_pipeline']
            print(f"\nUsage Pipeline:")
            print(f"  - Processed: {g['records_processed']}")
            print(f"  - Loaded: {g['records_loaded']}")
            print(f"  - Duration: {g['duration_seconds']:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")