import logging
from typing import Any, Dict, List, Optional, Type, Union, Set

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect

from src.database.postgres_singleton import get_postgres
from src.database.models import Base

logger = logging.getLogger(__name__)


class PostgresLoader:
    """
    Safe and robust PostgreSQL loader with upsert/insert capabilities.
    """

    def __init__(self):
        self.db = get_postgres()
        self.logger = logger

    # =========================================================
    # Public API
    # =========================================================
    def load_dataframe(
        self,
        model: Type[Base],
        df: Union[pd.DataFrame, List[Dict[str, Any]]],
        mode: str = "upsert",
        conflict_columns: Optional[List[str]] = None,
        batch_size: int = 1000,
        validate_schema: bool = True,
        skip_null_foreign_keys: bool = True,
        skip_duplicates: bool = True,
    ) -> int:
        """
        Load a pandas DataFrame into PostgreSQL using INSERT or UPSERT.
        
        Args:
            model: SQLAlchemy model class
            df: DataFrame or list of dictionaries to load
            mode: 'insert' or 'upsert'
            conflict_columns: Columns to check for conflicts in upsert mode
            batch_size: Number of records per batch
            validate_schema: Validate schema before loading
            skip_null_foreign_keys: Skip records with NULL foreign keys
            skip_duplicates: Skip duplicate records (INSERT mode only)
        
        Returns:
            Number of records loaded
        """
        self.logger.debug(f"Loading DataFrame to {model.__tablename__} (mode={mode})")

        if isinstance(df, pd.DataFrame):
            records = df.to_dict(orient="records")
        else:
            records = df

        if not records:
            self.logger.warning(f"No records to load into {model.__tablename__}")
            return 0

        # Optional schema validation
        if validate_schema:
            self._validate_dataframe_schema(model, records)

        prepared_records = self._prepare_dataframe(model, records)

        # ✅ Foreign key NULL 체크 (Interaction 테이블의 경우)
        if skip_null_foreign_keys and model.__tablename__ == "interactions":
            prepared_records = self._filter_null_foreign_keys(prepared_records, "user_id")

        # ✅ 중복 제거 (INSERT 모드에서만)
        if skip_duplicates and mode == "insert":
            prepared_records = self._filter_duplicate_records(model, prepared_records)

        if mode == "insert":
            return self._load_insert(model, prepared_records, batch_size)
        elif mode == "upsert":
            return self._load_upsert(model, prepared_records, conflict_columns, batch_size)
        else:
            raise ValueError(f"Unsupported load mode: {mode}")

    # =========================================================
    # Internal Helpers
    # =========================================================
    def _filter_null_foreign_keys(self, records: List[Dict[str, Any]], fk_column: str) -> List[Dict[str, Any]]:
        """
        ✅ Foreign key가 NULL인 레코드 제거
        """
        original_count = len(records)
        filtered_records = [rec for rec in records if rec.get(fk_column) is not None]
        filtered_count = len(filtered_records)
        
        if filtered_count < original_count:
            skipped = original_count - filtered_count
            self.logger.warning(
                f"⚠️  Skipped {skipped} records with NULL {fk_column} "
                f"({filtered_count}/{original_count} will be loaded)"
            )
        
        return filtered_records

    def _filter_duplicate_records(self, model: Type[Base], records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ✅ 중복 레코드 제거 (unique constraint 기반)
        """
        table_name = model.__tablename__
        
        # Interactions 테이블의 경우 insert_id 체크
        if table_name == "interactions":
            return self._filter_duplicate_insert_ids(records)
        
        # Users 테이블의 경우 distinct_id 체크
        elif table_name == "users":
            return self._filter_duplicate_distinct_ids(records)
        
        # 기타 테이블은 필터링 없이 반환
        return records

    def _filter_duplicate_insert_ids(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """✅ Interactions 테이블의 중복 insert_id 제거"""
        if not records:
            return records
        
        # DB에서 기존 insert_id 가져오기
        existing_ids = self._get_existing_insert_ids()
        self.logger.info(f"Found {len(existing_ids)} existing insert_ids in database")
        
        # 중복 제거
        original_count = len(records)
        filtered_records = [r for r in records if r.get("insert_id") not in existing_ids]
        skipped = original_count - len(filtered_records)
        
        if skipped > 0:
            self.logger.warning(
                f"⚠️  Skipped {skipped} duplicate insert_ids "
                f"({len(filtered_records)}/{original_count} will be loaded)"
            )
        
        return filtered_records

    def _filter_duplicate_distinct_ids(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """✅ Users 테이블의 중복 distinct_id 제거"""
        if not records:
            return records
        
        # DB에서 기존 distinct_id 가져오기
        existing_ids = self._get_existing_distinct_ids()
        self.logger.info(f"Found {len(existing_ids)} existing distinct_ids in database")
        
        # 중복 제거
        original_count = len(records)
        filtered_records = [r for r in records if r.get("distinct_id") not in existing_ids]
        skipped = original_count - len(filtered_records)
        
        if skipped > 0:
            self.logger.warning(
                f"⚠️  Skipped {skipped} duplicate distinct_ids "
                f"({len(filtered_records)}/{original_count} will be loaded)"
            )
        
        return filtered_records

    def _get_existing_insert_ids(self) -> Set[str]:
        """✅ DB에서 기존 insert_id 목록 가져오기"""
        try:
            query = "SELECT insert_id FROM interactions WHERE insert_id IS NOT NULL"
            results = self.db.execute_query(query)
            return {row["insert_id"] for row in results}
        except Exception as e:
            self.logger.error(f"Failed to fetch existing insert_ids: {e}")
            return set()

    def _get_existing_distinct_ids(self) -> Set[str]:
        """✅ DB에서 기존 distinct_id 목록 가져오기"""
        try:
            query = "SELECT distinct_id FROM users WHERE distinct_id IS NOT NULL"
            results = self.db.execute_query(query)
            return {row["distinct_id"] for row in results}
        except Exception as e:
            self.logger.error(f"Failed to fetch existing distinct_ids: {e}")
            return set()

    def _validate_dataframe_schema(self, model: Type[Base], records: List[Dict[str, Any]]) -> None:
        """Ensure DataFrame keys match table schema."""
        mapper = inspect(model)
        valid_columns = {column.name for column in mapper.columns}
        record_keys = set(records[0].keys())

        invalid_cols = record_keys - valid_columns
        if invalid_cols:
            self.logger.warning(f"Ignoring invalid columns: {invalid_cols}")
            for rec in records:
                for col in invalid_cols:
                    rec.pop(col, None)
        else:
            self.logger.debug(f"Schema validation passed for {model.__tablename__}")

    def _prepare_dataframe(self, model: Type[Base], records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out non-table columns for clean insert/upsert."""
        mapper = inspect(model)
        valid_columns = {column.name for column in mapper.columns}

        cleaned = []
        for rec in records:
            filtered = {k: v for k, v in rec.items() if k in valid_columns}
            
            # ✅ pandas NaN/NaT를 None으로 변환
            for key, value in filtered.items():
                if pd.isna(value):
                    filtered[key] = None
            
            cleaned.append(filtered)

        self.logger.debug(
            f"Prepared DataFrame for {model.__tablename__}: "
            f"{len(cleaned)} records, {len(valid_columns)} valid columns"
        )
        return cleaned

    # =========================================================
    # Insert Logic
    # =========================================================
    def _load_insert(self, model: Type[Base], records: List[Dict[str, Any]], batch_size: int) -> int:
        """Insert-only mode with transaction safety."""
        if not records:
            self.logger.warning(f"No records to insert into {model.__tablename__}")
            return 0
        
        total_inserted = 0
        
        with self.db.transaction() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                session.bulk_insert_mappings(model, batch)
                total_inserted += len(batch)
        
        self.logger.info(f"✅ Inserted {total_inserted} rows into {model.__tablename__}")
        return total_inserted

    # =========================================================
    # Upsert Logic (Conflict Safe)
    # =========================================================
    def _load_upsert(
        self,
        model: Type[Base],
        records: List[Dict[str, Any]],
        conflict_columns: Optional[List[str]],
        batch_size: int,
    ) -> int:
        """UPSERT (ON CONFLICT DO UPDATE) implementation."""
        if not records:
            return 0

        table = model.__table__
        conflict_cols = conflict_columns or ["distinct_id"]
        insert_keys = list(records[0].keys())
        update_cols = [c for c in insert_keys if c not in conflict_cols]

        total_upserted = 0

        with self.db.transaction() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                stmt = pg_insert(table).values(batch)

                if not update_cols:
                    # No columns to update → DO NOTHING
                    stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
                    self.logger.debug(
                        f"No updatable columns for {table.name}. Using DO NOTHING (conflict_cols={conflict_cols})"
                    )
                else:
                    update_mapping = {col: getattr(stmt.excluded, col) for col in update_cols}
                    stmt = stmt.on_conflict_do_update(
                        index_elements=conflict_cols,
                        set_=update_mapping,
                    )

                result = session.execute(stmt)
                total_upserted += result.rowcount or len(batch)

        self.logger.info(f"✅ Upsert complete for {model.__tablename__}: {total_upserted} rows processed")
        return total_upserted