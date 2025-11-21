from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from datetime import datetime
import re

from src.database.postgres_singleton import get_postgres

logger = logging.getLogger(__name__)


class InteractionTransformer:
    """Transform raw event data into structured Interaction records."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db = get_postgres()
        self._user_id_cache: Dict[str, int] = {}

    def transform(self, raw_data: List[Dict[str, Any]], load_user_mapping: bool = True) -> pd.DataFrame:
        """Transform raw event data to structured format."""
        if not raw_data:
            self.logger.warning("No data to transform")
            return pd.DataFrame()

        self.logger.info(f"Transforming {len(raw_data)} interaction records")

        df = pd.DataFrame(raw_data)
        self.logger.info(f"DataFrame created with {len(df.columns)} columns")

        df = self._remove_suffix_columns_aggressive(df)
        df = self._normalize_fields(df)
        df = self._parse_timestamps(df)
        df = self._extract_event_features(df)
        df = self._categorize_events(df)
        df = self._handle_missing_values(df)
        df = self._add_derived_features(df)

        if load_user_mapping:
            df = self._map_user_ids(df)

        df = self._apply_length_constraints(df)
        df = self._select_final_columns(df)

        self.logger.info(f"Transformation complete: {len(df)} records, {len(df.columns)} columns")
        return df

    # ---------------------------------------------------------------
    # 🔥 여기부터 `_map_user_ids`만 개선된 버전 (나머지 코드 전부 동일)
    # ---------------------------------------------------------------
    def _map_user_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        improved version:
        - DB distinct_id, DF distinct_id 모두 문자열로 강제
        - 캐시 활용
        - 매핑 실패 로그 정교화
        - (중요) users.user_id는 절대 사용하지 않고 users.id만 사용
        """
        if "distinct_id" not in df.columns:
            self.logger.error("distinct_id column not found for user mapping")
            df["user_id"] = None
            return df

        try:
            query = "SELECT id, distinct_id FROM users WHERE distinct_id IS NOT NULL"
            user_rows = self.db.execute_query(query)

            if not user_rows:
                self.logger.error("User table empty. Cannot map distinct_id → user_id")
                df["user_id"] = None
                return df

            # DB mapping (string → int)
            user_map = {
                str(row["distinct_id"]).strip(): row["id"]
                for row in user_rows
                if row["distinct_id"] is not None
            }

            self._user_id_cache = user_map

            # DF mapping 대상 정규화
            df["distinct_id"] = df["distinct_id"].astype(str).str.strip()

            # 매핑 수행
            df["user_id"] = df["distinct_id"].map(self._user_id_cache).astype("Int64")

            missing_count = df["user_id"].isna().sum()
            if missing_count > 0:
                sample = df[df["user_id"].isna()]["distinct_id"].unique()[:10]
                self.logger.warning(
                    f"{missing_count} rows could not map to user_id. sample={sample}"
                )

            mapped_count = df["user_id"].notna().sum()
            self.logger.info(
                f"Mapped user_id for {mapped_count}/{len(df)} interaction rows"
            )

        except Exception as e:
            self.logger.error(f"Failed to map user IDs: {e}", exc_info=True)
            df["user_id"] = None

        return df
    # ---------------------------------------------------------------
    # 🔥 여기까지 `_map_user_ids`만 수정됨
    # ---------------------------------------------------------------

    def _remove_suffix_columns_aggressive(self, df: pd.DataFrame) -> pd.DataFrame:
        original_cols = df.columns.tolist()
        suffix_pattern = re.compile(r'(_m\d+|\.\d+)$')

        suffix_cols = [col for col in original_cols if suffix_pattern.search(str(col))]

        if not suffix_cols:
            self.logger.info("No suffix columns detected")
            return df

        self.logger.warning(f"Found {len(suffix_cols)} columns with suffix pattern")

        base_to_cols = {}
        for col in original_cols:
            base_name = suffix_pattern.sub('', str(col))
            base_to_cols.setdefault(base_name, []).append(col)

        selected_cols = []
        for base_name, cols in base_to_cols.items():
            if len(cols) == 1:
                selected_cols.append(cols[0])
            else:
                no_suffix = [c for c in cols if not suffix_pattern.search(str(c))]
                if no_suffix:
                    selected_cols.append(no_suffix[0])
                else:
                    sorted_cols = sorted(cols, key=lambda x: self._extract_suffix_number(x))
                    selected_cols.append(sorted_cols[0])

        new_df = pd.DataFrame()
        for col in selected_cols:
            base_name = suffix_pattern.sub('', str(col))
            new_df[base_name] = df[col]

        self.logger.info(f"Columns cleaned: {len(original_cols)} → {len(new_df.columns)}")

        if new_df.columns.duplicated().any():
            new_df = new_df.loc[:, ~new_df.columns.duplicated()]

        return new_df

    def _extract_suffix_number(self, col_name: str) -> int:
        match = re.search(r'(_m|\.)(\d+)$', str(col_name))
        return int(match.group(2)) if match else 0

    def _apply_length_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        LENGTH_LIMITS = {
            'event_name': 200, 'distinct_id': 255, 'insert_id': 200,
            'device_id': 255, 'browser': 200, 'os': 200,
            'event_type': 50, 'interaction_category': 50,
            'day_of_week': 20, 'llm_name': 200, 'llm_provider': 50,
            'llm_version': 100, 'field': 100, 'tone': 50,
            'maintenance': 50, 'target_language': 50
        }

        total_truncated = 0

        for col, max_len in LENGTH_LIMITS.items():
            if col not in df.columns:
                continue

            mask = df[col].notna()
            if not mask.any():
                continue

            df.loc[mask, col] = df.loc[mask, col].astype(str)
            lengths = df.loc[mask, col].str.len()
            over_limit = lengths > max_len

            if over_limit.any():
                count = over_limit.sum()
                total_truncated += count

                self.logger.warning(
                    f"Column '{col}': {count} values exceed {max_len} chars. Truncating..."
                )

                df.loc[mask & over_limit, col] = df.loc[mask & over_limit, col].str[:max_len]

        if total_truncated > 0:
            self.logger.warning(
                f"⚠️  Total {total_truncated} values truncated to meet DB constraints"
            )

        return df

    def _normalize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.lower()

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', 'null', 'NULL', ''], None)

        for col in ["event_name", "distinct_id"]:
            if col not in df.columns:
                df[col] = None
                self.logger.warning(f"Missing required column '{col}', added with None values")

        return df

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        if "time" in df.columns:
            df["event_time"] = pd.to_datetime(df["time"], unit='s', errors="coerce", utc=True)

        for col in ["created_at", "updated_at", "timestamp"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        if "event_time" not in df.columns and "time" in df.columns:
            df["event_time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

        return df

    def _extract_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "event_name" in df.columns:
            df["event_type"] = df["event_name"].apply(self._map_event_type)
        return df

    def _map_event_type(self, event_name: Optional[str]) -> str:
        if not event_name or pd.isna(event_name):
            return "unknown"

        e = str(event_name).lower()

        if "paraphrasing" in e or "run" in e:
            return "action"
        elif "view" in e or "pageview" in e:
            return "view"
        elif "click" in e or "selected" in e:
            return "click"
        elif "copy" in e:
            return "copy"
        elif "open" in e:
            return "open"
        elif "purchase" in e or "buy" in e:
            return "purchase"
        elif "scroll" in e:
            return "scroll"
        elif "login" in e or "auth" in e:
            return "auth"
        else:
            return "other"

    def _categorize_events(self, df: pd.DataFrame) -> pd.DataFrame:
        if "event_type" not in df.columns:
            return df

        def categorize(evt):
            if pd.isna(evt):
                return "other"

            evt = str(evt).lower()

            if evt in ["view", "scroll", "open"]:
                return "engagement"
            elif evt in ["click", "copy", "action"]:
                return "interaction"
            elif evt == "purchase":
                return "conversion"
            elif evt == "auth":
                return "session"
            else:
                return "other"

        df["interaction_category"] = df["event_type"].apply(categorize)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if "distinct_id" in df.columns:
            df["distinct_id"] = df["distinct_id"].replace(["", "null", "none", "None"], None)
            before = len(df)
            df = df[df["distinct_id"].notna()]
            if len(df) < before:
                self.logger.warning(f"Removed {before - len(df)} rows with null distinct_id")

        if "event_time" in df.columns:
            before = len(df)
            df = df[df["event_time"].notna()]
            if len(df) < before:
                self.logger.warning(f"Removed {before - len(df)} rows with null event_time")

        if "insert_id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["insert_id"], keep="first")
            if len(df) < before:
                self.logger.warning(f"⚠️  Removed {before - len(df)} duplicate insert_ids within batch")

        df = df.fillna({"event_type": "unknown", "interaction_category": "other"})

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "event_time" in df.columns:
            df["hour_of_day"] = df["event_time"].dt.hour
            df["day_of_week"] = df["event_time"].dt.day_name()
            df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

        if "session_start" in df.columns and "session_end" in df.columns:
            df["session_duration"] = (
                pd.to_datetime(df["session_end"], errors="coerce", utc=True)
                - pd.to_datetime(df["session_start"], errors="coerce", utc=True)
            ).dt.total_seconds()

        df["processed_at"] = pd.Timestamp.now(tz="UTC")

        return df

    def _select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_columns = [
            'event_name', 'distinct_id', 'insert_id', 'device_id', 'user_id',
            'event_time', 'event_type', 'interaction_category',
            'hour_of_day', 'day_of_week', 'is_weekend',
            'tone', 'maintenance', 'field', 'target_language',
            'response_time_ms', 'input_sentence_length',
            'llm_provider', 'llm_name', 'llm_version',
            'position', 'trigger',
            'browser', 'os', 'current_url',
            'processed_at'
        ]

        available = [c for c in expected_columns if c in df.columns]

        missing = [c for c in ['event_name', 'distinct_id', 'event_time'] if c not in available]
        if missing:
            raise ValueError(f"Required columns missing: {missing}")

        self.logger.info(f"Final columns selected: {len(available)}/{len(expected_columns)}")

        tone_cols = ['tone', 'maintenance', 'field', 'target_language']
        existing_tone_cols = [c for c in tone_cols if c in available]
        if existing_tone_cols:
            self.logger.info(f"Included tone columns: {existing_tone_cols}")
        else:
            self.logger.warning("No tone columns found")

        return df[available]
