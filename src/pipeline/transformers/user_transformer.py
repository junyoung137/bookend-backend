from typing import List, Dict, Any, Optional, Set
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import re

logger = logging.getLogger(__name__)


class UserTransformer:
    """Transform raw user/client data into structured User records."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def transform(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform raw user data to structured format."""
        if not raw_data:
            self.logger.warning("No data to transform")
            return pd.DataFrame()

        self.logger.info(f"Transforming {len(raw_data)} user records")
        
        # DataFrame 생성
        df = pd.DataFrame(raw_data)
        
        self.logger.info(f"DataFrame created with {len(df.columns)} columns")
        self.logger.debug(f"Sample columns: {list(df.columns)[:20]}")
        
        # ✅ 즉시 suffix 컬럼 제거 (가장 먼저 실행)
        df = self._remove_suffix_columns_aggressive(df)

        # 변환 파이프라인
        df = self._normalize_fields(df)
        df = self._parse_datetime_fields(df)
        df = self._extract_account_type(df)
        df = self._handle_missing_values(df)
        df = self._add_derived_features(df)
        
        # ✅ DB 스키마 길이 제한 적용
        df = self._apply_length_constraints(df)
        
        df = self._select_final_columns(df)

        self.logger.info(f"Transformation complete: {len(df)} records, {len(df.columns)} columns")
        return df

    def _remove_suffix_columns_aggressive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ 강화된 suffix 제거: _mNNN, .N 패턴 완전 제거
        
        전략:
        1. 모든 suffix 패턴 컬럼 식별
        2. base_name별로 그룹화
        3. suffix 없는 컬럼 우선, 없으면 가장 낮은 번호 선택
        """
        original_cols = df.columns.tolist()
        suffix_pattern = re.compile(r'(_m\d+|\.\d+)$')
        
        # suffix 컬럼 개수 확인
        suffix_cols = [col for col in original_cols if suffix_pattern.search(str(col))]
        
        if not suffix_cols:
            self.logger.info("No suffix columns detected")
            return df
        
        self.logger.warning(f"Found {len(suffix_cols)} columns with suffix pattern")
        self.logger.debug(f"Suffix columns sample: {suffix_cols[:10]}")
        
        # base_name별로 그룹화
        base_to_cols = {}
        for col in original_cols:
            base_name = suffix_pattern.sub('', str(col))
            if base_name not in base_to_cols:
                base_to_cols[base_name] = []
            base_to_cols[base_name].append(col)
        
        # 최종 선택할 컬럼들
        selected_cols = []
        for base_name, cols in base_to_cols.items():
            if len(cols) == 1:
                # 중복 없음
                selected_cols.append(cols[0])
            else:
                # 중복 있음: suffix 없는 것 우선 선택
                no_suffix = [c for c in cols if not suffix_pattern.search(str(c))]
                if no_suffix:
                    selected_cols.append(no_suffix[0])
                    self.logger.debug(f"Selected '{no_suffix[0]}' over {len(cols)-1} variants")
                else:
                    # 모두 suffix → 가장 낮은 번호 선택
                    sorted_cols = sorted(cols, key=lambda x: self._extract_suffix_number(x))
                    selected_cols.append(sorted_cols[0])
                    self.logger.debug(f"Selected '{sorted_cols[0]}' (lowest suffix) from {cols}")
        
        # DataFrame 재구성
        new_df = pd.DataFrame()
        for col in selected_cols:
            # base_name으로 컬럼명 정규화
            base_name = suffix_pattern.sub('', str(col))
            new_df[base_name] = df[col]
        
        self.logger.info(f"Columns cleaned: {len(original_cols)} → {len(new_df.columns)}")
        
        # 최종 중복 확인
        if new_df.columns.duplicated().any():
            self.logger.error("Duplicate columns still exist after cleaning!")
            new_df = new_df.loc[:, ~new_df.columns.duplicated()]
        
        return new_df

    def _extract_suffix_number(self, col_name: str) -> int:
        """suffix 번호 추출 (_m123 → 123, .5 → 5)"""
        match = re.search(r'(_m|\.)(\d+)$', str(col_name))
        return int(match.group(2)) if match else 0

    def _apply_length_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ DB 스키마 길이 제한 적용
        VARCHAR 길이를 초과하는 값은 자르고 경고 로그 출력
        """
        LENGTH_LIMITS = {
            'distinct_id': 255, 'user_id': 255, 'email': 255, 'name': 255,
            'browser': 100, 'browser_version': 50, 'os': 100,
            'city': 100, 'region': 100, 'country_code': 10, 'timezone': 100,
            'initial_referrer': 500, 'initial_referring_domain': 255,
            'initial_utm_source': 100, 'initial_utm_medium': 100,
            'initial_utm_campaign': 100, 'account_type': 50,
            'browser_family': 100, 'device_category': 100,
            'referrer_domain_clean': 255
        }
        
        total_truncated = 0
        
        for col, max_len in LENGTH_LIMITS.items():
            if col not in df.columns:
                continue
            
            mask = df[col].notna()
            if not mask.any():
                continue
            
            # 문자열로 변환 후 길이 체크
            df.loc[mask, col] = df.loc[mask, col].astype(str)
            lengths = df.loc[mask, col].str.len()
            over_limit = lengths > max_len
            
            if over_limit.any():
                count = over_limit.sum()
                total_truncated += count
                max_found = lengths.max()
                
                self.logger.warning(
                    f"Column '{col}': {count} values exceed {max_len} chars "
                    f"(max found: {max_found}). Truncating..."
                )
                
                # 값 자르기
                df.loc[mask & over_limit, col] = df.loc[mask & over_limit, col].str[:max_len]
        
        if total_truncated > 0:
            self.logger.warning(
                f"⚠️  Total {total_truncated} values truncated to meet DB constraints"
            )
        
        return df

    def _normalize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize field names and values."""
        df.columns = df.columns.str.strip().str.lower()

        # Boolean 컬럼 정규화
        bool_columns = [
            "is_logged_in", "is_apple_account", "is_email_account",
            "is_google_account", "is_kakao_account", "is_naver_account",
        ]
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False)
                df[col] = df[col].replace({
                    'true': True, 'True': True, 'TRUE': True,
                    'false': False, 'False': False, 'FALSE': False,
                    '1': True, 1: True, '0': False, 0: False,
                    'yes': True, 'no': False,
                    'nan': False, 'None': False, None: False
                }).astype(bool)

        # Object 타입 컬럼 정규화
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', 'null', 'NULL', ''], None)

        return df

    def _parse_datetime_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """날짜/시간 필드 파싱"""
        for col in ["last_seen", "created_at", "updated_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        return df

    def _extract_account_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """계정 타입 추출"""
        account_flags = {
            "is_kakao_account": "kakao",
            "is_naver_account": "naver",
            "is_google_account": "google",
            "is_apple_account": "apple",
            "is_email_account": "email",
        }

        def get_account_type(row):
            for flag, acc in account_flags.items():
                if flag in row and row[flag] is True:
                    return acc
            return None

        if any(flag in df.columns for flag in account_flags.keys()):
            df["account_type"] = df.apply(get_account_type, axis=1)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        # Boolean 컬럼 기본값
        bool_cols = df.select_dtypes(include=["bool"]).columns
        df[bool_cols] = df[bool_cols].fillna(False)

        # 숫자 컬럼 기본값
        if "role" in df.columns:
            df["role"] = pd.to_numeric(df["role"], errors="coerce").fillna(1).astype(int)
        if "total_sessions" in df.columns:
            df["total_sessions"] = pd.to_numeric(df["total_sessions"], errors="coerce").fillna(0).astype(int)

        # Placeholder 값 제거
        placeholders = ["$direct", "$unknown", "none", "null", ""]
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].replace(placeholders, None)
        
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 특성 추가"""
        # 도메인 정리
        if "initial_referring_domain" in df.columns:
            df["referrer_domain_clean"] = df["initial_referring_domain"].apply(
                lambda x: self._clean_domain(x) if pd.notna(x) else None
            )
        
        # 브라우저 패밀리
        if "browser" in df.columns:
            df["browser_family"] = df["browser"].apply(
                lambda x: self._get_browser_family(x) if pd.notna(x) else None
            )
        
        # 디바이스 카테고리
        if "os" in df.columns:
            df["device_category"] = df["os"].apply(
                lambda x: self._categorize_device(x) if pd.notna(x) else None
            )
        
        # 활동성 지표
        if "last_seen" in df.columns:
            now = pd.Timestamp.now(tz="UTC")
            df["days_since_last_seen"] = (now - df["last_seen"]).dt.days
            df["is_active"] = df["days_since_last_seen"] <= 30
        
        return df

    def _select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """최종 컬럼 선택"""
        expected_columns = [
            'distinct_id', 'email', 'name', 'browser', 'browser_version', 'browser_family',
            'os', 'device_category', 'city', 'region', 'country_code', 'timezone',
            'initial_referrer', 'initial_referring_domain', 'referrer_domain_clean',
            'initial_utm_source', 'initial_utm_medium', 'initial_utm_campaign',
            'is_logged_in', 'account_type', 'role', 'last_seen', 'total_sessions',
            'days_since_last_seen', 'is_active'
        ]
        
        available_columns = [col for col in expected_columns if col in df.columns]
        
        if 'distinct_id' not in available_columns:
            raise ValueError("distinct_id column is required but not found")
        
        return df[available_columns]

    @staticmethod
    def _clean_domain(domain: str) -> Optional[str]:
        """도메인 정리"""
        if not domain or str(domain).lower() in ["$direct", "$unknown", "none", "nan"]:
            return None
        domain = str(domain).lower().replace("www.", "")
        parts = domain.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else domain

    @staticmethod
    def _get_browser_family(browser: str) -> Optional[str]:
        """브라우저 패밀리 분류"""
        if not browser:
            return None
        b = str(browser).lower()
        if "chrome" in b:
            return "chrome"
        elif "firefox" in b or "mozilla" in b:
            return "firefox"
        elif "safari" in b:
            return "safari"
        elif "edge" in b:
            return "edge"
        elif "whale" in b:
            return "whale"
        return "other"

    @staticmethod
    def _categorize_device(os: str) -> Optional[str]:
        """디바이스 카테고리 분류"""
        if not os:
            return None
        o = str(os).lower()
        if "mac os x" in o or "darwin" in o:
            return "mac"
        elif "windows" in o or "linux" in o or "chrome os" in o:
            return "desktop"
        elif "ios" in o or "iphone" in o or "ipad" in o:
            return "ios"
        elif "android" in o:
            return "android"
        return "unknown"