from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class QualityCheck:
    def __init__(self, check_name: str, passed: bool, message: str, severity: str = "warning", details: Optional[Dict[str, Any]] = None):
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now()

    def __repr__(self) -> str:
        return f"<QualityCheck [{'PASS' if self.passed else 'FAIL'}] {self.check_name}: {self.message}>"


class QualityReport:
    def __init__(self):
        self.checks: List[QualityCheck] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

    def add_check(self, check: QualityCheck):
        self.checks.append(check)

    def finalize(self):
        self.end_time = datetime.now()

    @property
    def overall_passed(self) -> bool:
        return all(check.passed or check.severity != "error" for check in self.checks)

    def get_summary(self) -> Dict[str, Any]:
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        failed = [c for c in self.checks if not c.passed]
        return {
            "overall_passed": self.overall_passed,
            "total_checks": len(self.checks),
            "passed_checks": sum(c.passed for c in self.checks),
            "failed_checks": len(failed),
            "errors": sum(c.severity == "error" for c in failed),
            "warnings": sum(c.severity == "warning" for c in failed),
            "duration_seconds": duration
        }


class QualityValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(self, data: pd.DataFrame) -> QualityReport:
        report = QualityReport()
        if data.empty:
            report.add_check(QualityCheck("empty_dataset", False, "Dataset is empty", "error"))
            report.finalize()
            return report

        self._check_completeness(data, report)
        self._check_uniqueness(data, report)
        self._check_consistency(data, report)
        self._check_outliers(data, report)
        self._check_distributions(data, report)
        self._check_temporal_validity(data, report)

        report.finalize()
        self.logger.info(f"Quality validation finished with {len(report.checks)} checks.")
        return report

    def _check_completeness(self, data: pd.DataFrame, report: QualityReport):
        total_cells = data.size
        missing_cells = int(data.isna().sum().sum())
        completeness_rate = 1 - (missing_cells / total_cells) if total_cells else 0
        severity = "error" if completeness_rate < 0.8 else "warning"
        report.add_check(QualityCheck("data_completeness", completeness_rate >= 0.95, f"Completeness {completeness_rate:.2%}", severity))

        for col in data.columns:
            rate = data[col].isna().mean()
            if rate > 0.1:
                report.add_check(QualityCheck(f"column_completeness_{col}", False, f"{col}: {rate:.1%} missing", "warning"))

    def _check_uniqueness(self, data: pd.DataFrame, report: QualityReport):
        dup_count = data.duplicated().sum()
        report.add_check(QualityCheck("duplicate_rows", dup_count == 0, f"{dup_count} duplicate rows", "warning"))
        for col in [c for c in data.columns if 'id' in c.lower()]:
            uniq_rate = data[col].nunique() / data[col].notna().sum() if data[col].notna().sum() > 0 else 0
            if uniq_rate < 0.95:
                report.add_check(QualityCheck(f"uniqueness_{col}", False, f"Uniqueness {uniq_rate:.2%}", "warning"))

    def _check_consistency(self, data: pd.DataFrame, report: QualityReport):
        num_cols = data.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            neg = (data[col] < 0).sum() if any(k in col.lower() for k in ['count', 'age', 'duration']) else 0
            if neg:
                report.add_check(QualityCheck(f"{col}_negative_values", False, f"{neg} negative values", "error"))

    def _check_outliers(self, data: pd.DataFrame, report: QualityReport):
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].notna().sum() < 10:
                continue
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            rate = ((data[col] < low) | (data[col] > high)).mean()
            if rate > 0.05:
                report.add_check(QualityCheck(f"outliers_{col}", False, f"{rate:.2%} outliers", "info"))

    def _check_distributions(self, data: pd.DataFrame, report: QualityReport):
        cat_cols = data.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            counts = data[col].value_counts(dropna=True)
            if not counts.empty and counts.iloc[0] / len(data) > 0.95:
                report.add_check(QualityCheck(f"{col}_imbalance", False, f"{counts.index[0]} dominates >95%", "info"))

    def _check_temporal_validity(self, data: pd.DataFrame, report: QualityReport):
        date_cols = [c for c in data.columns if any(x in c.lower() for x in ["date", "time", "timestamp"])]
        now = datetime.now()
        for col in date_cols:
            try:
                s = pd.to_datetime(data[col], errors="coerce")
                if s.isna().all():
                    continue
                if (s > now).sum() > 0:
                    report.add_check(QualityCheck(f"{col}_future_dates", False, f"Contains future dates", "warning"))
                if (s < now - timedelta(days=3650)).sum() > 0:
                    report.add_check(QualityCheck(f"{col}_old_dates", False, f"Contains very old dates", "info"))
            except Exception as e:
                self.logger.warning(f"Date parse error in {col}: {e}")
