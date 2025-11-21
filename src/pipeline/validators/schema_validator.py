from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import re
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DataType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    EMAIL = "email"
    URL = "url"


class ValidationRule:
    """Single validation rule for a field."""
    def __init__(
        self,
        field_name: str,
        data_type: DataType,
        required: bool = True,
        nullable: bool = False,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allowed_values: Optional[List[Any]] = None,
        pattern: Optional[str] = None,
        custom_validator: Optional[Callable[[Any], bool]] = None
    ):
        self.field_name = field_name
        self.data_type = data_type
        self.required = required
        self.nullable = nullable
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = allowed_values
        self.pattern = re.compile(pattern) if pattern else None
        self.custom_validator = custom_validator


class ValidationError:
    """Represents a single validation error."""
    def __init__(self, field_name: str, error_type: str, message: str, record_index: Optional[int] = None):
        self.field_name = field_name
        self.error_type = error_type
        self.message = message
        self.record_index = record_index

    def __repr__(self) -> str:
        idx = f" [record {self.record_index}]" if self.record_index is not None else ""
        return f"<ValidationError{idx} {self.error_type}: {self.field_name} - {self.message}>"


class ValidationResult:
    """Result of schema validation."""
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
        self.total_records: int = 0
        self.valid_records: int = 0

    def add_error(self, field_name: str, error_type: str, message: str, record_index: Optional[int] = None):
        self.errors.append(ValidationError(field_name, error_type, message, record_index))

    def add_warning(self, message: str):
        self.warnings.append(message)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def error_rate(self) -> float:
        return 0.0 if self.total_records == 0 else (self.total_records - self.valid_records) / self.total_records

    def get_summary(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.total_records - self.valid_records,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "error_rate": self.error_rate
        }


class SchemaValidator:
    """Validates data against defined schema rules."""
    def __init__(self, rules: Optional[List[ValidationRule]] = None):
        self.logger = logging.getLogger(__name__)
        self.rules: Dict[str, ValidationRule] = {}
        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: ValidationRule):
        self.rules[rule.field_name] = rule
        self.logger.debug(f"Added validation rule for field: {rule.field_name}")

    def validate(self, data: List[Dict[str, Any]], stop_on_first_error: bool = False) -> ValidationResult:
        result = ValidationResult()
        result.total_records = len(data)

        for idx, record in enumerate(data):
            record_valid = self._validate_record(record, idx, result)
            if record_valid:
                result.valid_records += 1
            elif stop_on_first_error:
                self.logger.warning(f"Validation stopped at record {idx} due to error.")
                break

        self.logger.info(f"Validation complete: {result.valid_records}/{result.total_records} valid")
        return result

    def _validate_record(self, record: Dict[str, Any], record_index: int, result: ValidationResult) -> bool:
        record_valid = True
        for field_name, rule in self.rules.items():
            # 중복 키 제거: 마지막 값만 사용
            value = record.get(field_name, None)

            if value is None:
                if rule.required:
                    result.add_error(field_name, "missing_field", f"Missing required field '{field_name}'", record_index)
                    record_valid = False
                continue

            if value is None and not rule.nullable:
                result.add_error(field_name, "null_value", f"'{field_name}' cannot be null", record_index)
                record_valid = False
                continue

            if not self._validate_type(value, rule.data_type):
                result.add_error(
                    field_name, "type_mismatch",
                    f"Expected {rule.data_type.value}, got {type(value).__name__}", record_index
                )
                record_valid = False
                continue

            if not self._validate_constraints(value, rule, result, record_index):
                record_valid = False

        return record_valid

    def _validate_type(self, value: Any, expected_type: DataType) -> bool:
        try:
            if expected_type == DataType.STRING:
                return isinstance(value, str)
            if expected_type == DataType.INTEGER:
                return isinstance(value, int) and not isinstance(value, bool)
            if expected_type == DataType.FLOAT:
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if expected_type == DataType.BOOLEAN:
                return isinstance(value, bool)
            if expected_type == DataType.DATETIME:
                if isinstance(value, datetime):
                    return True
                datetime.fromisoformat(str(value))
                return True
            if expected_type == DataType.JSON:
                return isinstance(value, (dict, list))
            if expected_type == DataType.EMAIL:
                return isinstance(value, str) and re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", value)
            if expected_type == DataType.URL:
                return isinstance(value, str) and value.startswith(("http://", "https://"))
        except Exception:
            return False
        return False

    def _validate_constraints(self, value: Any, rule: ValidationRule, result: ValidationResult, record_index: int) -> bool:
        valid = True
        if rule.min_value is not None and isinstance(value, (int, float)) and value < rule.min_value:
            result.add_error(rule.field_name, "value_too_small", f"{value} < min {rule.min_value}", record_index)
            valid = False

        if rule.max_value is not None and isinstance(value, (int, float)) and value > rule.max_value:
            result.add_error(rule.field_name, "value_too_large", f"{value} > max {rule.max_value}", record_index)
            valid = False

        if rule.allowed_values and value not in rule.allowed_values:
            result.add_error(rule.field_name, "invalid_value", f"{value} not in {rule.allowed_values}", record_index)
            valid = False

        if rule.pattern and isinstance(value, str) and not rule.pattern.match(value):
            result.add_error(rule.field_name, "pattern_mismatch", f"'{value}' does not match pattern", record_index)
            valid = False

        if rule.custom_validator:
            try:
                if not rule.custom_validator(value):
                    result.add_error(rule.field_name, "custom_validation_failed", f"Custom validation failed for {value}", record_index)
                    valid = False
            except Exception as e:
                result.add_error(rule.field_name, "custom_validation_error", f"Validator exception: {e}", record_index)
                valid = False

        return valid


# ==========================
# Example Usage
# ==========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    validator = SchemaValidator([
        ValidationRule("user_id", DataType.STRING, required=True, nullable=False),
        ValidationRule("age", DataType.INTEGER, min_value=0, max_value=150, required=False, nullable=True),
        ValidationRule("email", DataType.EMAIL, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", required=False, nullable=True),
        ValidationRule("properties", DataType.JSON, required=True, nullable=False),
    ])

    test_data = [
        {"user_id": "user_123", "age": 25, "email": "valid@example.com", "properties": {"city": "Seoul"}},
        {"user_id": "user_456", "age": 200, "email": "invalid-email", "properties": {"city": "Busan"}},
        {"age": 30, "email": "test@example.com"},  # missing user_id and properties
    ]

    result = validator.validate(test_data)
    print("Validation Summary:", result.get_summary())
    print("\nErrors:")
    for e in result.errors:
        print("  ", e)
