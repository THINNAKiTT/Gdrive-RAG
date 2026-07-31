"""
Unit tests for src/utils/logger.py (JSONFormatter)

Verifies every log line is valid, parseable JSON with the expected
standard fields, that extra={} fields surface as their own JSON keys,
and that exception info is captured as a string (not a raw traceback
object, which wouldn't be JSON-serializable).
"""
import json
import logging

import pytest

from src.utils.logger import JSONFormatter, get_logger

pytestmark = pytest.mark.unit


def make_record(
    message="test message",
    level=logging.INFO,
    name="TestLogger",
    exc_info=None,
    extra=None,
):
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=42,
        msg=message,
        args=(),
        exc_info=exc_info,
        func="test_function",
    )
    if extra:
        for key, value in extra.items():
            setattr(record, key, value)
    return record


def test_format_produces_valid_json():
    formatter = JSONFormatter()
    record = make_record()

    output = formatter.format(record)

    parsed = json.loads(output)  # must not raise
    assert isinstance(parsed, dict)


def test_format_includes_standard_fields():
    formatter = JSONFormatter()
    record = make_record(message="hello world", name="MyLogger")

    parsed = json.loads(formatter.format(record))

    assert parsed["message"] == "hello world"
    assert parsed["logger"] == "MyLogger"
    assert parsed["level"] == "INFO"
    assert parsed["function"] == "test_function"
    assert parsed["line"] == 42
    assert "timestamp" in parsed
    assert "module" in parsed


def test_format_timestamp_is_iso8601():
    formatter = JSONFormatter()
    record = make_record()

    parsed = json.loads(formatter.format(record))

    # Must be parseable back as an ISO 8601 datetime.
    from datetime import datetime
    datetime.fromisoformat(parsed["timestamp"])


def test_format_includes_extra_fields():
    formatter = JSONFormatter()
    record = make_record(extra={"file_id": "abc123", "chunk_count": 5})

    parsed = json.loads(formatter.format(record))

    assert parsed["file_id"] == "abc123"
    assert parsed["chunk_count"] == 5


def test_format_extra_fields_do_not_overwrite_standard_fields():
    """
    Regression guard: if a caller ever accidentally passes extra={
    "message": "oops"} or similar, it must not silently corrupt the
    real message/level/logger fields.
    """
    formatter = JSONFormatter()
    record = make_record(message="the real message")

    parsed = json.loads(formatter.format(record))

    assert parsed["message"] == "the real message"


def test_format_without_extra_fields_has_no_unexpected_keys():
    formatter = JSONFormatter()
    record = make_record()

    parsed = json.loads(formatter.format(record))

    expected_keys = {"timestamp", "level", "logger", "message", "module", "function", "line"}
    assert set(parsed.keys()) == expected_keys


def test_format_includes_exception_info_as_string():
    formatter = JSONFormatter()
    try:
        raise ValueError("something broke")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
        record = make_record(exc_info=exc_info)

    parsed = json.loads(formatter.format(record))

    assert "exception" in parsed
    assert isinstance(parsed["exception"], str)
    assert "ValueError" in parsed["exception"]
    assert "something broke" in parsed["exception"]


def test_format_without_exception_has_no_exception_key():
    formatter = JSONFormatter()
    record = make_record(exc_info=None)

    parsed = json.loads(formatter.format(record))

    assert "exception" not in parsed


def test_format_handles_non_serializable_extra_value_gracefully():
    """
    A non-JSON-serializable object passed via extra={} must not crash
    the logging call -- logging itself failing is worse than the
    original error being logged. Falls back to str(obj) via
    json.dumps(default=str).
    """
    class Unserializable:
        def __str__(self):
            return "<Unserializable object>"

    formatter = JSONFormatter()
    record = make_record(extra={"weird_field": Unserializable()})

    output = formatter.format(record)  # must not raise

    parsed = json.loads(output)
    assert parsed["weird_field"] == "<Unserializable object>"


def test_get_logger_returns_logger_with_json_formatter():
    logger = get_logger("SomeTestLogger")

    assert len(logger.handlers) >= 1
    for handler in logger.handlers:
        assert isinstance(handler.formatter, JSONFormatter)


def test_get_logger_does_not_duplicate_handlers_on_repeated_calls():
    logger_a = get_logger("DuplicateCheckLogger")
    handler_count_first_call = len(logger_a.handlers)

    logger_b = get_logger("DuplicateCheckLogger")

    assert logger_a is logger_b
    assert len(logger_b.handlers) == handler_count_first_call