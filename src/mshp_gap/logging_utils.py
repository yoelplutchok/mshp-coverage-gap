"""JSONL structured logging utilities."""
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mshp_gap.paths import LOGS_DIR, ensure_dir


def generate_run_id() -> str:
    """Generate unique run ID: YYYYMMDD_HHMMSS_<8-char-uuid>"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


class JSONLHandler(logging.Handler):
    """Logging handler that writes structured JSON lines."""

    def __init__(self, log_path: Path, run_id: str):
        super().__init__()
        self.log_path = log_path
        self.run_id = run_id
        self._file = None

    def _ensure_file(self):
        if self._file is None:
            ensure_dir(self.log_path.parent)
            self._file = open(self.log_path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord):
        try:
            self._ensure_file()
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if hasattr(record, "event_type"):
                log_entry["event_type"] = record.event_type
            if hasattr(record, "context"):
                log_entry["context"] = record.context
            # `context` may include numpy scalar types which aren't JSON serializable.
            # `default=str` keeps logging robust without breaking the pipeline.
            self._file.write(json.dumps(log_entry, default=str) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self._file:
            self._file.close()
        super().close()


_RUN_ID: str | None = None


def get_run_id() -> str:
    """Get or generate a unique run ID for this session."""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = generate_run_id()
    return _RUN_ID


def get_logger(script_name: str) -> logging.Logger:
    """Get logger with console and JSONL handlers."""
    run_id = get_run_id()
    logger = logging.getLogger(f"{script_name}_{run_id}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(console)

        # JSONL file handler
        log_file = LOGS_DIR / f"{script_name}_{run_id}.jsonl"
        jsonl_handler = JSONLHandler(log_file, run_id)
        jsonl_handler.setLevel(logging.DEBUG)
        logger.addHandler(jsonl_handler)

    return logger


def log_step_start(logger, step_name: str, **context):
    """Log the start of a processing step."""
    logger.info(f"Starting: {step_name}", extra={
        "event_type": "step_start", "context": {"step_name": step_name, **context}
    })


def log_step_end(logger, step_name: str, **context):
    """Log the end of a processing step."""
    logger.info(f"Completed: {step_name}", extra={
        "event_type": "step_end", "context": {"step_name": step_name, **context}
    })


def log_output_written(logger, path: Path, row_count: int = None, **context):
    """Log that an output file was written."""
    msg = f"Output written: {path}"
    if row_count:
        msg += f" ({row_count:,} rows)"
    logger.info(msg, extra={
        "event_type": "output_written",
        "context": {"path": str(path), "row_count": row_count, **context}
    })

