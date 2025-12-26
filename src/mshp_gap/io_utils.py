"""Atomic writes and I/O utilities."""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mshp_gap.paths import ensure_dir


def atomic_write(target_path: Path, write_func: callable, *args, **kwargs) -> Path:
    """Write atomically using temp file + rename."""
    target_path = Path(target_path)
    ensure_dir(target_path.parent)

    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=f"{target_path.stem}_", dir=target_path.parent
    )
    temp_path = Path(temp_path)

    try:
        os.close(temp_fd)
        write_func(temp_path, *args, **kwargs)
        temp_path.replace(target_path)
        return target_path
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> Path:
    """Atomically write JSON data to a file."""
    def _write(temp_path, data, indent):
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
    return atomic_write(path, _write, data, indent)


def atomic_write_csv(path: Path, df, **kwargs) -> Path:
    """Atomically write a DataFrame to CSV."""
    def _write(temp_path, df, **kwargs):
        df.to_csv(temp_path, index=False, **kwargs)
    return atomic_write(path, _write, df, **kwargs)


def atomic_write_geojson(path: Path, gdf) -> Path:
    """Atomically write a GeoDataFrame to GeoJSON."""
    def _write(temp_path, gdf):
        gdf.to_file(temp_path, driver="GeoJSON")
    return atomic_write(path, _write, gdf)


def write_metadata_sidecar(
    data_path: Path, script_name: str, run_id: str, description: str,
    inputs: list[str], row_count: int = None, columns: list[str] = None, **extra
) -> Path:
    """Write metadata sidecar file."""
    meta_path = data_path.parent / f"{data_path.stem}_metadata.json"
    metadata = {
        "_generated": datetime.now(timezone.utc).isoformat(),
        "_script": script_name,
        "_run_id": run_id,
        "description": description,
        "inputs": inputs,
    }
    if row_count is not None:
        metadata["row_count"] = row_count
    if columns is not None:
        metadata["columns"] = columns
    metadata.update(extra)
    return atomic_write_json(meta_path, metadata)


def update_manifest(raw_dir: Path, filename: str, source_url: str, row_count: int):
    """Update manifest.json with download info."""
    manifest_path = raw_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest[filename] = {
        "source_url": source_url,
        "download_date": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
    }
    atomic_write_json(manifest_path, manifest)

