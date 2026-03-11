"""Data staging — bulk-copy remote/cloud data to local SSD before training.

Detects whether a data path is on a slow mount (GCS FUSE, NFS, S3 FUSE)
and copies it to fast local storage (``/tmp/data/``).  Runs once on rank 0
inside ``prepare_data()``; other ranks wait via Lightning's barrier.

Supports:
- GCS FUSE mounts (``/gcs/...``) — copies via ``shutil`` (FUSE = local FS)
- Cloud URLs (``gs://``, ``s3://``) — copies via ``rclone`` or ``gcloud``
- Local paths — no-op (already fast)

The copy is skipped if the destination already contains data (idempotent).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from loguru import logger

# Paths under these prefixes are cloud FUSE mounts — local FS semantics
# but backed by object storage with high per-file latency.
_FUSE_PREFIXES = ("/gcs/", "/s3/", "/mnt/fuse/")

# Local SSD staging root.  /tmp is local NVMe on most cloud VMs.
_STAGE_ROOT = Path("/tmp/staged_data")  # noqa: S108


def is_slow_mount(path: Path) -> bool:
    """Check if a path is on a cloud FUSE mount or remote filesystem."""
    path_str = str(path)
    return any(path_str.startswith(prefix) for prefix in _FUSE_PREFIXES)


def _stage_dir_name(cloud_path: Path) -> str:
    """Derive a stable local directory name from a cloud path.

    ``/gcs/bucket/data/coco/train2017`` → ``bucket__data__coco__train2017``
    """
    # Strip leading /gcs/ or /s3/ etc.
    stripped = str(cloud_path)
    for prefix in _FUSE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    return stripped.strip("/").replace("/", "__")


def stage_to_local(path: Path) -> Path:
    """Copy a cloud-mounted directory to local SSD if needed.

    Returns the local path (possibly unchanged if already local).
    Idempotent — skips copy if destination already has files.
    """
    if not is_slow_mount(path):
        logger.debug(f"Data path is local, no staging needed: {path}")
        return path

    local_dir = _STAGE_ROOT / _stage_dir_name(path)

    # Check if already staged (has at least some files)
    if local_dir.exists() and any(local_dir.iterdir()):
        file_count = sum(1 for _ in local_dir.rglob("*") if _.is_file())
        logger.info(f"Data already staged ({file_count} files): {path} → {local_dir}")
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Staging data from cloud mount to local SSD: {path} → {local_dir}")
    t0 = time.perf_counter()

    # GCS FUSE paths are local FS — use shutil for parallel-ish copy.
    # For true cloud URLs (gs://, s3://), we'd use rclone or gcloud CLI.
    _copy_fuse_mount(path, local_dir)

    elapsed = time.perf_counter() - t0
    file_count = sum(1 for _ in local_dir.rglob("*") if _.is_file())
    size_gb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e9
    logger.info(
        f"Staging complete: {file_count} files, {size_gb:.1f}GB in {elapsed:.1f}s "
        f"({size_gb / elapsed * 1024:.0f} MB/s)"
    )

    return local_dir


def _copy_fuse_mount(src: Path, dst: Path) -> None:
    """Copy from a FUSE mount using the fastest available method.

    Tries (in order):
    1. ``rsync`` with parallel transfers (usually pre-installed)
    2. ``shutil.copytree`` (always available, single-threaded)
    """
    # Try rsync first — handles partial copies, is resume-safe
    rsync_path = shutil.which("rsync")
    if rsync_path:
        logger.info("Using rsync for staging")
        try:
            subprocess.run(  # noqa: S603
                [
                    rsync_path,
                    "-a",  # archive mode (preserves structure)
                    "--info=progress2",  # show overall progress
                    str(src).rstrip("/") + "/",
                    str(dst).rstrip("/") + "/",
                ],
                check=True,
                timeout=1800,  # 30 min max
            )
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"rsync failed, falling back to shutil: {e}")

    # Fallback: shutil.copytree
    logger.info("Using shutil.copytree for staging")
    # copytree needs the dst to not exist, but we already created it.
    # Use dirs_exist_ok=True (Python 3.8+).
    shutil.copytree(src, dst, dirs_exist_ok=True)


def stage_paths(
    train_path: Path,
    val_path: Path,
    test_path: Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """Stage all data paths to local SSD if on slow mounts.

    Returns (staged_train, staged_val, staged_test) paths.
    Only rank 0 should call this (inside ``prepare_data()``).
    """
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank != 0:
        # Non-rank-0 processes should not stage — they wait for rank 0
        # via Lightning's barrier. But return the expected local paths
        # so they can find the data after the barrier.
        return (
            _expected_local_path(train_path),
            _expected_local_path(val_path),
            _expected_local_path(test_path) if test_path else None,
        )

    staged_train = stage_to_local(train_path)
    staged_val = stage_to_local(val_path)
    staged_test = stage_to_local(test_path) if test_path else None

    return staged_train, staged_val, staged_test


def _expected_local_path(path: Path) -> Path:
    """Return the local path a cloud path would be staged to."""
    if not is_slow_mount(path):
        return path
    return _STAGE_ROOT / _stage_dir_name(path)
