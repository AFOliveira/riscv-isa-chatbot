#!/usr/bin/env python3
"""
Export RISC-V data from riscv-unified-db to the chatbot data directory.

Usage:
    python export_data.py                           # Export all available configs
    python export_data.py /path/to/riscv-unified-db # Specify path explicitly
    RISCV_UNIFIED_DB=/path/to/repo python export_data.py  # Via env var
"""

import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TARGET_DIR = SCRIPT_DIR / "data"

# Configs to export (in order of preference)
CONFIGS = ["rv64", "rv32"]


def find_riscv_unified_db() -> Path | None:
    """Try to find riscv-unified-db repository."""

    # 1. Check environment variable
    env_path = os.environ.get("RISCV_UNIFIED_DB")
    if env_path:
        p = Path(env_path)
        if (p / "gen" / "resolved_spec").exists():
            return p

    # 2. Check command line argument
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if (p / "gen" / "resolved_spec").exists():
            return p

    # 3. Check if we're a submodule (parent directory)
    parent = SCRIPT_DIR.parent
    if (parent / "gen" / "resolved_spec").exists():
        return parent

    # 4. Check sibling directory
    sibling = SCRIPT_DIR.parent / "riscv-unified-db"
    if (sibling / "gen" / "resolved_spec").exists():
        return sibling

    return None


def export_config(repo_root: Path, config: str) -> int:
    """Export a single config. Returns number of files exported."""

    source_dir = repo_root / "gen" / "resolved_spec" / config
    config_target = TARGET_DIR / config

    if not source_dir.exists():
        print(f"  Skipping {config} (not generated)")
        return 0

    print(f"\nExporting {config}...")
    print(f"  Source: {source_dir}")
    print(f"  Target: {config_target}")

    # Clean config target directory
    if config_target.exists():
        shutil.rmtree(config_target)

    config_target.mkdir(parents=True, exist_ok=True)

    # Directories to copy
    dirs_to_copy = ["inst", "csr", "ext"]
    total_files = 0

    for dir_name in dirs_to_copy:
        src = source_dir / dir_name
        dst = config_target / dir_name

        if not src.exists():
            print(f"    Skipping {dir_name}/ (not found)")
            continue

        print(f"    Copying {dir_name}/...")
        shutil.copytree(src, dst)

        file_count = sum(1 for _ in dst.rglob("*.yaml"))
        total_files += file_count
        print(f"      -> {file_count} files")

    return total_files


def export_data(repo_root: Path):
    """Export YAML data from generated specs to chatbot data directory."""

    print(f"Target directory: {TARGET_DIR}")

    # Clean entire target directory
    if TARGET_DIR.exists():
        print("Cleaning existing data directory...")
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Export each config
    total_files = 0
    exported_configs = []

    for config in CONFIGS:
        files = export_config(repo_root, config)
        if files > 0:
            total_files += files
            exported_configs.append(config)

    if not exported_configs:
        print("\nERROR: No configs were exported!")
        print("Please generate at least one config first:")
        print(f"  cd {repo_root}")
        print("  ./do gen:resolved_arch CFG=rv64")
        print("  ./do gen:resolved_arch CFG=rv32")
        sys.exit(1)

    print(f"\nDone! Exported {total_files} YAML files")
    print(f"Configs: {', '.join(exported_configs)}")


def main():
    repo_root = find_riscv_unified_db()

    if repo_root is None:
        print("ERROR: Could not find riscv-unified-db repository")
        print("\nPlease specify the path:")
        print(f"  python {sys.argv[0]} /path/to/riscv-unified-db")
        print("\nOr set the environment variable:")
        print(f"  RISCV_UNIFIED_DB=/path/to/repo python {sys.argv[0]}")
        sys.exit(1)

    print(f"Found riscv-unified-db at: {repo_root}")
    export_data(repo_root)


if __name__ == "__main__":
    main()
