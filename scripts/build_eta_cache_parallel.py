#!/usr/bin/env python
"""并行构建 eta cache - 使用多进程加速"""
import json
import pickle
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

LOG_DIR = Path("saves/train_logs/llama32_1b_tofu_safe")
ETA_CACHE_FILE = LOG_DIR / "eta_cache.json"


def load_eta_from_chunk(chunk_file: Path) -> Tuple[int, float]:
    """从单个 chunk 文件加载 eta"""
    try:
        with open(chunk_file, "rb") as f:
            records = pickle.load(f)
        if records and len(records) > 0:
            rec = records[0]
            return rec["step_id"], rec.get("eta", 0.0)
    except Exception as e:
        print(f"Error loading {chunk_file}: {e}", file=sys.stderr)
    return -1, 0.0


def build_eta_cache_parallel(num_workers: int = 32) -> Dict[int, float]:
    """并行构建完整的 eta cache"""
    # 加载现有缓存
    eta_cache = {}
    if ETA_CACHE_FILE.exists():
        try:
            with open(ETA_CACHE_FILE) as f:
                eta_cache = {int(k): v for k, v in json.load(f).items()}
            print(f"Loaded existing eta cache: {len(eta_cache)} entries")
        except Exception as e:
            print(f"Failed to load existing cache: {e}")

    # 找出需要加载的 chunk 文件
    chunk_files = sorted(LOG_DIR.glob("step_records_chunk_*.pkl"))
    print(f"Total chunk files: {len(chunk_files)}")

    # 过滤已缓存的
    chunks_to_load = []
    for cf in chunk_files:
        # 从文件名解析 step_id
        step_id = int(cf.stem.split("_")[-1])
        if step_id not in eta_cache:
            chunks_to_load.append(cf)

    print(f"Chunks to load: {len(chunks_to_load)}")

    if not chunks_to_load:
        print("All etas already cached!")
        return eta_cache

    # 并行加载
    loaded = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_eta_from_chunk, cf): cf for cf in chunks_to_load}

        for future in as_completed(futures):
            step_id, eta = future.result()
            if step_id >= 0:
                eta_cache[step_id] = eta
                loaded += 1
                if loaded % 100 == 0:
                    print(f"Loaded {loaded}/{len(chunks_to_load)} etas...")
                    # 增量保存
                    with open(ETA_CACHE_FILE, "w") as f:
                        json.dump(eta_cache, f)

    # 最终保存
    with open(ETA_CACHE_FILE, "w") as f:
        json.dump(eta_cache, f)

    print(f"Done! Total eta cache entries: {len(eta_cache)}")
    return eta_cache


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()

    build_eta_cache_parallel(num_workers=args.workers)
