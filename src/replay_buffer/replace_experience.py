import os
import json
import argparse
import numpy as np
from datetime import datetime
from collections import deque

"""
Dynamic Experience Replacement Script

Replaces older saved experience with newer experience.
Customizable through arguments with sensible defaults.

Usage:
    python src/replay_buffer/replace_experience.py --algorithm qlearning
    python src/replay_buffer/replace_experience.py --algorithm dqn --keep 3
    python src/replay_buffer/replace_experience.py --dry-run
"""

def get_buffer_files(base_path, algorithm, task='default'):
    """Get all saved buffer files for an algorithm/task sorted by age"""
    search_dir = os.path.join(base_path, algorithm, task)
    if not os.path.exists(search_dir):
        print(f"No data found at {search_dir}")
        return []
    
    files = [f for f in os.listdir(search_dir) if f.startswith('buffer_')]
    files.sort()  # Oldest first (timestamp in filename)
    return [os.path.join(search_dir, f) for f in files]

def get_total_size_mb(files):
    """Get total size of files in MB"""
    total = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    return total / (1024 * 1024)

def replace_old_experience(algorithm, task='default', keep=3, 
                           max_size_mb=100, base_path='results/replay_data',
                           dry_run=False):
    """
    Replace older experience files with newer ones.
    
    Args:
        algorithm:   Algorithm name (qlearning, sarsa, dqn etc.)
        task:        Task name (default)
        keep:        Number of most recent buffer files to keep
        max_size_mb: Maximum total storage in MB before forcing deletion
        base_path:   Root directory for replay data
        dry_run:     If True, show what would be deleted without deleting
    """
    files = get_buffer_files(base_path, algorithm, task)
    
    if not files:
        print(f"No buffer files found for {algorithm}/{task}")
        return

    total_mb = get_total_size_mb(files)
    print(f"\nAlgorithm: {algorithm} | Task: {task}")
    print(f"Files found: {len(files)} | Total size: {total_mb:.2f} MB")
    print(f"Keeping {keep} most recent files\n")

    # Files to delete: everything except the most recent `keep` files
    to_delete = files[:-keep] if len(files) > keep else []

    # Also delete if over size limit
    if total_mb > max_size_mb:
        print(f"⚠️  Over size limit ({total_mb:.1f}MB > {max_size_mb}MB)")
        while get_total_size_mb(files) > max_size_mb and len(files) > 1:
            to_delete.append(files.pop(0))

    if not to_delete:
        print("Nothing to delete — storage is clean!")
        return

    for f in to_delete:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        if dry_run:
            print(f"[DRY RUN] Would delete: {f} ({size_mb:.2f} MB)")
        else:
            os.remove(f)
            # Also remove matching meta file
            meta = f.replace('buffer_', 'meta_').replace('.npy', '.json')
            if os.path.exists(meta):
                os.remove(meta)
            print(f"Deleted: {f} ({size_mb:.2f} MB)")

    print(f"\n✅ Done! Kept {min(keep, len(files))} most recent files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace old replay buffer experience')
    parser.add_argument('--algorithm',   type=str, default='all',
                        help='Algorithm to clean (qlearning/sarsa/dqn/all)')
    parser.add_argument('--task',        type=str, default='default')
    parser.add_argument('--keep',        type=int, default=3,
                        help='Number of recent buffer files to keep')
    parser.add_argument('--max-size-mb', type=float, default=100,
                        help='Max total storage in MB')
    parser.add_argument('--base-path',   type=str, 
                        default='results/replay_data')
    parser.add_argument('--dry-run',     action='store_true',
                        help='Show what would be deleted without deleting')
    
    args = parser.parse_args()

    algorithms = ['qlearning', 'sarsa', 'td_lambda', 'dqn'] \
                 if args.algorithm == 'all' else [args.algorithm]

    for algo in algorithms:
        replace_old_experience(
            algorithm=algo,
            task=args.task,
            keep=args.keep,
            max_size_mb=args.max_size_mb,
            base_path=args.base_path,
            dry_run=args.dry_run
        )