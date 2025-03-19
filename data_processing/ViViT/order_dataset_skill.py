#!/usr/bin/env python3

import argparse
import json
import shutil
import csv
from pathlib import Path
from tqdm import tqdm

def load_annotations(json_path: Path):
    with open(json_path, 'r') as f:
        return json.load(f)["annotations"]

def build_label_data(annotations):
    return [
        {
            "scenario_name": ann["scenario_name"],
            "task_name": ann["video_paths"].get("ego").split("/")[-3],
            "skill": ann["proficiency_score"],
            "video_filename": Path(ann["video_paths"]["ego"]).name
        }
        for ann in annotations
    ]

def reorganize_videos(source_root: Path, dest_root: Path, label_data: list, csv_path: Path):
    dest_root.mkdir(parents=True, exist_ok=True)
    rows = []
    uid = 0

    for entry in tqdm(label_data, desc="Reorganizing videos"):
        task = entry["task_name"]
        skill = entry["skill"]
        filename = entry["video_filename"]

        # Determine source subfolder
        for split in ("train", "test"):
            candidate = source_root / split / "takes" / task / "frame_aligned_videos" / "downscaled" / "448" / filename
            print("path:", candidate)
            if candidate.exists():
                src = candidate
                dst_dir = dest_root / split / skill
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f"{task}.mp4"
                shutil.copy(src, dst)
                rows.append((uid, src, task, skill))
                uid += 1
                break
        else:
            print(f"[WARNING] Could not find video for task '{task}' -> {filename}")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Unique ID", "Original Path", "Task", "Skill"])
        writer.writerows(rows)

def main(args):
    annotations = load_annotations(Path(args.annotations))
    label_data = build_label_data(annotations)
    reorganize_videos(
        source_root=Path(args.source_root),
        dest_root=Path(args.dest_root),
        label_data=label_data,
        csv_path=Path(args.dest_root) / "metadata.csv",
    )
    print("Done. Metadata saved to", Path(args.dest_root) / "metadata.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize EgoExo4D videos into task/skill folders")
    parser.add_argument("--source_root", required=True, help="Path to EgoExo4D/takes directory")
    parser.add_argument("--annotations", required=True, help="Path to proficiency_demonstrator_train.json")
    parser.add_argument("--dest_root", required=True, help="Destination root for reorganized dataset")
    args = parser.parse_args()
    main(args)
