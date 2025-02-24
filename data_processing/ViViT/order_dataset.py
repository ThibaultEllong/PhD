import os
import json
import shutil
import csv
import tqdm
from pathlib import Path

def get_subfolders(root_dir):
    """Retrieve subfolder names from the specified root directory."""
    return os.listdir(root_dir)

def read_json(file_path):
    """Read and parse a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def match_name_label(data):
    """Create a dictionary mapping take names to parent task names."""
    return {task.get("take_name"): task.get("parent_task_name") for task in data}

def move_and_rename_videos(source_dir, dest_dir, pattern, label_data, output_csv):
    """Move and rename videos based on label data and log changes to a CSV file."""
    renamed_files = []
    unique_id = 0

    with tqdm.tqdm(total=len(os.listdir(source_dir))) as pbar:
        for root, _, filenames in os.walk(source_dir):
            for filename in filenames:
                if root.split(os.sep)[-1] == "448" and pattern in filename:
                    src_path = os.path.join(root, filename)
                    class_name = root.split(os.sep)[-4]
                    label = label_data.get(class_name, "Unknown")

                    train_dir = os.path.join(dest_dir, "train")
                    test_dir = os.path.join(dest_dir, "test")

                    if unique_id <= len(os.listdir(source_dir)) * 0.7:
                        if label not in os.listdir(train_dir):
                            os.mkdir(os.path.join(train_dir, label))
                        dest_path = os.path.join(train_dir, label, f"{class_name}.mp4")
                    else:
                        if label not in os.listdir(test_dir):
                            os.mkdir(os.path.join(test_dir, label))
                        dest_path = os.path.join(test_dir, label, f"{class_name}.mp4")

                    shutil.copy(src_path, dest_path)
                    renamed_files.append((unique_id, filename, f"{class_name}.mp4", label))
                    unique_id += 1
                    pbar.update(1)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Unique ID", "Original Name", "New Name", "Label"])
        writer.writerows(renamed_files)

def main():
    
    task_data = read_json("/media/thibault/DATA/these_thibault/Dataset/data/EgoExo4D/annotations/metadata/takes.json")
    label_data = match_name_label(task_data)

    source_directory = Path('/media/thibault/DATA/these_thibault/Dataset/data/EgoExo4D/takes/train/takes/')
    pattern = '_214'
    dest_path = "/media/thibault/DATA/these_thibault/Dataset/data/EgoExo4D/test_dataset/"
    output_csv_file = Path(dest_path) / 'metadata.csv'

    move_and_rename_videos(source_directory, dest_path, pattern, label_data, output_csv_file)

if __name__ == "__main__":
    main()
