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
                     # Extract class name from the path
                    class_name = root.split(os.sep)[-4]  # Assumes class_name is two levels up from the file
                    if "eye_gaze" in os.listdir(os.path.join(source_dir, class_name)):
                     
                        
                        # Construct full file path
                        video_src_path = os.path.join(root, filename)
                        gaze_src_path = os.path.join(source_dir, class_name, "eye_gaze/general_eye_gaze_2d.csv")

                        # Get the corresponding label
                        label = label_data.get(class_name, "Unknown")
                        

                        train_dir = os.path.join(dest_dir, "train")
                        test_dir = os.path.join(dest_dir, "test")    
                        if unique_id <= len(os.listdir(source_dir)) * .7:
                            if label not in os.listdir(train_dir) or class_name not in os.listdir(os.path.join(train_dir, label)):
                                os.makedirs(os.path.join(train_dir,label, class_name), exist_ok=True)
                            video_dest_path = os.path.join(train_dir, label, class_name)
                            gaze_dest_path = os.path.join(train_dir, label, class_name)
                        else:
                            if label not in os.listdir(test_dir):
                                os.makedirs(os.path.join(test_dir, label, class_name), exist_ok=True)
                            video_dest_path = os.path.join(test_dir, label, class_name)
                            gaze_dest_path = os.path.join(test_dir, label, class_name)

                        # Move and rename the file
                        print()
                        shutil.copy(video_src_path, video_dest_path)
                        shutil.copy(gaze_src_path, gaze_dest_path)
                        # Log the original and new file names along with the label
                        renamed_files.append((unique_id, filename, class_name, label))

                        # Increment the unique ID
                        unique_id += 1
                        pbar.update(1)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Unique ID", "Original Name", "New Name", "Label"])
        writer.writerows(renamed_files)

def main():
    
    task_data = read_json("/home/cerisnadm/Bureau/Thibault/Ego4d/takes.json")
    label_data = match_name_label(task_data)

    source_directory = Path('/home/cerisnadm/Bureau/Thibault/Ego4d/takes/train/takes/')
    pattern = '_214'
    dest_path = "/home/cerisnadm/Bureau/Thibault/Ego4d/dataset/train/"
    output_csv_file = Path(dest_path) / 'metadata.csv'

    move_and_rename_videos(source_directory, dest_path, pattern, label_data, output_csv_file)

if __name__ == "__main__":
    main()

