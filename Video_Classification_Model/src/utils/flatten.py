import os
import shutil

def flatten_and_rename_videos(base_dir):
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            video_count = 1
            for root, _, files in os.walk(category_path):
                for file in files:
                    # Pad the number with preceding zeros to ensure they have the same length
                    # padded_number = str(video_count).zfill(4)  # Change 4 to the desired number of digits
                    new_filename = f"{category.replace(' ', '_')}{video_count}{os.path.splitext(file)[1]}"
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(category_path, new_filename)
                    shutil.move(old_file_path, new_file_path)
                    print(new_filename)
                    print(f"Moved {old_file_path} to {new_file_path}")
                    video_count += 1

def remove_empty_directories(base_dir):
    for root, dirs, _ in os.walk(base_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  # Check if the directory is empty
                os.rmdir(dir_path)
                print(f"Removed empty directory {dir_path}")

# Specify the base directory where the categories are located
base_dir = "/mnt/hbnas/home/fgan/Proj-Entropy/Video_Classification_Model/1_input_videos/hevc_test/"

flatten_and_rename_videos(base_dir)
remove_empty_directories(base_dir)

