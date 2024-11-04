import os
import filecmp

def compare_files_in_directories(dir1, dir2):
    # List of files that are the same
    same_files = []
    # List of files that differ
    diff_files = []
    
    # Get the list of files in each directory
    files1 = {f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
    files2 = {f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}
    
    # Ensure both directories have the same files
    common_files = files1.intersection(files2)
    unique_files_dir1 = files1.difference(files2)
    unique_files_dir2 = files2.difference(files1)
    
    if unique_files_dir1 or unique_files_dir2:
        print("The directories do not contain the same files.")
        if unique_files_dir1:
            print(f"Files only in {dir1}: {unique_files_dir1}")
        if unique_files_dir2:
            print(f"Files only in {dir2}: {unique_files_dir2}")
        return same_files, diff_files
    
    for file_name in common_files:
        file1 = os.path.join(dir1, file_name)
        file2 = os.path.join(dir2, file_name)
        
        # Compare the files
        if filecmp.cmp(file1, file2, shallow=False):
            same_files.append(file_name)
        else:
            diff_files.append(file_name)
    
    return same_files, diff_files

# Example usage
dir1 = '/mnt/hbnas/home/fgan/Proj-Entropy/Video_Classification_Model/3_model_input_data/test'
dir2 = '/mnt/hbnas/home/fgan/Proj-Entropy/Video_Classification_Model/3_model_input_data/test_2'
same_files, diff_files = compare_files_in_directories(dir1, dir2)

print("Files that are the same:")
for file in same_files:
    print(file)

print("\nFiles that differ:")
for file in diff_files:
    print(file)