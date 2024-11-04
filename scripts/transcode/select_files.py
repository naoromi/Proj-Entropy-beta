import os, json
import argparse
import subprocess, multiprocessing


def save_to_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def collect_file_paths(directory):
    # this create "flattened" version of file path, easier to parse
    paths = {
        "root": directory, 
        "relativePath": []
    }
    for root, dirs, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            paths['relativePath'].append(relative_path)
    return paths


def collect_file_structure(directory):
    # this creates a "standard" file system representation in json
    # but needs recursion to parse, not easy
    file_structure = {}
    for root, dirs, files in os.walk(directory):
        # Get the relative path of the current directory
        rel_path = os.path.relpath(root, directory)
        if rel_path == '.':
            rel_path = ''
        
        # Create nested dictionaries for directories
        dir_structure = file_structure
        if rel_path:
            for part in rel_path.split(os.sep):
                dir_structure = dir_structure.setdefault(part, {})

        # Add files to the current directory structure
        for file in files:
            dir_structure[file] = None

    return file_structure


def parse_args():
    parser = argparse.ArgumentParser(description="Collect file paths from a directory and save to a JSON file.")
    parser.add_argument("directory", help="Path to the directory to walk through.", type=str)
    parser.add_argument("output", help="Path to the output JSON file.", type=str)
    args = parser.parse_args()
    
    return args.directory, args.output

def main():
    directory, output = parse_args()
    file_paths = collect_file_paths(directory)
    save_to_json(file_paths, output)
    # file_structure = collect_file_structure(directory)
    # save_to_json(file_structure, output)
    print(f"Collected {len(file_paths)} file paths and saved to {output}")

if __name__ == "__main__":
    main()

