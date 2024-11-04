"""
    Summary:
        Extracts the bitrate information from all videos in a directory 
        and saves it as CSV files.
    
    Main Contributors:
     - Felix Gan               <ganf@acm.org>
     - Yunan Ding              <yunan.ding@hkust.edu.cn>

"""
import os
from os import listdir, walk, makedirs
from os.path import (
    join, 
    dirname, realpath, basename, normpath, relpath, 
    isfile, 
    splitext)

from tqdm import tqdm
from numpy import extract
from sklearn.model_selection import train_test_split
from .ffmpeg_bitrate_stats.__main__ import extract_bitrate_python_interface
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils.fg_utils import timeit, reset_directory


def extract_cover(input_file_path, _output_dir, out_filename):
    # checking if it is a file
    if isfile(input_file_path):
        extract_bitrate_python_interface(_input=input_file_path, 
                                         custom_output_dir=_output_dir,
                                         custom_filename=out_filename,
                                         aggregation="time",
                                         output_format="csv")
        
        output_file_path = join(_output_dir, f"{out_filename}.csv")
        return output_file_path
    else:
        raise FileNotFoundError


def grab_all_videos(input_dir):
    # recursively grabs all videos in input_dir and all its subdirectories
    video_extensions = (".mp4", ".avi", ".mov",
                        ".mkv", ".flv", ".wmv", 
                        ".webm", ".mpeg", ".mpg")
    video_files = [ join(curdir, f)
                    for curdir, dirs, files in walk(input_dir, topdown=False) 
                    for f in files 
                    if f.endswith(video_extensions)]
    
    return video_files

def standardize_output_filenames(files, root_dir, depth=1):
    """ Given input file path, decide at which depth 
        to standardize output filename.
        e.g.
            input_file_path = /tmp/Game/Minecraft/PewDiePie/Minecraft_Skyblock_#1.mp4
            root_dir = /tmp/
            depth=1 --> Game
            depth=2 --> Minecraft
            depth=3 --> PewDiePie
            depth=4 --> Minecraft_Skyblock_#1
            ...
        
        video src: https://youtu.be/XozZYCqNo8Q
    """
    filenames = []
    
    for i, file in enumerate(files):
        root_dir, file = normpath(root_dir), normpath(file)

        if not file.startswith(root_dir):
            raise ValueError("The input file path must start with the root directory")
        
        path_components = relpath(file, root_dir).split(os.sep)  # the heavy lifting
        
        # Ensure the depth is within valid range
        if depth < 1 or depth > len(path_components):
            raise ValueError(f"Depth {depth} is out of range for the given path.")
        
        selected_component = path_components[depth - 1]
        
        # some numbering trickery, where if flattened, renumber, else keep original
        if depth == len(path_components):
            out_file_name = splitext(selected_component)[0]
        else:
            out_file_name = splitext(selected_component)[0] + str(i)
        
        filenames.append(out_file_name)
    
    return filenames

def extract_covers_with_threads(vids, filenames, output_dir, max_workers):
    # cookie-cutter embarrasingly parallelizable task executor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_cover, vid, output_dir, fname)
                   for vid, fname in zip(vids, filenames)]
        with tqdm(total=len(vids)) as pbar:
            for future in as_completed(futures):
                future.result()  # This will re-raise any exception that occurred in the thread
                pbar.update(1)

def extract_covers(input_dir, train_vids, test_vids, output_dir):
    # convert relpath to fullpath
    train_vids = [join(input_dir, p) for p in train_vids if p != ""]
    test_vids = [join(input_dir, p) for p in test_vids if p != ""]
    print(f"Train Vids: {len(train_vids)}, Test Vids: {len(test_vids)}")
    
    # flatten and standardize filename by category
    train_filenames = standardize_output_filenames(train_vids, input_dir, depth=1)
    print("standardized train", flush=True)
    
    test_filenames = standardize_output_filenames(test_vids, input_dir, depth=1)
    print("standardized test", flush=True)
    
    # make output directory for trains and tests
    train_output, test_output = join(output_dir, "train"), join(output_dir, "test")
    
    reset_directory(train_output)      # if we're here, we are re-extracting
    reset_directory(test_output)
    extract_covers_with_threads(train_vids, train_filenames, train_output, max_workers=16)
    extract_covers_with_threads(test_vids, test_filenames, test_output, max_workers=16)
    
    print("Done!")
    

