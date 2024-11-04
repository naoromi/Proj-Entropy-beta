import os, json, argparse
import subprocess, multiprocessing

# configs
NUM_PROCESSES = 4   # so, each GPU can only allow... 4 concurrent nvenc?
                    # to avoid having to coordinate GPUs, 
                    # we're just using 4 processes. 
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5" # A100 does not support nvenc

# constants
KB = 2^10
MB = 2^20
GB = 2^30

def get_input_and_output_paths(json_file, output_directory):
    with open(json_file, mode='r') as f:
        vids = json.load(f)

    root = vids["root"]
    paths = vids["relativePath"]
    
    path_pairs = []
    for path in paths:
        input_path = os.path.join(root, path)
        output_path = os.path.join(output_directory, path)
        path_pairs.append( (input_path, output_path) )
    
    return path_pairs


def transcode_video(path_pair):
    ''' using path_pair because pool.map expects function 
        to takes one argument only
        could use pool.starmap but why
        '''
    input_path, output_path = path_pair
    
    # make parent directories if not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # if file exists and sufficiently large (>1MB), skip
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / MB
        if file_size_mb >= 1:
            print(f"CP-SKIP: File '{output_path}' exists and > 1MB.")
            return
    
    command = ['ffmpeg', 
               '-y',
               '-i', input_path, 
               '-ss', '0', '-t', '150', 
               '-c:v', 'h264_nvenc', 
               '-threads', '4', 
               '-preset', 'fast', 
               '-f', 'mp4', '-strict', '-2', 
               output_path]
    subprocess.call(command)

def main():
    parser = argparse.ArgumentParser(description="Transcode all videos defined by JSON file")
    parser.add_argument("--json", help="path to reference json file", type=str, required=True)
    parser.add_argument("--output_dir", help="path to output directory", type=str, required=True)
    args = parser.parse_args()
    path_pairs = get_input_and_output_paths(args.json, args.output_dir)

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        pool.map(transcode_video, path_pairs)

    print("Processing complete.")


if __name__ == "__main__":
    main()


