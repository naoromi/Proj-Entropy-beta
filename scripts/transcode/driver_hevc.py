import os, json, argparse
from queue import Empty
import multiprocessing as mp
import threading
import subprocess

# configs
NUM_TASK_PER_GPU = 1    # so, it seems our current setup allows 
                        # max 5 concurrent nvenc?
                        # so, we're using 5 GPU, with 1 task per GPU
visible_gpus = [3, 4, 5, 6, 7]

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


def transcode_video(path_pairs, env):
    ''' using path_pair because pool.map expects function 
        to takes one argument only
        could use pool.starmap but why
        '''
    input_path, output_path = path_pairs
        
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
               '-c:v', 'hevc_nvenc', 
               '-threads', '4',
               '-preset', 'fast', 
               '-f', 'mp4', '-strict', '-2', 
               output_path]
    subprocess.call(command, env=env)
    
    
def worker(gpu_index, task_queue):
    """ worker that abtracts central Queue coordination
            aka middle management
        sets the environment variables
        calls the functions
        and talks with orchestrating Queue
    """
    # sets environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    
    # poll your task queue until there's no more, 
    # mp.queue is thread and process safe
    while True:
        try:
            task = task_queue.get(timeout=1)  # Adjust timeout as necessary
            if task is None:
                break
            transcode_video(task, env)
        except Empty:
            break



def schedule_queues(path_pairs, num_gpus):
    """Divide the tasks into sub-queues based on the number of GPUs."""
    queues = [mp.Queue() for _ in range(len(visible_gpus))]
    
    for i, pair in enumerate(path_pairs):
        queues[i % num_gpus].put( pair )
    
    return queues

def main():
    parser = argparse.ArgumentParser(description="Transcode all videos defined by JSON file")
    parser.add_argument("--json", help="path to reference json file", type=str, required=True)
    parser.add_argument("--output_dir", help="path to output directory", type=str, required=True)
    args = parser.parse_args()
    path_pairs = get_input_and_output_paths(args.json, args.output_dir)
    
    # populate tasks queue in round-robin fashion
    queues = schedule_queues(path_pairs, len(visible_gpus))
    
    # Create and start worker processes
    processes = []
    for i, gpu_index in enumerate(visible_gpus):
        for _ in range(NUM_TASK_PER_GPU):
            p = mp.Process(target=worker, args=(gpu_index, queues[i]))
            processes.append(p)
            p.start()

    # Wait for al©l worker processes to exit
    for p in processes:
        p.join()


    print("Processing complete.")


if __name__ == "__main__":
    main()




'''
    # Create and start worker processes
    processes = []
    for i, gpu_index in enumerate(visible_gpus):
        for _ in range(NUM_TASK_PER_GPU):
            p = mp.Process(target=worker, args=(gpu_index, queues[i]))
            processes.append(p)
            p.start()

    # Wait for all worker processes to exit
    for p in processes:
        p.join()
        
        
    # Create and start worker threads
    threads = []
    for i, gpu_index in enumerate(visible_gpus):
        for _ in range(NUM_TASK_PER_GPU):
            t = threading.Thread(target=worker, args=(gpu_index, queues[i]))
            threads.append(t)
            t.start()
        # break
    
    for t in threads:
        t.join()


'''