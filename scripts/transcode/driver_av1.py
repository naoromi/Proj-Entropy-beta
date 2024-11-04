"""
    Summary:
        Multi-core, multi-gpu, thread and process safe 
        transcoding of videos to AV1 standard
        
    Changelog:
        Mon, 23 Jun 2024: 
            (feature) added coordinated logging of tasks. 
            (feature) can now save unfinished tasks to new json for quick resume
        Sun, 23 Jun 2024: 
            initial working prototype
    
    Main Contributors:
     - Felix Gan               <ganf@acm.org>

"""

import os, json, argparse
from queue import Empty
import multiprocessing as mp
import threading
import subprocess

# configs
NUM_TASK_PER_GPU = 8   # so, it seems our current setup allows 
                        # max 5 concurrent nvenc?
                        # so, we're using 5 GPU, with 1 task per GPU
                        # by the way
                        # number of processes = NUM_TASK_PER_GPU * len(visible_gpus)
visible_gpus = [1]

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
        # no need to skip, if we're reading from a dump of unfinished tasks
    # if os.path.exists(output_path):
    #     file_size_mb = os.path.getsize(output_path) / MB
    #     if file_size_mb >= 1:
    #         print(f"CP-SKIP: File '{output_path}' exists and > 1MB.")
    #         return
    
    command = ['ffmpeg', 
               '-y',
               '-i', input_path, 
               '-ss', '0', '-t', '150', 
               '-c:v', 'libaom-av1', 
               '-threads', '8', 
               '-tile-columns', "2",
               '-tile-rows', "1",
               '-row-mt', '1',  # Enable row-based multi-threading
               '-frame-parallel', '1',  # Enable frame parallel encoding
               '-preset', 'fast', 
               '-strict', '-2', 
               '-f', 'webm',
               output_path ]
    
    try:
        subprocess.call(command, 
                        env=env, 
                        # stdout=subprocess.DEVNULL, 
                        # stderr=subprocess.DEVNULL,
                        )
    except subprocess.CalledProcessError as e:
        return 1
    
    return 0
        
    
def worker(gpu_index, task_queue, completed_videos_threadsafe):
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
            path_pairs = task_queue.get(timeout=1)  # Adjust timeout as necessary
            if path_pairs is None:
                break
            transcode_video(path_pairs, env)
            
            # log finished task
            input_path, _ = path_pairs
            completed_videos_threadsafe.append(input_path)
            
        except Empty:
            break


def schedule_queues(path_pairs, num_gpus):
    """Divide the tasks into sub-queues based on the number of GPUs."""
    queues = [mp.Queue() for _ in range(len(visible_gpus))]
    
    for i, pair in enumerate(path_pairs):
        queues[i % num_gpus].put( pair )
    
    return queues


def cleanup(original_json_path, completed_videos_threadsafe):
    """ during premature termination, 
        finds out what tasks has NOT been completed, 
        and adds those to a list to avoid duplicate work next time
    """
    with open(original_json_path, mode='r') as f:
        vids = json.load(f)

    root = vids["root"]
    completed_videos_relpath = [os.path.relpath(path, root) 
                                for path in completed_videos_threadsafe]
    
    vids["relativePath"] = list( set(vids["relativePath"]) 
                                  - set(completed_videos_relpath) )
    
    # drop in directory of original json file
    same_dir = os.path.dirname(original_json_path)
    save_path = os.path.join(same_dir, "unfinished_av1.json")
    
    with open(save_path, 'w') as json_file:
        json.dump(vids, json_file, indent=4)
        
    print(f"## Finished transcoding {len(completed_videos_threadsafe)} tasks. ##")
    
    return save_path


def initiate_orchestration(input_json, output_dir):    
    path_pairs = get_input_and_output_paths(input_json, output_dir)
    
    # populate tasks queue in round-robin fashion
    queues = schedule_queues(path_pairs, len(visible_gpus))
    
    with mp.Manager() as manager:
        completed_videos_threadsafe = manager.list()
        processes = []
        # Create and start worker processes
        for i, gpu_index in enumerate(visible_gpus):
            for _ in range(NUM_TASK_PER_GPU):
                p = mp.Process(target=worker, args=(gpu_index, 
                                                    queues[i], 
                                                    completed_videos_threadsafe))
                processes.append(p)
                p.start()
    
        try:
            # Wait for all worker processes to exit
            for p in processes:
                p.join()
                
        except KeyboardInterrupt or EOFError:   # ctrl-c or ctrl-d
            save_path = cleanup(input_json, completed_videos_threadsafe)
            print(f"## Cleanup complete, unfinished tasks saved into {save_path}. ##")
            return 1
    
    return 0
    

def main():
    parser = argparse.ArgumentParser(description="Transcode all videos defined by JSON file")
    parser.add_argument("--json", help="path to reference json file", type=str, required=True)
    parser.add_argument("--output_dir", help="path to output directory", type=str, required=True)
    args = parser.parse_args()
    
    # check if there's a quick resume file (defaulted to same directory as original json file)
    same_dir = os.path.dirname(args.json)
    quick_resume_path = os.path.join(same_dir, "unfinished_av1.json")
    if os.path.exists(quick_resume_path):
        args.json = quick_resume_path
    
    # transcoding with some multiprocessing magic
    status = initiate_orchestration(args.json, args.output_dir)
    
    if status == 0:
        print("Processing complete for all files.")
    

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
