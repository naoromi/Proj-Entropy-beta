from os import makedirs
from os.path import exists
from shutil import rmtree

# cookie-cutter timing decorator
import time
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

@timeit
def reset_directory(directory_path):
    """
    Deletes the specified directory and creates a new empty one with the same name.
    
    Args:
    - directory_path (str): Path to the directory to reset.
    
    Returns:
    - None
    """
    if exists(directory_path):
        rmtree(directory_path)
    makedirs(directory_path)
    

def preflight_version_check():
    ### mandatory check if CUDA is available
    import torch, subprocess

    print('\n'+"#"*50)
    # version of pytorch
    print(f"PyTorch version: {torch.__version__}")

    # version of pytorch cuda
    cuda_version = torch.version.cuda
    print(f"Pytorch CUDA version: {cuda_version}")
    
    # version of pytorch cudnn
    cudnn_version = torch.backends.cudnn.version()
    print(f"Pytorch cudNN version: {cudnn_version}")

    
    # version of system cuda
    system_cuda_version = subprocess.run(['nvcc', '--version'], 
                                         stdout=subprocess.PIPE).stdout.decode('utf-8').strip().split('\n')[-2]
    print(f"System CUDA version: {system_cuda_version}")

    # Step 1: Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Step 2: Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        # Step 3: Get GPU properties
        for i in range(num_gpus):
            gpu_properties = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {gpu_properties.name}")
            # print(f"  Capability: {gpu_properties.major}.{gpu_properties.minor}")
            # print(f"  Total Memory: {gpu_properties.total_memory / (1024 ** 3):.2f} GB")
            # print(f"  MultiProcessor Count: {gpu_properties.multi_processor_count}")
            # print(f"  Max Threads per Block: {gpu_properties.max_threads_per_block}")
            # print(f"  Max Threads per SM: {gpu_properties.max_threads_per_multiprocessor}")
            # print(f"  Warp Size: {gpu_properties.warp_size}")
            # print(f"  Shared Memory per Block: {gpu_properties.shared_memory_per_block / 1024:.2f} KB")
    else:
        print("No CUDA-compatible GPU found.")

    print("#"*50+'\n')
    
