import torch

def check():
    # version of pytorch
    print(f"PyTorch version: {torch.__version__}")

    # version of cuda
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")

    # version of cudnn
    cudnn_version = torch.backends.cudnn.version()
    print(f"cudNN version: {cudnn_version}")

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
            print(f"GPU {i}: {gpu_properties.name}")
            # print(f"  Capability: {gpu_properties.major}.{gpu_properties.minor}")
            # print(f"  Total Memory: {gpu_properties.total_memory / (1024 ** 3):.2f} GB")
            # print(f"  MultiProcessor Count: {gpu_properties.multi_processor_count}")
            # print(f"  Max Threads per Block: {gpu_properties.max_threads_per_block}")
            # print(f"  Max Threads per SM: {gpu_properties.max_threads_per_multiprocessor}")
            # print(f"  Warp Size: {gpu_properties.warp_size}")
            # print(f"  Shared Memory per Block: {gpu_properties.shared_memory_per_block / 1024:.2f} KB")
    else:
        print("No CUDA-compatible GPU found.")

if __name__ == "__main__":
    check()    
