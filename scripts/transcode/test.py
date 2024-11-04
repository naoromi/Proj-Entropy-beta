

from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])
    q.put(1)

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    print(q.get(timeout))    # prints "[42, None, 'hello']"
    p.join()





exit()

from queue import Empty
import multiprocessing as mp
import os

MAX_TASKS_PER_GPU = 4
visible_gpus = [3, 4, 5]
num_gpus = len(visible_gpus)
queues = [mp.Queue() for _ in range(max(visible_gpus)+1)]



def some_func(args):
    # _input, _output = path_pair
    # print(gpu_index, _input, _output)
    print(args)


def worker(gpu_index, task_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    while True:
        try:
            task = task_queue.get(timeout=1)  # Adjust timeout as necessary
            if task is None:
                break
            some_func(task)
        except Empty:
            break
        
        
path_pair = [ (i, i+1) for i in range(1, 50)]

for i, path_pair in enumerate(path_pair):
    gpu_index = visible_gpus[i % num_gpus]
    queues[gpu_index].put( (path_pair[0], path_pair[1]) )


# Create and start worker processes
processes = []
for gpu_index in visible_gpus:
    for _ in range(MAX_TASKS_PER_GPU):
        p = mp.Process(target=worker, args=(gpu_index, queues[gpu_index]))
        p.start()
        processes.append(p)

# Wait for all worker processes to exit
for p in processes:
    p.join()