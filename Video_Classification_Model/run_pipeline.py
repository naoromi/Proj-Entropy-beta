'''
Overview: videos -> covers -> design matrix -> train -> eval -> predict

1. place videos to extract in folder `1-input_videos`
2. run `python extract.py`
3. the extracted covers are placed in `2-covers`
    - The resulting CSVs will have three columns, namely Video Path, Frame Number, Frame size (in bytes)
4. run `python collate.py` to collate covers into a single well formatted design matrix, saved in `3_model_input_data`
5. run `python train.py`, checkpoints saved in `4_checkpoints`
6. run `python eval.py` to evaluate, be sure to change `CHECKPOINT_PATH` before running
7. run `python predict.py` to predict on new dataset, which needs to be in a design matrix. Change `MODEL_PATH` and `data_path` accordingly. 

# usage example:
    python3 run_pipeline.py --config experiments/hevc_marginal_1000K.yaml --gpus 0
    
'''
# always attempt gain priority, for now.
import os
try:
    new_nice = os.nice(-2)
    print(f"sudo: changed niceness to {new_nice}.")
except Exception as e:
    print("no sudo: Cannot reduce niceness.")
    pass

import os, sys, yaml, argparse
from os import listdir, makedirs
from os.path import join, isdir, dirname, realpath, abspath
from datetime import datetime
from time import perf_counter
import wandb

# configure path to src to allow relative path without infinite '.....'
sys.path.append(join(dirname(realpath(__file__)), "src"))
print(join(dirname(realpath(__file__)), "src"))

# Import the necessary functions or modules from the src directory
from src import (
    extract_covers,
    collate_into_matrix,
    train_model,
    load_and_test_model,
    # inference_new_data
)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-g', '--gpus', type=str, required=True)
parser.add_argument('-f', '--force',
                    action='store_true')  # on/off flag
args = parser.parse_args()

# Load configuration from YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

paths                 = config['paths']
mode                  = config['mode']
gpus                  = list(map(int, args.gpus.split(',')))  # input as option
architecture          = config['architecture']
INPUT_VIDEOS_PATH     = paths['input_videos'] # may be internal, but often isn't

# run specifics
model                 = paths['model']
archive               = paths['archive']
dataset               = paths['dataset']

# list of train/test split files (to prevent shooting of one's own foot)
train_list            = paths['train_list']     # created using create_train_test_list.ipynb
test_list             = paths['test_list']

## project internal paths
PROJ_BASE_PATH        = paths['base']
RUN_DIRECTORY         = join(model, archive, dataset)
COVERS_PATH           = join(PROJ_BASE_PATH, "2_covers", RUN_DIRECTORY)
DESIGN_MATRIX_PATH    = join(PROJ_BASE_PATH, "3_model_input_data", RUN_DIRECTORY)
CHECKPOINTS_PATH      = join(PROJ_BASE_PATH, "4_checkpoints", RUN_DIRECTORY)
RUN_SAVE_PATH         = join(CHECKPOINTS_PATH)
                            #optionally add: datetime.today().strftime('%Y_%m_%d__%H_%M_%S'))
# RUN_SAVE_PATH         = f"{CHECKPOINTS_PATH}-large"

print(INPUT_VIDEOS_PATH)
print(COVERS_PATH)
print(DESIGN_MATRIX_PATH)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
print("CUDA Visible GPUs:", os.environ["CUDA_VISIBLE_DEVICES"])

def run():
    if mode == 'train':
        # 1 - Extract covers
        print("1. Extracting Video Covers...")
        if not os.path.exists(COVERS_PATH) or args.force:
            with open(train_list, 'r') as f:
                train_vids = f.read().split("\n")
            
            with open(test_list, 'r') as f:
                test_vids = f.read().split("\n")
            
            extract_covers(INPUT_VIDEOS_PATH, 
                           train_vids, test_vids,
                           COVERS_PATH)
    
        # 2 - Collate covers into design matrix
        print("2. Running collate_frames...")
        if not os.path.exists(DESIGN_MATRIX_PATH) or args.force:
            collate_into_matrix(COVERS_PATH, DESIGN_MATRIX_PATH, "train")
            collate_into_matrix(COVERS_PATH, DESIGN_MATRIX_PATH, "test")

    
        # 3 - Train model
        print("3. Running train_model...")
        # if force enabled, overwrite previous checkpoints
        if args.force:
            makedirs(RUN_SAVE_PATH, exist_ok=True)
        else:
            makedirs(RUN_SAVE_PATH, exist_ok=False)
        
        # Start a W&B Run
        wandb.login()
        run = wandb.init(project="entropy",
                         name=f"{model}-{archive}-{dataset}",
                         tags=[model, archive, dataset])
        
        train_model(DESIGN_MATRIX_PATH, 
                    RUN_SAVE_PATH, 
                    architecture,
                    gpus)
        
        run.finish()

    if mode == 'eval':
        # 4 - Evaluate model
        best_model_savepath = join(RUN_SAVE_PATH, "best_model.pth")
        
        print(f"4. Running evaluate_model with {best_model_savepath}...")
        load_and_test_model(DESIGN_MATRIX_PATH, 
                            best_model_savepath, 
                            gpus)

    # # 5 - Predict new data
    # if mode == 'predict':
        # print(f"Running predict_new_data with MODEL_PATH={MODEL_PATH} and data_path={NEW_DATA_PATH}...")
        # os.environ['MODEL_PATH'] = MODEL_PATH
        # os.environ['data_path'] = NEW_DATA_PATH
        # predict_new_data(MODEL_PATH, NEW_DATA_PATH)


if __name__ == '__main__':
    start = perf_counter()
    run()
    print(f"------------\
          \n  Total Runtime: {perf_counter()-start:.2f}s\
          \n------------")
