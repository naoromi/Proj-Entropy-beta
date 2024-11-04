# -*- coding: utf-8 -*-
from pathlib import Path
from os import makedirs
from os.path import join
from shutil import rmtree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
from .utils.fg_utils import timeit, reset_directory

def process_file(file_path, path_new, frame_num):
    if len(open(file_path).readlines()) >= frame_num:
        df = pd.read_csv(file_path, header=None, names=['name', 'time', 'size'])
        
        # Add the 'label' column using the file name
        df['label'] = file_path.stem
        
        # Apply the regex substitution to the 'label' column
        df['label'] = df['label'].str.extract(r'^([a-zA-Z]+)', expand=False)
        
        # Save the transformed DataFrame to a new CSV file
        new_file_path = join(path_new, file_path.name)
        df.to_csv(new_file_path, index=False)
    return file_path.name

# Helper function to read and process CSV files
def load_and_process_csv(i, sequence, read_path, frame_num):
    # i is a sequence order (alphabetical), so that order is deterministic
    file_path = join(read_path, f'{sequence}.csv')
    df = pd.read_csv(file_path, nrows=frame_num)
    df.insert(0, 'sequence', sequence)
    df['step'] = np.arange(df.shape[0])
    return i, df

@timeit
def process_file_parallel(path_ori, path_new, frame_num):
    all_files = sorted(Path(path_ori).glob('*.csv'))
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_path, path_new, frame_num) 
                   for file_path in all_files]
        
        for future in as_completed(futures):
            file_name = future.result()
            # print(f"Processed {file_name}")

@timeit
def process_sequences_parallel(sequence_ids, read_path, frame_num):
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(load_and_process_csv, i, seq, read_path, frame_num) 
                   for i, seq in enumerate(sequence_ids)]
        results = [f.result() for f in as_completed(futures)]
    
    # sorts again by original sequence ordering, to enforce determinism
    results = [x[1] for x in sorted(results, key=lambda x: x[0])]   
    return results

@timeit
def combine_into_design_matrix(path_new, frame_num):
    # compile all sequences into large df
    all_files = sorted(Path(path_new).glob('*.csv'))  # lol, this small step makes this process deterministic
    
    sequence_ids = []
    for fn in all_files:
        sequence_ids.append(fn.stem)
    
    cover_dfs = process_sequences_parallel(
        sequence_ids, path_new, frame_num)
    
    # transform X
    X = pd.concat(cover_dfs, ignore_index=True)[['sequence', 'size']]
    v = X.melt(id_vars=['sequence'])
    v['variable'] += v.groupby(['sequence', 'variable']).cumcount().astype(str)
    res = v.pivot_table(index=['sequence'], columns='variable', values='value',sort=False)  
    c = res.columns.str.extract(r'(\d+)')[0].values.astype(int)
    indexing = np.argsort(c)
    X = res.iloc[:, indexing]
    X.to_csv('./X_transformed.csv')
    
    # transform Y
    Y = pd.Series(c['label'][0] for c in cover_dfs)
    
    # save
    return X, Y

def collate_into_matrix(covers_path, design_matrix_path, split_name):
    original_covers = join(covers_path, split_name)
    transformed_covers = join(design_matrix_path, f'{split_name}_tmp')  # transformed covers
    
    frame_num = 3000
    
    # make temporary directory for storing transformed covers
    reset_directory(transformed_covers)

    # then in multiprocess transform covers
    process_file_parallel(original_covers, transformed_covers, frame_num)
    
    # then sequentially combine
    X, y = combine_into_design_matrix(transformed_covers, frame_num = 3000)
    
    # and save to file
    X.to_csv(join(design_matrix_path, f"X_{split_name}.csv"), index=False)
    y.to_csv(join(design_matrix_path, f"Y_{split_name}.csv"), index=False)
    
    # cleanup: remove tmp directory
    rmtree(transformed_covers)
    