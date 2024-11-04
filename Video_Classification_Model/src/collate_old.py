# -*- coding: utf-8 -*-
import os, sys, re, glob, pathlib, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed

cur_directory = os.path.dirname(os.path.realpath(__file__))


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

def readline_count(file_name):
    return len(open(file_name).readlines())

@timeit
def file_select(path_ori, path_new, frame_num):
    all_files = sorted(pathlib.Path(path_ori).glob('*.csv'))
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_path, path_new, frame_num) for file_path in all_files]
        
        for future in as_completed(futures):
            file_name = future.result()
            # print(f"Processed {file_name}")

def process_file(file_path, path_new, frame_num):
    if readline_count(file_path) >= frame_num:
        df = pd.read_csv(file_path, header=None, names=['name', 'time', 'size'])
        
        # Add the 'label' column using the file name
        df['label'] = file_path.stem
        
        # Apply the regex substitution to the 'label' column
        df['label'] = df['label'].str.extract(r'^([a-zA-Z]+)', expand=False)
        
        # Save the transformed DataFrame to a new CSV file
        new_file_path = os.path.join(path_new, file_path.name)
        df.to_csv(new_file_path, index=False)
    return file_path.name

@timeit
def rename(path_new):
    i = 0
    files = [file for file in os.listdir(path_new) if not file.startswith('.')]
    for file in files:
        i = i + 1
        Olddir = os.path.join(path_new, file)   
        if os.path.isdir(Olddir):     
                continue
        filetype = '.csv'      
        Newdir = os.path.join(path_new, str(i) + filetype) 
        os.rename(Olddir, Newdir)  
    return True
    exit()

@timeit
def generate_train_test_idx_and_labels(path_new):
    all_files = sorted(pathlib.Path(path_new).glob('*.csv'))
    
    sequence_ids, class_labels = [], []
    for fn in all_files:
        temp = pd.read_csv(fn, usecols=['label'], nrows=1, skiprows=0)
        sequence_ids.append(fn.stem)
        class_labels.append(temp['label'][0])
    
    # Split into train and test sets
    
    return train_test_split(sequence_ids, 
                            class_labels, 
                            test_size=0.2, 
                            random_state=42)
    
    


# Helper function to read and process CSV files
def load_and_process_csv(sequence, read_path, frame_num):
    file_path = os.path.join(read_path, f'{sequence}.csv')
    df = pd.read_csv(file_path, nrows=frame_num)
    df.insert(0, 'sequence', sequence)
    df['step'] = np.arange(df.shape[0])
    return df    

def process_sequences_parallel(sequence_ids, read_path, frame_num):
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(load_and_process_csv, seq, read_path, frame_num) for seq in sequence_ids]
        result = pd.concat([f.result() for f in as_completed(futures)], ignore_index=True)
    return result

def prepare_labels(train_labels, test_labels, path_train):
    train_labels_path = os.path.join(path_train, 'train_labels.csv')
    test_labels_path = os.path.join(path_train, 'test_labels.csv')
    
    pd.DataFrame(train_labels, columns=["class_label"]).to_csv(
        train_labels_path, index=False)
    pd.DataFrame(test_labels, columns=["class_label"]).to_csv(
        test_labels_path, index=False)
    

def data_combine(train_ids, test_ids, path_new, path_train, frame_num):
    X_train_path = os.path.join(path_train, 'train.csv')
    X_test_path = os.path.join(path_train, 'test.csv')

     # Process train and test sequences in parallel
    X_train = process_sequences_parallel(
        train_ids, path_new, frame_num
    ).drop(columns=["name", "label"])
    
    X_test = process_sequences_parallel(
        test_ids, path_new, frame_num
    ).drop(columns=["name", "label"])

    X_train.to_csv(X_train_path,index=False)
    X_test.to_csv(X_test_path, index=False)
    

@timeit
def data_combine_train(path_train):
    train_ori_path = os.path.join(path_train, 'train.csv')
    test_ori_path = os.path.join(path_train, 'test.csv')
    train_new_sequence_path = os.path.join(path_train, 'train_new_sequence.csv')
    test_new_sequence_path = os.path.join(path_train, 'test_new_sequence.csv')
    train_new_path = os.path.join(path_train, 'train_new.csv')
    test_new_path = os.path.join(path_train, 'test_new.csv')
    train_label = os.path.join(path_train, 'train_labels.csv')
    test_label = os.path.join(path_train, 'test_labels.csv')
    data_combine = os.path.join(path_train, 'test_combine.csv')
    label_combine = os.path.join(path_train, 'label_combine.csv')
    
    train_ori = pd.read_csv(train_ori_path)
    test_ori = pd.read_csv(test_ori_path)


    # train_csv处理
    df_new = train_ori.drop(columns=[ "time", "step"])  
    v = df_new.melt(id_vars=['sequence'])
    v['variable'] += v.groupby(['sequence', 'variable']).cumcount().astype(str)
    res = v.pivot_table(index=['sequence'], columns='variable', values='value',sort=False)  
    c = res.columns.str.extract(r'(\d+)')[0].values.astype(int)  
    res.iloc[:, np.argsort(c)].to_csv(
        train_new_sequence_path)  # size0   size1   size2  ...  size2997  size2998  size2999 排序并保存
    res.iloc[:, np.argsort(c)].to_csv(train_new_path, index=False) 

    # test_csv处理
    df_new1 = test_ori.drop(columns=["time", "step"]) 
    v = df_new1.melt(id_vars=['sequence'])
    v['variable'] += v.groupby(['sequence', 'variable']).cumcount().astype(str)
    res = v.pivot_table(index=['sequence'], columns='variable', values='value',
                        sort=False)  # sort=False  
    c = res.columns.str.extract(r'(\d+)')[0].values.astype(int)  
    res.iloc[:, np.argsort(c)].to_csv(test_new_sequence_path)
    res.iloc[:, np.argsort(c)].to_csv(test_new_path, index=False)

    ####train test数据合并 测试集使用，训练集不使用以下文件
    df1 = pd.read_csv(test_new_path)
    df2 = pd.read_csv(train_new_path)
    df = pd.concat([df1, df2])  # 合并
    # df.to_csv(data_combine,encoding = 'utf-8',index=False)
    df.to_csv(data_combine, encoding='utf-8', index=['sequence'])
    # print(df)
    ####label数据合并 测试集使用，训练集不使用以下文件
    df1 = pd.read_csv(test_label)
    df2 = pd.read_csv(train_label)
    df = pd.concat([df1, df2])  
    # df.to_csv(label_combine,encoding = 'utf-8',index=False)
    df.to_csv(label_combine, encoding='utf-8', index=['sequence'])

@timeit
def reset_directory(directory_path):
    """
    Deletes the specified directory and creates a new empty one with the same name.
    
    Args:
    - directory_path (str): Path to the directory to reset.
    
    Returns:
    - None
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)


def collate_into_matrix(covers_path, design_matrix_path):
    #path_ori 原始文件 (csv_output)
    #path_new 筛选完帧数的文件 (all new csv, with 3000 frames)
    #path_train选择合并帧数文件
    
    path_ori = covers_path
    path_new = os.path.join(design_matrix_path, 'tmp')  # transformed covers
    path_train= design_matrix_path  # training design matrix
    
    frame_num = 3000

    # first clean output directory
    reset_directory(path_train)
    os.makedirs(path_new)

    # then sequentially each of these
    file_select(path_ori, path_new, frame_num)
    rename(path_new)
    
    (train_ids, 
     test_ids, 
     train_labels, 
     test_labels) = generate_train_test_idx_and_labels(path_new)
    
    prepare_labels(train_labels, test_labels, path_train)
    
    data_combine(train_ids, test_ids, path_new, path_train, frame_num)
    data_combine_train(path_train)
    

