# -*- coding: utf-8 -*-
import pandas as pd

csv_path = "/mnt/hbnas/home/fgan/Proj-Entropy/models/VidClassModel/results/inference_results_VP9.csv"
# csv_path = "/mnt/hbnas/datasets/video_classification_videos/h264-HEVC-150s/trans/HEVC_inference_results.csv"

df = pd.read_csv(csv_path)


df['true_category'] = df['filepath'].apply(lambda x: x.split('/')[7])

df['prediction_correct'] = df['true_category'] == df['category']

category_counts = df.groupby('true_category')['prediction_correct'].agg(['sum', 'count'])

category_counts['accuracy'] = category_counts['sum'] / category_counts['count']

tot_sum = category_counts['sum'].sum()
tot_cnt = category_counts['count'].sum()

category_counts.loc["TOTAL"] = [tot_sum, tot_cnt, tot_sum / tot_cnt]

print(category_counts)

