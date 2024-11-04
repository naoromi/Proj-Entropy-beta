# -*- coding: utf-8 -*-
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('/mnt/hbnas/datasets/video_classification_videos/h264-HEVC-150s/trans/HEVC_inference_results.csv')

# 从正序分割路径并获取第五个目录名称
df['true_category'] = df['filepath'].apply(lambda x: x.split('/')[7])

# 检查是否预测正确
df['prediction_correct'] = df['true_category'] == df['category']

# 统计每个类别的正确预测数量和总数
category_counts = df.groupby('true_category')['prediction_correct'].agg(['sum', 'count'])

# 计算每个类别的正确率
category_counts['accuracy'] = category_counts['sum'] / category_counts['count']

# 获取每个类别的预测
predictions = df.groupby(['true_category', 'category']).size().unstack(fill_value=0)

# 打印结果
print(category_counts)
print(predictions)

# 将结果输出为 CSV 文件
category_counts.to_csv('h264-HEVC-150s_category_counts.csv')
predictions.to_csv('h264-HEVC-150s_predictions.csv')

