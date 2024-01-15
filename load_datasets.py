# Datasets used:
# https://www.kaggle.com/datasets/khotijahs1/data-train
# https://www.kaggle.com/datasets/danofer/sarcasm
# https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection 
# https://github.com/EducationalTestingService/sarcasm 

import pandas as pd
import numpy as np

"""
 Boolean labels
 0: Not sarcastic
 1: Sarcastic

 Tweet  Label
"""
df1 = pd.read_csv('archive/English_Sarcasm.csv')
df1 = df1.rename(columns={'Tweet': 'text', 'Label': 'label'})
df1 = df1[['text', 'label']]
long_df = df1.copy()
long_df['context'] = np.nan

"""
 Boolean labels
 0: Not sarcastic
 1: Sarcastic

 Tweet  Reverse  label
"""

df2 = pd.read_csv('archive/sarcasm.csv')
df2 = df2.rename(columns={'Tweet': 'text'})
df2 = df2[['text', 'label']]
df2['context'] = np.nan
long_df = pd.concat([long_df, df2], ignore_index=True)

"""
 Boolean labels
 0: Not sarcastic
 1: Sarcastic

 label  comment   author  subreddit   score   ups   downs   date  created_utc   parent_comment
"""

df3 = pd.read_csv('archive/train-balanced-sarcasm.csv')
df3 = df3.rename(columns={'comment': 'text', 'parent_comment': 'context'})
df3 = df3[['text', 'label', 'context']]
long_df = pd.concat([long_df, df3], ignore_index=True)

"""
 Boolean labels
 0: Not sarcastic
 1: Sarcastic

 article_link   headline    is_sarcastic
"""

df4 = pd.read_json('archive/Sarcasm_Headlines_Dataset.json', lines=True)
df4 = df4.rename(columns={'headline': 'text', 'is_sarcastic': 'label'})
df4 = df4[['text', 'label']]
df4['context'] = np.nan
long_df = pd.concat([long_df, df4], ignore_index=True)

"""
 Boolean labels
 0: Not sarcastic
 1: Sarcastic

 is_sarcastic   headline    article_link
"""

df5 = pd.read_json('archive/Sarcasm_Headlines_Dataset_v2.json', lines=True)
df5 = df5.rename(columns={'headline': 'text', 'is_sarcastic': 'label'})
df5 = df5[['text', 'label']]
df5['context'] = np.nan
long_df = pd.concat([long_df, df5], ignore_index=True)

"""
 Boolean labels
 NOT_SARCASM: Not sarcastic
 SARCASM: Sarcastic

 label   context    response    id
"""

df6 = pd.read_json('archive/sarcasm_detection_shared_task_reddit_testing.jsonl', lines=True)
df6 = df6.rename(columns={'response': 'text'})
df6 = df6[['text', 'label', 'context']]
df6['label'] = df6['label'].map({'NOT_SARCASM': 0, 'SARCASM': 1})
df6['context'] = df6['context'].apply(lambda context: ' '.join(context))
long_df = pd.concat([long_df, df6], ignore_index=True)

"""
 Boolean labels
 NOT_SARCASM: Not sarcastic
 SARCASM: Sarcastic

 label   response    context
"""

df7 = pd.read_json('archive/sarcasm_detection_shared_task_reddit_training.jsonl', lines=True)
df7 = df7.rename(columns={'response': 'text'})
df7 = df7[['text', 'label', 'context']]
df7['label'] = df7['label'].map({'NOT_SARCASM': 0, 'SARCASM': 1})
df7['context'] = df7['context'].apply(lambda context: ' '.join(context))
long_df = pd.concat([long_df, df7], ignore_index=True)

"""
 Boolean labels
 NOT_SARCASM: Not sarcastic
 SARCASM: Sarcastic

 label   context    response    id
"""

df8 = pd.read_json('archive/sarcasm_detection_shared_task_twitter_testing.jsonl', lines=True)
df8 = df8.rename(columns={'response': 'text'})
df8 = df8[['text', 'label', 'context']]
df8['label'] = df8['label'].map({'NOT_SARCASM': 0, 'SARCASM': 1})
df8['context'] = df8['context'].apply(lambda context: ' '.join(context))
long_df = pd.concat([long_df, df8], ignore_index=True)

"""
 Boolean labels
 NOT_SARCASM: Not sarcastic
 SARCASM: Sarcastic

 label   response    context
"""

df9 = pd.read_json('archive/sarcasm_detection_shared_task_twitter_training.jsonl', lines=True)
df9 = df9.rename(columns={'response': 'text'})
df9 = df9[['text', 'label', 'context']]
df9['label'] = df9['label'].map({'NOT_SARCASM': 0, 'SARCASM': 1})
df9['context'] = df9['context'].apply(lambda context: ' '.join(context))
long_df = pd.concat([long_df, df9], ignore_index=True)

print(len(long_df))

long_df.to_csv('complete_data.csv', encoding='utf-8', index=False)