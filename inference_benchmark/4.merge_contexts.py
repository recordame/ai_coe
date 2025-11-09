import pandas as pd
import os

now = (pd.Timestamp.now(tz='Asia/Seoul')).strftime('%Y%m%d%H%M')

base_context_df = pd.read_csv(f'{os.getcwd()}/output/base_contexts_202511071953.csv', index_col='id')
shuffled_english_context_df = pd.read_csv(f'{os.getcwd()}/output/shuffled_english_contexts_202511062332.csv', index_col='id')
unrelated_context_df = pd.read_csv(f'{os.getcwd()}/output/unrelated_contexts_202511062317.csv', index_col='id')

base_contexts = base_context_df['context']
shuffled_english_contexts = shuffled_english_context_df['context']
unrelated_contexts = unrelated_context_df['context']

merged_contexts = []

for index in range(0, 70):
    b = base_contexts.iloc[index]
    e = shuffled_english_contexts.iloc[index]
    u = unrelated_contexts.iloc[index]

    if (index % 3) == 0:
        merged_contexts.append(f'{b}\n{e}\n{u}')
    elif (index % 3) == 1:
        merged_contexts.append(f'{e}\n{u}\n{b}')
    else:
        merged_contexts.append(f'{u}\n{b}\n{e}')

base_context_df['context'] = merged_contexts

base_context_df.to_csv(f'{os.getcwd()}/output/merged_contexts_{now}.csv', index=True)