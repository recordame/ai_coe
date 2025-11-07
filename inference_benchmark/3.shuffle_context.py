import pandas as pd
import os

now = (pd.Timestamp.now(tz='Asia/Seoul')).strftime('%Y%m%d%H%M')

source_contexts = pd.read_csv(f'{os.getcwd()}/english_contexts_202511062216.csv', index_col='id')

shuffled_contexts = source_contexts['context'].sample(frac=1, random_state=42)

print(source_contexts.head())
print(shuffled_contexts.head())

shuffled_contexts.to_csv(f'./shuffled_english_contexts_{now}.csv', index=True)