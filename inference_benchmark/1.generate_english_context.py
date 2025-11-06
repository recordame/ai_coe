# pip install openai

from openai import OpenAI 
import pandas as pd
import os
import datetime

client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc",
    base_url="https://api.upstage.ai/v1"
)

def translate_to_english(statement: str) -> str:
    stream = client.chat.completions.create(
        model="upstage/solar-1-mini-chat",
        messages=[
            {
                "role": "system",
                "content": "당신은 금융 법률용어 전문가로 한국어와 영어에 능통한 번역가이기도 합니다. 사용자가 입력한 법 조항을 친절하게 영어로 번역해주세요! 대답을 할 때는 이런저런 설명(예: Translate the following into English!) 붙이지 말고 번역문만 출력해주세요!!"
            },
            {
                "role": "user",
                "content": f"{statement}\n"
            }
        ],
        stream=False,
        temperature=0,
        max_tokens=16384,
    )

    english_statement = stream.choices[0].message.content

    return english_statement

##################

base_contexts_df = pd.read_csv(f'{os.getcwd()}/base_contexts_202511062030.csv', index_col='id')
now = (pd.Timestamp.now(tz='Asia/Seoul')).strftime('%Y%m%d%H%M')

# 영문으로 번역하기 위한 context 복사
english_context = base_contexts_df['context']

for index in range(len(english_context)):
    print(index)
    english_context.iloc[index] = translate_to_english(english_context.iloc[index])
    english_context.to_csv(f'english_contexts_{now}.csv')

