# pip install openai

from openai import OpenAI 
import pandas as pd
import os

client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc",
    base_url="https://api.upstage.ai/v1"
)

now = (pd.Timestamp.now(tz='Asia/Seoul')).strftime('%Y%m%d%H%M')

unrelated_context_df = pd.DataFrame()
unrelated_context = []

for index in range(1, 71):
    print(index)
    stream = client.chat.completions.create(
            model="solar-pro2",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 법률 전문가 입니다. 사용자가 '아무 법조항이나 알려줘' 라는 입력을 하면 금융관련 법률 이 외의 법률(예, 헌법, 민법, 형법, 행정법 등)에서 법률과 하위 조, 항, 호,목 을 출력해줘! 그 어떤 법률 해설이나 마크다운 서식을 적용하지 말고 줄바꿈만 해줘. 단 매 생성마다 새로운 법안을 불러와줘!"
                },
                {
                    "role": "user",
                    "content": "아무 법조항이나 알려줘"
                }
            ],
            stream=False,
            temperature=1,
            max_tokens=16384,
        )
    unrelated_context_str = stream.choices[0].message.content
    print(unrelated_context_str)
    unrelated_context.append(unrelated_context_str)

unrelated_context_df['context'] = unrelated_context
unrelated_context_df.to_csv(f'{os.getcwd()}/output/unrelated_contexts_{now}.csv', index=True)