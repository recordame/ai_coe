import argparse
import json
import os
import common.utils as utils
import pandas as pd

from tqdm import tqdm
from common.generation_prompts import PROMPTS
from openai import OpenAI

pipeline = "Pipeline-1"
model = "upstage/solar-1-mini-chat"
expert_levels = ["low_level_expert", "mid_level_expert", "high_level_expert"]
num_of_data = 1
max_batch = 1

client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc", base_url="https://api.upstage.ai/v1"
)

def batch_chat_template(batch, prompt, expert_level, args):
    messages = []
    system_prompt = prompt["system"][expert_level].format(domain_name=args.domain, lang=args.lang)

    for article in batch["Article"]:
        user_prompt = prompt["user"].format(domain_name=args.domain, input_text=article)

        messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    return messages


def make_question(args, dataset):
    print(f"[{pipeline}] 기사별 질문 생성 시작")

    model_name = args.model_name
    prompt = PROMPTS["question_generation"][args.lang]

    for expert_level in expert_levels:
        output_filename = f"P1-{args.domain}_{expert_level}_{args.num_of_data}_articles-question.json"
        processed_dataset = dataset

        formatted_msg = []

        for batch in tqdm(processed_dataset.iter(batch_size=args.max_batch_size), total=args.num_of_data, desc=f"[{pipeline}] {expert_level} 질문 생성중"):
            messages_list = batch_chat_template(batch, prompt, expert_level, args)

            for headline, article, messages in zip(batch["Headline"], batch["Article"], messages_list):
                while True:
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=4096,
                            temperature=0.1,
                        )

                        questions = response.choices[0].message.content

                        # 답변이 json형식이 아닌 경우, json 응답을 내놓을 때 까지 반복
                        json.loads(questions)

                        formatted_msg.append(
                            {
                                "headline": headline,
                                "article": article,
                                "questions": json.loads(questions),
                            }
                        )

                        break
                    except:
                        continue

        utils.write_json_file(formatted_msg, output_filename)
        utils.write_jsonl_file(formatted_msg, output_filename + "l")

    print(f"[{pipeline}] 기사별 질문 생성 완료")