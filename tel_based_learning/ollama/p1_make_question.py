import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
# import py7zr
import requests
from tqdm import tqdm

import c_utils
from c_generation_prompts import PROMPTS

# llama3.2:1b
# gpt-oss:120b
# gpt-oss:20b
# qwen3-coder:30b
# alibayram/Qwen3-30B-A3B-Instruct-2507:latest
# qwen3:30b-a3b-instruct-2507-fp16
# dengcao/Qwen3-Reranker-8B:F16
# dengcao/Qwen3-Reranker-8B:Q8_0
# dengcao/Qwen3-Reranker-8B:Q5_K_M
# dengcao/Qwen3-Reranker-8B:Q4_K_M
# dengcao/Qwen3-Reranker-8B:Q3_K_M
# dengcao/Qwen3-Embedding-8B:F16
# dengcao/Qwen3-Embedding-8B:Q8_0
# dengcao/Qwen3-Embedding-8B:Q5_K_M
# dengcao/Qwen3-Embedding-8B:Q4_K_M
# nomic-embed-text:latest
# qwen3:235b-a22b
# qwen3:30b-a3b
# qwen3:30b
# deepseek-r1:70b
# qwen3:8b
# deepseek-r1:8b
# deepseek-r1:7b
model = "gpt-oss:20b"

# 아래 전문가 레벨을 조절하여 질문 생성
# low:  초급인력 (퍼플렉시티 기준: 중급 상단~고급 초입)
# mid:  중급인력 (고급 독해자)
# high: 고급인력 (최상위 고급, 즉 전문가 수준)
expert_levels = ["low", "mid", "high"]
num_of_data = 100
max_batch = 1
num_of_workers = 1


def call_ollama(model, messages=None, stream=False):
    url = "http://ollama:11434/api/chat"

    with requests.Session() as session:
        response = session.post(url, json={"model": model, "messages": messages, "stream": stream})
        response.raise_for_status()

        return response.json()["message"]["content"]


def batch_chat_template(batch, prompt, expert_level, args):
    messages = []
    system_prompt = prompt["system"][expert_level].format(domain_name=args.domain)

    for article in batch["Article"]:
        user_prompt = prompt["user"].format(domain_name=args.domain, input_text=article)

        messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    return messages


def parsing_q_list(row):
    qa_list = row["questions"]

    return {"qa_list": json.dumps(qa_list, ensure_ascii=False)}


def process_single_question(headline, article, messages, model_name):
    while True:
        try:
            questions = call_ollama(model=model_name, messages=messages)

            json.loads(questions)

            return {
                "headline": headline,
                "article": article,
                "questions": json.loads(questions),
            }
        except Exception as e:
            print(e)
            continue


def make_question(args):
    model_name = args.model_name
    prompt = PROMPTS[args.prompt_type][args.lang]

    csv_path = "../bloomberg_financial_news_120k.csv"

    print(f"데이터셋 로드 중...")
    dataset = pd.read_csv(csv_path).head(args.num_of_data)
    print(f"데이터셋 로드 완료. 총 {len(dataset)}개 처리 예정.")

    for expert_level in expert_levels:
        output_filename = f"sample_questions/1.{args.domain}_{expert_level}_{args.model_name.replace(':', '-')}_{args.num_of_data}.json"  # _{pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")}
        processed_dataset = dataset

        formatted_msg = []

        for i in tqdm(
                range(0, len(processed_dataset)),
                total=args.num_of_data,
                desc=f"{expert_level}레벨 담당자 질문 생성중",
        ):
            batch_df = processed_dataset.iloc[i: i + args.max_batch_size]
            batch = {
                "Headline": batch_df["Headline"].tolist(),
                "Article": batch_df["Article"].tolist(),
            }

            messages_list = batch_chat_template(batch, prompt, expert_level, args)

            tasks = []
            with ThreadPoolExecutor(max_workers=args.num_of_workers) as executor:
                for idx, (headline, article, messages) in enumerate(zip(batch["Headline"], batch["Article"], messages_list)):
                    future = executor.submit(process_single_question, headline, article, messages, model_name)
                    tasks.append((idx, future))

                results = [None] * len(tasks)
                for idx, future in tasks:
                    result = future.result()
                    results[idx] = result

                formatted_msg.extend(results)

        c_utils.write_json_file(formatted_msg, output_filename)
        c_utils.write_jsonl_file(formatted_msg, output_filename + "l")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Inference Script")
    parser.add_argument("--domain", type=str, default="finance", help="dataset domain")
    parser.add_argument("--max_batch_size", type=int, default=max_batch, help="batch_size")
    parser.add_argument("--lang", type=str, default="korean", help="lang")
    parser.add_argument("--prompt_type", type=str, default="qa_pair_with_re", help="prompt type")
    parser.add_argument("--model_name", type=str, default=model)
    parser.add_argument("--num_of_workers", type=int, default=num_of_workers)
    parser.add_argument("--num_of_data", type=int, default=num_of_data)

    args = parser.parse_args()
    make_question(args)
