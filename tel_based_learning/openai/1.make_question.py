import argparse
import json
import os

from datasets import load_dataset
from openai import OpenAI
import py7zr
from tqdm import tqdm

from generation_prompts import PROMPTS
import utils

model = "upstage/solar-1-mini-chat"

# 아래 전문가 레벨을 조절하여 질문 생성
# low:  초급인력 (퍼플렉시티 기준: 중급 상단~고급 초입)
# mid:  중급인력 (고급 독해자)
# high: 고급인력 (최상위 고급, 즉 전문가 수준)
expert_levels = ["low", "mid", "high"]
num_of_data = 10
max_batch = 1

client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc", 
    base_url="https://api.upstage.ai/v1"
)


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


def main(args):
    model_name = args.model_name
    prompt = PROMPTS[args.prompt_type][args.lang]

    csv_path = "../bloomberg_financial_news_120k.csv"
    archive_path = "../bloomberg_financial_news_120k.csv.7z"

    if not os.path.exists(csv_path) and os.path.exists(archive_path):
        print(f"압축 파일 해제 중...")
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path="..")
        print(f"압축 해제 완료.")

    print(f"데이터셋 로드 중...")
    dataset = load_dataset("csv", data_files=csv_path, split="train").select(range(args.num_of_data))
    print(f"데이터셋 로드 완료. 총 {len(dataset)}개 처리 예정.")

    for expert_level in expert_levels:
        output_filename = (f"sample_questions/1.{args.domain}_{expert_level}_{args.num_of_data}.json")
        processed_dataset = dataset

        formatted_msg = []

        for batch in tqdm(
            processed_dataset.iter(batch_size=args.max_batch_size),
            total=args.num_of_data,
            desc=f"전문가 레벨 {expert_level} 질문 생성중",
        ):
            messages_list = batch_chat_template(batch, prompt, expert_level, args)

            for headline, article, messages in zip(batch["Headline"], batch["Article"], messages_list):
                while True:
                    print(headline[:20])
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
                    except Exception as e:
                        print(e)
                        continue

        utils.write_json_file(formatted_msg, output_filename)
        utils.write_jsonl_file(formatted_msg, output_filename + "l")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Inference Script")
    parser.add_argument("--domain", type=str, default="finance", help="dataset domain")
    parser.add_argument("--max_batch_size", type=int, default=max_batch, help="batch_size")
    parser.add_argument("--lang", type=str, default="korean", help="lang")
    parser.add_argument("--prompt_type", type=str, default="qa_pair_with_re", help="prompt type")
    parser.add_argument("--model_name", type=str, default=model)
    parser.add_argument("--num_of_data", type=int, default=num_of_data, help="number of data to process")

    args = parser.parse_args()
    main(args)
