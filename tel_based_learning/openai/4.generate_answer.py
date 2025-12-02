import argparse
import json
from math import ceil
import os

from tqdm import tqdm
from datasets import load_dataset
from generation_prompts import PROMPTS
from openai import OpenAI
import utils


client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc", base_url="https://api.upstage.ai/v1"
)
model = "upstage/solar-1-mini-chat"

batch_size = 1


def main(args):
    model_name = args.model_name
    PROMPT = PROMPTS[args.prompt_type][args.lang]

    dataset_path = f"sample_questions/2.{args.domain}_merged.jsonl"
    print(f"데이터셋 로드 중...")

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    print(f"데이터셋 로드 완료. 총 {len(dataset)}개 문서 처리 예정.")

    output_filename = f"sample_questions/4.{args.domain}_QA.json"

    result_docs = []
    for batch in tqdm(
        dataset.iter(batch_size=args.max_batch_size),
        total=ceil(len(dataset) / args.max_batch_size),
        desc="문서 처리 중",
    ):
        batch_size = len(batch["headline"])

        for idx in range(batch_size):
            headline = batch["headline"][idx]
            article_text = batch["article"][idx]

            result_doc = {
                "headline": headline,
                "article": article_text,
                "low": batch["low"][idx].copy(),
                "mid": batch["mid"][idx].copy(),
                "high": batch["high"][idx].copy(),
            }

            for level in ["low", "mid", "high"]:
                if "questions" in result_doc[level]:
                    for qa in result_doc[level]["questions"]:
                        while True:
                            system_prompt = PROMPT["system"].format(
                                domain_name=args.domain, input_text=article_text
                            )
                            user_prompt = PROMPT["user"].format(question=qa["question"])

                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ]

                            try:
                                response = client.chat.completions.create(
                                    model=model_name,
                                    messages=messages,
                                    max_tokens=8192,
                                    temperature=0.1,
                                )

                                generated_text = response.choices[0].message.content
                                qa["answer"] = generated_text
                                break
                            except:
                                continue

            result_docs.append(result_doc)

    utils.write_json_file(result_docs, output_filename)
    utils.write_jsonl_file(result_docs, output_filename + "l")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Answer Generation Script")
    parser.add_argument("--domain", type=str, default="finance", help="dataset domain")
    parser.add_argument("--max_batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--lang", type=str, default="korean", help="lang")
    parser.add_argument(
        "--prompt_type", type=str, default="answer_generation", help="prompt type"
    )
    parser.add_argument("--model_name", type=str, default=model)

    args = parser.parse_args()
    main(args)
