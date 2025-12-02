import argparse
import json
import os
from math import ceil

import pandas as pd
import utils
from datasets import Dataset, load_dataset
from generation_prompts import PROMPTS
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc", base_url="https://api.upstage.ai/v1"
)
model = "upstage/solar-1-mini-chat"

batch_size = 1


def normalize_qa_column(item):
    """
    'low', 'mid', 'high' 컬럼 내용을 {'questions': [...]} 형식으로 정규화
    """
    if isinstance(item, dict):
        if "questions" in item and isinstance(item["questions"], list):
            # 이미 예상 형식: {'questions': [...]}
            return item
        elif "question" in item and "reasoning_effort" in item:
            # 단일 QA 딕셔너리이므로 {'questions': [item]}로 래핑
            return {"questions": [item]}
    elif item is None:
        # None 값을 정상적으로 처리
        return {"questions": []}

    # 문자열 표현 구문 분석 시도
    if isinstance(item, str):
        try:
            parsed_item = json.loads(item)
            if isinstance(parsed_item, dict):
                if "questions" in parsed_item and isinstance(
                    parsed_item["questions"], list
                ):
                    return parsed_item
                elif "question" in parsed_item and "reasoning_effort" in parsed_item:
                    return {"questions": [parsed_item]}
            elif isinstance(parsed_item, list):
                # 직접 QA 목록인 경우 'questions' 아래에 래핑
                return {"questions": parsed_item}
        except (json.JSONDecodeError, TypeError):
            pass  # 유효한 JSON 문자열이 아니거나 구문 분석할 수 없음

    return {
        "questions": []
    }  # 처리할 수 없거나 예상치 못한 형식인 경우 기본적으로 빈 목록을 반환


def main(args):
    model_name = args.model_name
    PROMPT = PROMPTS[args.prompt_type][args.lang]

    dataset_path = f"sample_questions/2.{args.domain}_merged.jsonl"
    print(f"데이터셋 로드 중...")

    raw_data = utils.load_jsonl_file(dataset_path)
    df_raw = pd.DataFrame(raw_data)

    # 일관된 구조를 보장하기 위해 'low', 'mid', 'high' 컬럼을 정규화
    for col in ["low", "mid", "high"]:
        df_raw[col] = df_raw[col].apply(normalize_qa_column)

    # pandas DataFrame을 HuggingFace Dataset으로 변환
    dataset = Dataset.from_pandas(df_raw)

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
