import json

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

import common.utils as utils
from common.generation_prompts import PROMPTS

pipeline = "Pipeline-4"


def normalize_qa_column(item):
    if isinstance(item, dict):
        if "questions" in item and isinstance(item["questions"], list):
            return item
        elif "question" in item and "reasoning_effort" in item:
            return {"questions": [item]}
    elif item is None:
        return {"questions": []}

    if isinstance(item, str):
        try:
            parsed_item = json.loads(item)
            if isinstance(parsed_item, dict):
                if "questions" in parsed_item and isinstance(parsed_item["questions"], list):
                    return parsed_item
                elif "question" in parsed_item and "reasoning_effort" in parsed_item:
                    return {"questions": [parsed_item]}
            elif isinstance(parsed_item, list):
                return {"questions": parsed_item}
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "questions": []
    }


def generate_answer(args):
    print(f"[{pipeline}] 질문별 답변 생성 시작")

    model_name = args.model_name
    prompt = PROMPTS["answer_generation"][args.lang]

    dataset_path = f"P2-{args.domain}_all_level_expert_{args.num_of_data}_articles-question_merged.jsonl"
    print(f"[{pipeline}] 질문 데이터 로드({dataset_path})")

    raw_data = utils.load_jsonl_file(dataset_path)
    df_raw = pd.DataFrame(raw_data)

    for col in ["low_level_expert", "mid_level_expert", "high_level_expert"]:
        df_raw[col] = df_raw[col].apply(normalize_qa_column)

    dataset = Dataset.from_pandas(df_raw)

    print(f"[{pipeline}] 총 {len(dataset)}개 기사의 질문에 대한 답변 생성 시작")

    output_filename = f"P4-{args.domain}_all_level_expert_{args.num_of_data}_articles-answer.json"

    result_docs = []
    for batch in tqdm(dataset.iter(batch_size=args.max_batch_size), total=args.num_of_data, desc=f"[{pipeline}] 기사 처리 중"):
        batch_size = len(batch["headline"])

        for idx in range(batch_size):
            headline = batch["headline"][idx]
            article_text = batch["article"][idx]

            result_doc = {
                "headline": headline,
                "article": article_text,
                "low_level_expert": batch["low_level_expert"][idx].copy(),
                "mid_level_expert": batch["mid_level_expert"][idx].copy(),
                "high_level_expert": batch["high_level_expert"][idx].copy(),
            }

            for level in ["low_level_expert", "mid_level_expert", "high_level_expert"]:
                if "questions" in result_doc[level]:
                    for qa in result_doc[level]["questions"]:
                        while True:
                            system_prompt = prompt["system"].format(domain_name=args.domain, input_text=article_text, lang=args.lang)
                            user_prompt = prompt["user"].format(question=qa["question"])

                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ]

                            try:
                                generated_text = utils.ollama_chat(messages, model_name)
                                qa["answer"] = generated_text
                                break
                            except:
                                continue

            result_docs.append(result_doc)

    print(f"[{pipeline}] 질문/답변 저장({output_filename + 'l'})")
    utils.write_json_file(result_docs, output_filename)
    utils.write_jsonl_file(result_docs, output_filename + "l")
    print(f"[{pipeline}] 질문별 답변 생성 완료")
