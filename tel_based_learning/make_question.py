import os

os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"

import argparse
import json
import re
from math import ceil

from datasets import Dataset, load_dataset
from generation_prompts import PROMPTS
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main(args):
    # --- 1. 모델 및 토크나이저 설정 ---
    model_name = "openai/gpt-oss-120b"
    PROMPT = PROMPTS[args.prompt_type][args.lang]

    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(max_tokens=4096, skip_special_tokens=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- 2. 데이터셋 로드 ---
    print(f"데이터셋 로드 중...")

    dataset = load_dataset(
        "genloop/bloomberg_financial_news_120k", split="train"
    ).select(range(128))

    print(f"데이터셋 로드 완료. 총 {len(dataset)}개 처리 예정.")
    output_filename = f"results_{args.domain}.jsonl"
    processed_dataset = dataset

    def batch_chat_template(batch, tokenizer, args):
        messages = []
        system_prompt = PROMPT["system"].format(domain_name=args.domain)

        for text in batch["Article"]:
            user_prompt = PROMPT["user"].format(
                domain_name=args.domain, input_text=text
            )
            messages.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return input_text

    with open(output_filename, "a", encoding="utf-8") as f:
        for batch in tqdm(
            processed_dataset.iter(batch_size=args.max_batch_size),
            total=ceil(len(processed_dataset) / args.max_batch_size),
        ):
            input_prompts = batch_chat_template(batch, tokenizer, args)

            outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

            for idx, output in zip(batch["Headline"], outputs):
                f.write(
                    json.dumps(
                        {"Headline": idx, "generated_text": output.outputs[0].text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()

    with open(output_filename, "r", encoding="utf-8") as f:
        files = [json.loads(line) for line in f]

    new_files = []

    for text, file in zip(dataset["Article"], files):
        file["Article"] = text
        new_files.append(file)

    files = new_files

    def fix_latex_backslashes(text: str) -> str:
        patterns = [
            (r"\\\(", r"\\\\("),
            (r"\\\)", r"\\\\)"),
            (r"\\sqrt", r"\\\\sqrt"),
            (r"\\frac", r"\\\\frac"),
            (r"\\_", r"\\\\_"),
        ]

        for pat, repl in patterns:
            text = re.sub(pat, repl, text)

        return text

    def is_json(row):
        try:
            # 1️⃣ 기본 파싱 시도
            sample = json.loads(row["generated_text"].split("<|message|>")[-1])
            qa_list = sample["questions"]

            if not qa_list:
                return False

            for qa_pair in qa_list:
                if not all(key in qa_pair for key in ("question", "reasoning_effort")):
                    return False
        except Exception:
            # 2️⃣ 실패 시 백슬래시 보정 후 재시도
            raw_text = row["generated_text"].split("<|message|>")[-1]
            fixed_text = fix_latex_backslashes(raw_text)

            try:
                sample = json.loads(fixed_text)
                qa_list = sample["questions"]

                if not qa_list:
                    return False
            except Exception:
                # 3️⃣ 여전히 실패하면 True 리턴
                return False

        return True

    ds_filtered = Dataset.from_list(files).filter(is_json)

    def parsing_q_list(row):
        try:
            # 1️⃣ 기본 파싱 시도
            sample = json.loads(row["generated_text"].split("<|message|>")[-1])
            qa_list = sample["questions"]

            return {"qa_list": json.dumps(qa_list, ensure_ascii=False)}
        except Exception:
            # 2️⃣ 실패 시 백슬래시 보정 후 재시도
            raw_text = row["generated_text"].split("<|message|>")[-1]
            fixed_text = fix_latex_backslashes(raw_text)
            sample = json.loads(fixed_text)
            qa_list = sample["questions"]

            return {"qa_list": json.dumps(qa_list, ensure_ascii=False)}

    ds_parsed = ds_filtered.map(parsing_q_list, num_proc=32)
    ds_parsed.save_to_disk("temp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Inference Script")
    parser.add_argument("--domain", type=str, required=True, help="dataset domain")
    parser.add_argument("--max_batch_size", type=int, required=True, help="batch_size")
    parser.add_argument("--lang", type=str, default="korean", help="lang")
    parser.add_argument(
        "--prompt_type", type=str, default="qa_pair_with_re", help="lang"
    )

    args = parser.parse_args()
    main(args)
