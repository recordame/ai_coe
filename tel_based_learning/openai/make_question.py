import argparse
import json
import re
from math import ceil

from datasets import Dataset, load_dataset
from generation_prompts import PROMPTS
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc", base_url="https://api.upstage.ai/v1"
)


def main(args):
    # --- 1. OpenAI 클라이언트 설정 ---
    model_name = args.model_name

    PROMPT = PROMPTS[args.prompt_type][args.lang]

    # --- 2. 데이터셋 로드 ---
    print(f"데이터셋 로드 중...")

    dataset = load_dataset(
        "csv", data_files="../bloomberg_financial_news_120k.csv", split="train"
    ).select(range(2))

    print(f"데이터셋 로드 완료. 총 {len(dataset)}개 처리 예정.")
    output_filename = f"results_{args.domain}.json"
    processed_dataset = dataset

    def batch_chat_template(batch, args):
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

        return messages

    formatted_msg = []

    with open(output_filename, "w", encoding="utf-8") as f:
        for batch in tqdm(
            processed_dataset.iter(batch_size=args.max_batch_size),
            total=ceil(len(processed_dataset) / args.max_batch_size),
        ):
            messages_list = batch_chat_template(batch, args)

            # OpenAI API를 사용하여 각 메시지에 대해 생성
            for idx, messages in zip(batch["Headline"], messages_list):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.7,
                    )
                    generated_text = response.choices[0].message.content

                    try:
                        json.loads(generated_text)
                    except:
                        continue

                    formatted_msg.append(
                        {
                            "Headline": idx,
                            "generated_text": json.loads(generated_text),
                        }
                    )
                except Exception as e:
                    print(f"Error processing {idx}: {e}")
                    continue

        json.dump(formatted_msg, f, ensure_ascii=False, indent=2)

    with open(output_filename, "r", encoding="utf-8") as f:
        files = json.load(f)

    new_files = []

    for text, file in zip(dataset["Article"], files):
        file["Article"] = text
        new_files.append(file)

    files = new_files
    ds_filtered = Dataset.from_list(files)

    def parsing_q_list(row):
        generated_text = row["generated_text"]
        qa_list = generated_text.get("questions", [])

        return {"qa_list": json.dumps(qa_list, ensure_ascii=False, indent=2)}

    ds_parsed = ds_filtered.map(parsing_q_list, num_proc=32)

    ds_parsed.save_to_disk("temp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Inference Script")
    parser.add_argument("--domain", type=str, default="finance", help="dataset domain")
    parser.add_argument("--max_batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--lang", type=str, default="korean", help="lang")
    parser.add_argument(
        "--prompt_type", type=str, default="qa_pair_with_re", help="prompt type"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="upstage/solar-1-mini-chat",
        help="upstage/solar-1-mini-chat",
    )

    args = parser.parse_args()
    main(args)
