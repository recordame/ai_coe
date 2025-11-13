import os
os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '1'
os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN_VLLM_V1'

import argparse
import json
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from math import ceil
from generation_prompts import PROMPTS

def main(args):
    # --- 1. 모델 및 토크나이저 설정 ---
    model_name = 'openai/gpt-oss-120b'
    PROMPT = PROMPTS[args.prompt_type][args.lang]
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(max_tokens=8192, skip_special_tokens=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- 2. 데이터셋 로드 ---
    dataset = load_from_disk('temp')
    print(f"데이터셋 로드 완료. 총 {len(dataset)}개 문서 처리 예정.")

    output_filename = f"results_{args.domain}_QA.jsonl"
    
    def batch_chat_template(text, qa_list, tokenizer, args):
        prompts = []

        for qa in qa_list:
            system_prompt = PROMPTS[args.prompt_type][args.lang]['system'].format(
                domain_name=args.domain,
                input_text=text
            )
            user_prompt = PROMPTS[args.prompt_type][args.lang]['user'].format(
                question=qa['question']
            )
            msg = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]

            prompts.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort=qa.get('reasoning_effort')
                )
            )

        return prompts

    
    with open(output_filename, 'a', encoding='utf-8') as f:
        for batch in tqdm(dataset.iter(batch_size=args.max_batch_size),total=ceil(len(dataset) / args.max_batch_size)):
            all_prompts = []
            all_meta = []

            for headline, article_text, qa_list_json in zip(batch['Headline'], batch['Article'], batch['qa_list']):
                qa_list = json.loads(qa_list_json)
                new_qa_list = [qa for qa in qa_list]
                prompts = batch_chat_template(article_text, new_qa_list, tokenizer, args)
                for i, qa in enumerate(new_qa_list):
                    question_id = f"{headline}-q{i}" 
                    
                    all_meta.append({
                        "doc_id": headline,
                        "question": qa["question"],
                        'id': question_id,
                    })

                all_prompts.extend(prompts)

            if all_prompts:
                outputs = llm.generate(all_prompts, sampling_params, use_tqdm=False)

                for meta, output in zip(all_meta, outputs):
                    f.write(json.dumps({
                        "id": meta['id'],
                        "doc_id": meta["doc_id"],
                        "question": meta["question"],
                        "generated_text": output.outputs[0].text
                    }, ensure_ascii=False) + '\n')

                f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="vLLM Inference Script")
    parser.add_argument("--domain", type=str, required=True, help="dataset domain")
    parser.add_argument("--max_batch_size", type=int, required=True, help='batch_size')
    parser.add_argument("--lang", type=str, default='korean', help='lang')
    parser.add_argument("--prompt_type", type=str, default='answer_generation', help='lang')
    
    args = parser.parse_args()
    main(args)