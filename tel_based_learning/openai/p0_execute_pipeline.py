import argparse

import p1_make_question
import p2_merge_question
import p3_evaluate_question
import p4_generate_answer

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

model = "gpt-oss:120b"
num_of_data = 120000
max_batch = 1
num_of_workers = 1


def main(args):
    print(args)
    p1_make_question.make_question(args)
    p2_merge_question.merge_question(args)
    p3_evaluate_question.evaluate_question(args)
    p4_generate_answer.generate_answer(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI-Based QA Generator")
    parser.add_argument("--domain", type=str, default="finance", help="dataset domain")
    parser.add_argument("--lang", type=str, default="korean", help="lang")
    parser.add_argument("--prompt_type", type=str, default="qa_pair_with_re", help="prompt type")
    parser.add_argument("--model_name", type=str, default=model)
    parser.add_argument("--num_of_data", type=int, default=num_of_data)
    parser.add_argument("--max_batch_size", type=int, default=max_batch, help="batch_size")
    parser.add_argument("--num_of_workers", type=int, default=num_of_workers)

    args = parser.parse_args()
    main(args)
