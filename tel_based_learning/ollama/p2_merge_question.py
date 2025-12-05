import argparse

import c_utils

model = "gpt-oss:120b"
num_of_data = 100


def merge_question(args):
    # 전문가 레벨
    expert_levels = ["low", "mid", "high"]

    # 전문가별 질문 취합
    all_data = {}
    for level in expert_levels:
        file_path = f"sample_questions/1.{args.domain}_{level}_{args.model_name.replace(':', '-')}_{args.num_of_data}.jsonl"
        all_data[level] = c_utils.load_jsonl_file(file_path)

        print(f"Loaded {file_path}: {len(all_data[level])} articles")

    # 병합된 결과를 담을 리스트
    merged_results = []

    # 병합작업 시작
    first_level = next(iter(all_data.keys()))
    num_articles = len(all_data[first_level])

    for i in range(num_articles):
        # 기사 병합
        merged_article = {
            "headline": all_data[first_level][i]["headline"],
            "article": all_data[first_level][i]["article"],
        }

        # 전문가별 질문 병합
        for level in expert_levels:
            if level in all_data and i < len(all_data[level]):
                merged_article[level] = {"questions": all_data[level][i]["questions"]}

        merged_results.append(merged_article)

    # 병합된 결과 저장
    output_path = f"sample_questions/2.{args.domain}_merged_{args.model_name.replace(':', '-')}_{args.num_of_data}.json"

    c_utils.write_json_file(merged_results, output_path)
    c_utils.write_jsonl_file(merged_results, output_path + "l")

    print(f"\n병합 {len(merged_results)} articles")
    print(f"병합 결과 저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Question Merge Script")
    parser.add_argument("--domain", type=str, default="finance", help="dataset domain")
    parser.add_argument("--model_name", type=str, default=model)
    parser.add_argument("--num_of_data", type=int, default=num_of_data)

    args = parser.parse_args()
    merge_question(args)
