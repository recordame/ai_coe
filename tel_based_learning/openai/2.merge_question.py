import json
import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def merge_finance_question():
    # 전문가 레벨
    expert_levels = ["low", "mid", "high"]

    # 전문가별 질문 취합
    all_data = {}
    for level in expert_levels:
        file_path = f"sample_questions/finance_{level}.json"

        with open(file_path, "r", encoding="utf-8") as f:
            all_data[level] = json.load(f)

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
    output_path = "sample_questions/finance_merged.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    print(f"\n병합 {len(merged_results)} articles")
    print(f"병합 결과 저장: {output_path}")


if __name__ == "__main__":
    merge_finance_question()
