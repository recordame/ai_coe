import argparse
import pandas as pd
import common.utils as utils
import os

pipeline = "Pipeline-2"
model = "upstage/solar-1-mini-chat"
num_of_data = 1


def merge_question(args):
    print(f"[Pipeline-2] 질문 병합 시작")

    # 전문가 레벨
    expert_levels = ["low_level_expert", "mid_level_expert", "high_level_expert"]

    # 전문가별 질문 취합
    all_data = {}

    for expert_level in expert_levels:
        file_path = f"P1-{args.domain}_{expert_level}_{args.num_of_data}_articles-question.jsonl"
        all_data[expert_level] = utils.load_jsonl_file(file_path)

        print(f"[{pipeline}] {expert_level}의 {len(all_data[expert_level])}개 기사 로드({file_path})")

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
    output_path = f"P2-{args.domain}_all_level_expert_{args.num_of_data}_articles-question_merged.json"

    utils.write_json_file(merged_results, output_path)
    utils.write_jsonl_file(merged_results, output_path + "l")

    print(f"[{pipeline}] 병합 결과 저장({output_path})")
    print(f"[{pipeline}] 질문 병합 완료")