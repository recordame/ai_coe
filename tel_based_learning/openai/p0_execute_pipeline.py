import argparse
import p1_make_question
import p2_merge_question
import p3_evaluate_question
import p4_generate_answer
import pandas as pd
import os
import py7zr

from datasets import load_dataset

pipeline = "Pipeline-0"

# solar-pro2
# upstage/solar-1-mini-chat
model = "upstage/solar-1-mini-chat"
num_of_data = 10
max_batch = 1

def main(args, dataset):
    print(args)

    print(f"[{pipeline}] 기사 및 전문가 등급별 질문 생성/평가, 질문별 답변 생성 파이프라인 시작")
    print(f"[{pipeline}] 총 {len(dataset)}기사 처리 예정")

    print("="*40)
    p1_make_question.make_question(args, dataset)
    print("="*40)
    p2_merge_question.merge_question(args)
    print("="*40)
    p3_evaluate_question.evaluate_question(args)
    print("="*40)
    p4_generate_answer.generate_answer(args)
    print(f"[{pipeline}] 기사 및 전문가 등급별 질문 생성/평가, 질문별 답변 생성 파이프라인 종료")

if __name__ == "__main__":
    csv_path = "../bloomberg_financial_news_120k.csv"
    archive_path = "../bloomberg_financial_news_120k.csv.7z"

    if not os.path.exists(csv_path) and os.path.exists(archive_path):
        print(f"[Pipeline-0] 압축 파일 해제")
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path="..")
        print(f"[Pipeline-0] 압축 해제 완료")

    dataset = load_dataset("csv", data_files=csv_path, split="train").select(range(num_of_data))
    
    parser = argparse.ArgumentParser(description="OpenAI-Based QA Generator")
    parser.add_argument("--domain", type=str, default="finance", help="dataset domain")
    parser.add_argument("--lang", type=str, default="korean", help="lang")
    parser.add_argument("--model_name", type=str, default=model)
    parser.add_argument("--num_of_data", type=int, default=num_of_data)
    parser.add_argument("--max_batch_size", type=int, default=max_batch, help="batch_size")

    args = parser.parse_args()

    generation_date = pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d')
    escaped_model_name = args.model_name.replace(':', '-').replace('/', '-')
    working_dir = f"dataset/{generation_date}/{escaped_model_name}/"
    os.makedirs(os.path.dirname(working_dir), exist_ok=True)
    os.chdir(working_dir) 

    main(args, dataset)
