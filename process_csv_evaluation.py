#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 파일 위협도 평가 스크립트

CSV 파일에서 질문을 읽어서 ThreatEvaluator로 평가하고,
결과를 새로운 컬럼에 추가하여 저장합니다.

사용법:
    python process_csv_evaluation.py --input input.csv --output output.csv --question-column question

필수 파라미터:
    --input: 입력 CSV 파일 경로
    --output: 출력 CSV 파일 경로

선택 파라미터:
    --question-column: 질문이 있는 컬럼명 (기본값: 'question')
    --api-key: OpenAI API 키 (환경변수 OPENAI_API_KEY 사용 가능)
    --model: OpenAI 모델명 (기본값: gpt-4o)
    --start-row: 시작 행 번호 (기본값: 0)
    --end-row: 종료 행 번호 (기본값: 전체)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "llm_redteam_evaluator"))

from llm_redteam_evaluator.threat_evaluator import ThreatEvaluator


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="CSV 파일의 질문을 읽어서 위협도 평가를 수행합니다."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="입력 CSV 파일 경로"
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="출력 CSV 파일 경로"
    )

    parser.add_argument(
        "--question-column",
        "-q",
        default="attack_question",
        help="질문이 있는 컬럼명 (기본값: attack_question)"
    )

    parser.add_argument(
        "--api-key",
        help="OpenAI API 키 (환경변수 OPENAI_API_KEY 사용 가능)"
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI 모델명 (기본값: gpt-4o)"
    )

    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="시작 행 번호 (기본값: 0)"
    )

    parser.add_argument(
        "--end-row",
        type=int,
        help="종료 행 번호 (기본값: 전체)"
    )

    return parser.parse_args()


def evaluate_csv(
    input_path: str,
    output_path: str,
    question_column: str = "attack_question",
    api_key: str = None,
    model: str = "gpt-4o",
    start_row: int = 0,
    end_row: int = None
):
    """
    CSV 파일의 질문들을 평가하고 결과를 저장합니다.

    Args:
        input_path: 입력 CSV 파일 경로
        output_path: 출력 CSV 파일 경로
        question_column: 질문이 있는 컬럼명
        api_key: OpenAI API 키
        model: OpenAI 모델명
        start_row: 시작 행 번호
        end_row: 종료 행 번호
    """

    print("\n" + "=" * 80)
    print("CSV 위협도 평가 시작")
    print("=" * 80)

    # 1. CSV 파일 읽기
    print(f"\n📂 CSV 파일 로드: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
        print(f"✅ 로드 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
        print(f"   컬럼: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"❌ CSV 파일 로드 오류: {e}")
        return

    # 2. 질문 컬럼 확인
    if question_column not in df.columns:
        print(f"❌ 오류: '{question_column}' 컬럼을 찾을 수 없습니다.")
        print(f"   사용 가능한 컬럼: {', '.join(df.columns.tolist())}")
        return

    # 3. 행 범위 설정
    if end_row is None:
        end_row = len(df)
    else:
        end_row = min(end_row, len(df))

    print(f"\n📊 평가 범위: {start_row}행 ~ {end_row}행 (총 {end_row - start_row}개)")

    # 4. ThreatEvaluator 초기화
    print(f"\n🤖 ThreatEvaluator 초기화 (모델: {model})")
    try:
        evaluator = ThreatEvaluator(api_key=api_key, model=model)
        print("✅ 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        return

    # 5. 결과 컬럼 추가
    result_columns = [
        "threat_score",
        "risk_level",
        "syntactic_score",
        "syntactic_risk_level",
        "semantic_score",
        "semantic_risk_level",
        "recommendations",
        "evaluation_timestamp"
    ]

    for col in result_columns:
        if col not in df.columns:
            df[col] = None

    # 6. 각 행 평가
    print(f"\n🔍 위협도 평가 시작...")
    print("-" * 80)

    success_count = 0
    error_count = 0

    for idx in tqdm(range(start_row, end_row), desc="평가 진행"):
        question = df.loc[idx, question_column]

        # 질문이 비어있으면 스킵
        if pd.isna(question) or not str(question).strip():
            print(f"\n⚠️  행 {idx}: 질문이 비어있음, 스킵")
            continue

        try:
            # 평가 수행
            result = evaluator.evaluate_question(str(question))

            # 결과 저장
            df.loc[idx, "threat_score"] = result["final_threat_score"]
            df.loc[idx, "risk_level"] = result["risk_level"]
            df.loc[idx, "syntactic_score"] = result["syntactic_evaluation"]["threat_score"]
            df.loc[idx, "syntactic_risk_level"] = result["syntactic_evaluation"]["risk_level"]
            df.loc[idx, "semantic_score"] = result["semantic_evaluation"]["threat_score"]
            df.loc[idx, "semantic_risk_level"] = result["semantic_evaluation"]["risk_level"]
            df.loc[idx, "recommendations"] = " | ".join(result["recommendations"])
            df.loc[idx, "evaluation_timestamp"] = result["timestamp"]

            success_count += 1

            # 진행 상황 출력
            if (idx - start_row + 1) % 10 == 0:
                print(f"\n✅ 진행: {idx - start_row + 1}/{end_row - start_row} 완료 (성공: {success_count}, 오류: {error_count})")

        except Exception as e:
            error_count += 1
            print(f"\n❌ 행 {idx} 평가 오류: {e}")

            # 오류 정보 저장
            df.loc[idx, "threat_score"] = -1
            df.loc[idx, "risk_level"] = "ERROR"
            df.loc[idx, "recommendations"] = f"평가 오류: {str(e)}"
            df.loc[idx, "evaluation_timestamp"] = datetime.now().isoformat()

    # 7. 결과 저장
    print("\n" + "=" * 80)
    print("📁 결과 저장 중...")

    try:
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 저장 완료: {output_path}")
    except Exception as e:
        print(f"❌ 저장 오류: {e}")
        return

    # 8. 요약 정보 출력
    print("\n" + "=" * 80)
    print("📊 평가 요약")
    print("=" * 80)
    print(f"총 평가: {end_row - start_row}개")
    print(f"성공: {success_count}개")
    print(f"오류: {error_count}개")

    if success_count > 0:
        print("\n위험 수준 분포:")
        risk_level_counts = df.loc[start_row:end_row-1, "risk_level"].value_counts()
        for level, count in risk_level_counts.items():
            print(f"  {level}: {count}개 ({count/success_count*100:.1f}%)")

        print("\n평균 점수:")
        valid_scores = df.loc[start_row:end_row-1, "threat_score"]
        valid_scores = valid_scores[valid_scores >= 0]
        if len(valid_scores) > 0:
            print(f"  최종 위협도: {valid_scores.mean():.2f}")
            print(f"  최고 점수: {valid_scores.max():.2f}")
            print(f"  최저 점수: {valid_scores.min():.2f}")

    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)


def main():
    """메인 함수"""
    args = parse_args()

    # API 키 확인
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 오류: OpenAI API 키가 필요합니다.")
        print("   --api-key 옵션을 사용하거나 OPENAI_API_KEY 환경변수를 설정하세요.")
        sys.exit(1)

    # 평가 실행
    evaluate_csv(
        input_path=args.input,
        output_path=args.output,
        question_column=args.question_column,
        api_key=api_key,
        model=args.model,
        start_row=args.start_row,
        end_row=args.end_row
    )


if __name__ == "__main__":
    main()
