import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import requests
from tqdm import tqdm

import utils

model = "gpt-oss:120b"
num_of_data = 10
num_workers = 4


def call_ollama(model="gpt-oss:120b", messages=None, stream=False):
    url = "http://ollama:11434/api/chat"
    with requests.Session() as session:
        response = session.post(
            url, json={"model": model, "messages": messages, "stream": stream}
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def create_judge_prompt(article, headline, reasoning_effort, questions_dict):
    # LLM Judge를 위한 프롬프트 생성

    prompt = f"""당신은 금융 전문가이자 교육 평가 전문가입니다. 주어진 기사에 대해 여러 세 익명의 도메인 관련자 부터 생성된 질문들을 평가하고, 가장 전문가답고 질 높은 질문을 선택해야 합니다.
                **기사 제목**: {headline}

                **기사 본문**: 
                {article}

                **익명의 도메인 관련자**:
                1. **low**: {questions_dict.get('low')}
                2. **mid**: {questions_dict.get('mid')}
                3. **high**: {questions_dict.get('high')}

                위 질문들 중에서 다음 기준으로 가장 우수한 질문을 선택하세요
                1. 기사 내용과의 관련성
                2. 질문의 금융 도메인의 적합성
                3. 전문성과 통찰력

                **응답 형식** (반드시 JSON 형식으로만 응답):
                {{
                    "best_expert": "low|mid|high 중 하나",
                    "reason": "선택한 이유를 구체적으로 설명 (2-3문장)"
                }}
            """

    return prompt


def evaluate_questions(article_data):
    # 하나의 기사에 대한 모든 질문을 평가

    headline = article_data["headline"]
    article = article_data["article"]

    best_questions = []

    # 각 reasoning_effort 레벨에 대해 평가
    for reasoning_effort in ["low", "mid", "high"]:
        # 각 전문가 레벨의 질문 수집
        questions_dict = {}

        for expert_level in ["low", "mid", "high"]:
            if expert_level in article_data:
                # 해당 expert_level의 questions 리스트에서 reasoning_effort에 맞는 질문 찾기
                questions_list = article_data[expert_level].get("questions", [])
                for q in questions_list:
                    try:
                        if q.get("reasoning_effort") == reasoning_effort:
                            questions_dict[expert_level] = q.get("question", "")
                            break
                    except:
                        continue

        # 질문이 없으면 스킵
        if not questions_dict:
            continue

        # LLM Judge 호출
        prompt = create_judge_prompt(
            article, headline, reasoning_effort, questions_dict
        )

        while True:
            result_text = call_ollama(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 금융 전문가이자 교육 평가 전문가입니다. 질문의 질을 평가하고 JSON 형식으로만 응답합니다."},
                    {"role": "user", "content": prompt},
                ],
            ).strip()

            # JSON 파싱 시도
            try:
                result = json.loads(result_text)
                best_expert = result.get("best_expert", "")
                reason = result.get("reason", "")

                # 선택된 전문가의 질문 가져오기
                selected_question = questions_dict.get(best_expert, "")

                best_questions.append(
                    {
                        "reasoning_effort": reasoning_effort,
                        "best_expert": best_expert,
                        "question": selected_question,
                        "reason": reason,
                    }
                )

                break
            except:
                # 파싱 실패 시 응답 재생성
                continue

    return {"headline": headline, "article": article, "best_questions": best_questions}


def main(args):
    # 입력 파일 경로
    input_file = f"sample_questions/2.{args.domain}_merged_{args.num_of_data}.jsonl"
    output_detail = (
        f"sample_questions/3.{args.domain}_evaluated_{args.num_of_data}.json"
    )
    output_summary = f"sample_questions/3.{args.domain}_summary_{args.num_of_data}.json"

    # 입력 파일 로드
    print(f"입력 파일 로드 중: {input_file}")
    data = utils.load_jsonl_file(input_file)

    print(f"총 {len(data)}개의 기사를 평가합니다.")

    # 각 기사에 대해 평가 수행
    evaluated_results = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(evaluate_questions, article_data) for article_data in data
        ]

        for future in tqdm(futures, total=len(data), desc="기사 평가 중"):
            result = future.result()
            evaluated_results.append(result)

    print(f"\n결과 저장 중: {output_detail}")
    utils.write_json_file(evaluated_results, output_detail)
    utils.write_jsonl_file(evaluated_results, output_detail + "l")
    print(f"평가 완료! 결과가 {output_detail}에 저장되었습니다.")

    # 결과 요약 출력
    summary_table = {
        "low": {"low": 0, "mid": 0, "high": 0},
        "mid": {"low": 0, "mid": 0, "high": 0},
        "high": {"low": 0, "mid": 0, "high": 0},
    }

    print("\n=== 평가 결과 요약 ===")
    i = 0
    for i, result in enumerate(evaluated_results, 1):
        print(f"\n{i}. {result['headline'][:50]}...")
        for bq in result["best_questions"]:
            print(f"  - {bq['reasoning_effort']}: {bq['best_expert']} 전문가 선택")

            summary_table[bq["reasoning_effort"]][bq["best_expert"]] += 1

    i += 1
    print(f"\n{i}. 추론 난이도별 전문가 선택 결과")
    for reasoning_effort in summary_table.keys():
        print(f"  - {reasoning_effort}_reasoning_effort에 대한 전문가별 질문 선택 횟수")

        for expert_level in summary_table[reasoning_effort]:
            print(f"   . {expert_level} 레벨 전문가: {summary_table[reasoning_effort][expert_level]}회 선택")

        print("\n")

    utils.write_json_file(summary_table, output_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Question Evaluation Script")
    parser.add_argument("--domain", type=str, default="finance", help="dataset domain")
    parser.add_argument("--num_workers", type=int, default=num_workers, help="number of parallel workers")
    parser.add_argument("--num_of_data", type=int, default=num_of_data, help="number of articles to evaluate")

    args = parser.parse_args()

    main(args)
