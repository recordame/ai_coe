import json

from tqdm import tqdm

import common.utils as utils

pipeline = "Pipeline-3"


def create_judge_prompt(article, headline, questions_dict):
    prompt = f"""당신은 금융 전문가이자 교육 평가 전문가입니다. 주어진 기사에 대해 여러 세 익명의 도메인 관련자 부터 생성된 질문들을 평가하고, 가장 전문가답고 질 높은 질문을 선택해야 합니다.
                **기사 제목**: {headline}

                **기사 본문**: 
                {article}

                **익명의 도메인 관련자의 질문**:
                1. **low_level_expert**: {questions_dict.get('low_level_expert')}
                2. **mid_level_expert**: {questions_dict.get('mid_level_expert')}
                3. **high_level_expert**: {questions_dict.get('high_level_expert')}

                위 질문들 중에서 다음 기준으로 가장 우수한 질문을 선택하세요
                1. 기사 내용과의 관련성
                2. 질문의 금융 도메인의 적합성
                3. 전문성과 통찰력

                **응답 형식** (반드시 JSON 형식으로만 응답):
                {{
                    "best_expert": "low_level_expert|mid_level_expert|high_level_expert 중 하나",
                    "reason": "선택한 이유를 구체적으로 설명 (2-3문장)"
                }}
            """

    return prompt


def evaluate_questions(article_data, model_name):
    headline = article_data["headline"]
    article = article_data["article"]

    best_questions = []

    for reasoning_effort in ["low", "mid", "high"]:
        questions_dict = {}

        for expert_level in ["low_level_expert", "mid_level_expert", "high_level_expert"]:
            if expert_level in article_data:
                questions_list = article_data[expert_level].get("questions", [])
                for q in questions_list:
                    try:
                        if q.get("reasoning_effort") == reasoning_effort:
                            questions_dict[expert_level] = q.get("question", "")
                            break
                    except:
                        continue

        if not questions_dict:
            continue

        prompt = create_judge_prompt(article, headline, questions_dict)

        while True:
            messages = [
                {
                    "role": "system",
                    "content": """
                                당신은 인지 심리 및 교육 전문가입니다.
                                질문의 품질을 평가하고 JSON 형식으로만 응답합니다.
                                품질은 질문을 통해 학생들이 얼마나 많은것을 배울 수 있는가 입니다.
                               """
                },
                {"role": "user", "content": prompt},
            ]

            result_text = utils.ollama_chat(messages, model_name).strip()

            try:
                result = json.loads(result_text)
                best_expert = result.get("best_expert", "")
                reason = result.get("reason", "")

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
                continue

    return {"headline": headline, "article": article, "best_questions": best_questions}


def evaluate_question(args):
    print(f"[{pipeline}] 질문 평가 시작")

    input_file = f"P2-{args.domain}_all_level_expert_{args.num_of_data}_articles-question_merged.jsonl"
    output_detail = f"P3-{args.domain}_all_level_expert_{args.num_of_data}_articles-question_evaluated.json"
    output_summary = f"P3-{args.domain}_all_level_expert_{args.num_of_data}_articles-evaluation_summary.json"

    print(f"[{pipeline}] 병합된 질문 파일 로드({input_file})")
    data = utils.load_jsonl_file(input_file)

    print(f"[{pipeline}] 총 {len(data)}개 기사에 대한 질문 평가 시작")

    evaluated_results = []

    for article_data in tqdm(data, total=args.num_of_data, desc=f"[{pipeline}] 질문 평가 중"):
        result = evaluate_questions(article_data, args.model_name)
        evaluated_results.append(result)

    utils.write_json_file(evaluated_results, output_detail)
    utils.write_jsonl_file(evaluated_results, output_detail + "l")
    print(f"[{pipeline}] 평가 결과 저장({output_detail})")

    summary_table = {
        "low_reasoning_effort": {"low_level_expert": 0, "mid_level_expert": 0, "high_level_expert": 0},
        "mid_reasoning_effort": {"low_level_expert": 0, "mid_level_expert": 0, "high_level_expert": 0},
        "high_reasoning_effort": {"low_level_expert": 0, "mid_level_expert": 0, "high_level_expert": 0},
    }

    print(f"[{pipeline}] 평가 결과 요약")
    for i, result in enumerate(evaluated_results, 1):
        for bq in result["best_questions"]:
            summary_table[f'{bq["reasoning_effort"]}_reasoning_effort'][bq["best_expert"]] += 1

    print(f"[{pipeline}] 추론 난이도별 전문가 선택 결과")
    for reasoning_effort in summary_table.keys():
        print(f"[{pipeline}]  ■ {reasoning_effort} 질문")

        for expert_level in summary_table[reasoning_effort]:
            print(f"[{pipeline}]    - {expert_level}: {summary_table[reasoning_effort][expert_level]}회 선택")

    utils.write_json_file(summary_table, output_summary)
    print(f"[{pipeline}] 결과 요약 저장({output_summary})")
    print(f"[{pipeline}] 질문 평가 완료")
