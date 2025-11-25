PROMPTS = {
    "answer_generation_short": {
        "korean": {
            "system": """
                      당신은 {domain_name} 분야의 전문가입니다. 
                      다음은 질문의 맥락을 이해하기 위한 원문입니다:
                      ---
                      {input_text}
                      ---

                      질문에 대한 정답만 간결하고 명확하게 제시하세요.
                      불필요한 설명이나 부연 없이 핵심만 답변하세요.
                      """,
            "user": "질문: {question}",
        },
        "english": {
            "system": """
                      You are an expert in the field of {domain_name}.
                      Below is the source text to help you understand the context of the question:
                      ---
                      {input_text}
                      ---

                      Provide only the direct and concise answer to the question.
                      Do not include explanations, reasoning, or extra details.
                      """,
            "user": "Question: {question}",
        },
    },
    ###################
    # 질문 생성 프롬프트
    ###################
    "qa_pair_with_re": {
        "korean": {
            "system": """
                      당신은 {domain_name} 분야의 RAG 데이터셋을 위한 **질문 생성기**입니다.
                      {domain_name} 관련 문맥 발췌문(시드 텍스트)이 제공됩니다. 
                      이 텍스트는 **해당 도메인 지식에 대한 질문을 만들기 위한 참고 자료**로만 사용됩니다.

                      질문은 텍스트의 특정 문장을 그대로 찾거나 단순히 재현하는 것이 아니라,
                      텍스트가 암시하는 **도메인 지식**, **개념 간 관계**, **핵심 원리나 응용 사례**에 대한 실제 사용자의 질문처럼 구성되어야 합니다.
                      텍스트는 문제 해결 시 직접 주어지지 않는다고 가정하고, 
                      이를 기반으로 해당 분야의 일반적이고 교육적 가치가 높은 질문을 만들어주세요.

                      요구사항:
                      1) 질문은 발췌문에서 유추할 수 있는 **도메인 지식, 핵심 개념, 원리, 응용**을 바탕으로 작성하세요.
                        - 단순히 텍스트에 명시된 사실을 묻지 마세요.
                        - 텍스트가 암시하는 개념적, 이론적, 또는 실무적 관점을 이끌어내세요.
                      2) 질문 개수는 0~3개입니다.
                        - 발췌문에 유의미한 도메인 단서가 부족하면 `"questions": []`를 반환하세요.
                        - 충분한 단서가 있다면 최대 3개의 질문을 작성하세요.
                      3) 각 질문에는 추론 복잡도를 {{"low","medium","high"}} 중 하나로 라벨링하세요.
                        - 해당 지표는 gpt-oss 모델의 reasoning effort를 의미합니다.
                      4) 출력 형식(반드시 준수):
                      아래 **정확한** JSON 스키마로만 출력하세요. JSON 이외의 텍스트를 포함하지 마세요.
                      {{
                        "questions": [
                          {{"question": "...", "reasoning_effort": "low|medium|high"}},
                          ...
                        ]
                      }}

                      제약:
                      - 질문 문장은 반드시 **한국어**로 작성하세요. (단, JSON 키 이름은 영어 그대로 유지)
                      - reasoning_effort는 반드시 "low", "medium", "high" 중 하나로만 설정하세요.
                      """,
            "user": """
                      다음은 {domain_name} 도메인과 관련된 발췌문입니다. 
                      이 텍스트를 참고하여, 도메인 전반의 개념이나 응용에 대한 실제 사용자의 질문을 만들어주세요. 
                      단, 문제를 풀 때 이 텍스트가 주어지지 않는다고 가정하고,
                      텍스트를 바탕으로 **도메인 일반 지식 기반의 질문**을 생성해주세요.
                      정확한 json 형식을 따라주세요.

                      {input_text}
                      """,
        },
        "english": {
            "system": """
                      You are a **question generator** for a RAG dataset in the field of {domain_name}.
                      A contextual excerpt (seed text) related to {domain_name} is provided below. 
                      This text is **not meant to be referenced directly**, but serves as background material 
                      to help you infer **domain knowledge, key principles, or conceptual relationships**.

                      The generated questions should not depend on locating information within the text itself.
                      Instead, assume that the text will NOT be available during question answering.
                      Your task is to create realistic, educational, and domain-relevant questions 
                      based on the kind of knowledge the text implies or discusses.

                      Requirements:
                      1) Generate questions inspired by the excerpt that reflect **domain knowledge, conceptual understanding, or real-world application**.
                        - Do NOT create literal or text-dependent questions.
                        - Focus on underlying theories, relationships, or reasoning patterns implied by the text.
                      2) Generate 0–3 questions.
                        - If the excerpt lacks enough domain cues, return `"questions": []`.
                        - Otherwise, create up to 3 questions.
                      3) Label each question with a reasoning complexity level from {{"low", "medium", "high"}}:
                        - This means gpt-oss model's reasoning effort.
                      4) Output format (must follow exactly):
                      Output ONLY valid JSON strictly matching the schema below. Do not include any text outside of JSON.
                      {{
                        "questions": [
                          {{"question": "...", "reasoning_effort": "low|medium|high"}},
                          ...
                        ]
                      }}

                      Constraints:
                      - Write all questions **English** and also keep JSON key names in English.
                      - The `reasoning_effort` field must be one of "low", "medium", or "high".""",
            "user": """
                    Below is a contextual excerpt from the {domain_name} domain.
                    Use it as background inspiration to create domain-general, concept-driven questions — 
                    not literal questions tied to the specific text. 
                    Assume that the text itself will NOT be provided during question answering.

                    Follow the exact JSON format.

                    {input_text}
                    """,
        },
    },
}
