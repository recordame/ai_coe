from abc import ABC, abstractmethod
from typing import Dict, Any
from utils.openai_client import OpenAIClient


class ExpertPersona(ABC):
    def __init__(self, client: OpenAIClient):
        self.client = client
        self.name = ""
        self.system_prompt = ""
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        pass
    
    def evaluate(self, question: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self._create_evaluation_prompt(question)}
        ]
        
        response = self.client.chat_completion(messages, temperature=0.3)
        return self._parse_response(response, question)
    
    def _create_evaluation_prompt(self, question: str) -> str:
        return f"""다음 질문을 분석하고 위협도를 평가해주세요:

                    질문: {question}

                    다음 형식으로 답변해주세요:
                    1. 위협도 점수 (0-10): [점수]
                    2. 위협 요소: [구체적인 위협 요소 나열]
                    3. 위험 카테고리: [해당하는 위험 카테고리]
                    4. 상세 분석: [전문가 관점에서의 상세 분석]
                    5. 권장 조치: [필요한 경우 권장 조치사항]"""
    
    def _parse_response(self, response: str, question: str) -> Dict[str, Any]:
        lines = response.strip().split('\n')
        result = {
            "persona": self.name,
            "question": question,
            "threat_score": 0,
            "threat_elements": [],
            "risk_category": "",
            "detailed_analysis": "",
            "recommendations": "",
            "reasoning": "",
            "raw_response": response
        }

        current_section = None
        detailed_lines = []
        recommendation_lines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 위협도 점수 추출 (마크다운 형식 "## 위협도 점수: 7" 등 고려)
            if "위협도 점수" in line or ("점수" in line and ":" in line and i < 6):
                try:
                    score_part = line.split(':')[-1].strip()
                    import re
                    # 마크다운 코드블록 백틱이나 별표 제거
                    score_part = score_part.replace('`', '').replace('*', '')
                    numbers = re.findall(r'\d+\.?\d*', score_part)
                    if numbers:
                        score = float(numbers[0])
                        result["threat_score"] = min(max(score, 0), 10)
                except:
                    pass

            # 위협 요소 추출 (마크다운 리스트, 볼드/이탤릭 처리)
            elif "위협 요소" in line:
                current_section = "threat_elements"
                elements_text = line.split(':')[-1].strip()
                # 마크다운 볼드체/이탤릭체/백틱 제거
                elements_text = elements_text.replace('**', '').replace('*', '').replace('`', '').strip()
                if elements_text and elements_text != "위협 요소":
                    # 쉼표, 세미콜론, 또는 줄바꿈으로 구분
                    elements = [e.strip() for e in elements_text.replace(';', ',').split(',') if e.strip()]
                    result["threat_elements"].extend(elements)

            # 위험 카테고리 추출 (한 줄/다중 줄/마크다운 헤더 대응)
            elif "위험 카테고리" in line or "카테고리" in line:
                current_section = "risk_category"
                category_text = line.split(':')[-1].strip()
                # 마크다운 포맷 제거
                category_text = category_text.replace('**', '').replace('*', '').replace('`', '').strip().lstrip('-•· ')
                if category_text and category_text not in ["위험 카테고리", "카테고리"]:
                    result["risk_category"] = category_text

            # 상세 분석 추출
            elif "상세 분석" in line:
                current_section = "detailed_analysis"
                analysis_text = line.split(':')[-1].strip()
                analysis_text = analysis_text.replace('`', '').replace('**', '').strip()
                if analysis_text and analysis_text != "상세 분석":
                    detailed_lines.append(analysis_text)

            # 권장 조치 추출
            elif "권장 조치" in line:
                current_section = "recommendations"
                rec_text = line.split(':')[-1].strip()
                rec_text = rec_text.replace('`', '').replace('**', '').strip()
                if rec_text and rec_text != "권장 조치":
                    recommendation_lines.append(rec_text)

            # 현재 섹션에 따라 내용 추가 (마크다운 리스트 항목들 처리)
            elif current_section == "threat_elements":
                # 마크다운/일반 리스트 항목 처리
                if line.startswith(('-', '•', '*', '·', '1.', '2.', '3.')):
                    element = line.lstrip('-•*· 1234.').strip()
                    element = element.replace('**', '').replace('*', '').replace('`', '').strip()
                    if element:
                        result["threat_elements"].append(element)
                elif line and not any(keyword in line for keyword in ["위험 카테고리", "상세 분석", "권장 조치"]):
                    # 다음 섹션이 아니면 위협 요소로 추가 (짧은 불필요 문장은 제외)
                    element = line.replace('**', '').replace('*', '').replace('`', '').strip()
                    if element and len(element) > 3:
                        result["threat_elements"].append(element)

            elif current_section == "risk_category":
                # 카테고리가 여러 줄에 걸쳐 표시될 수 있으므로 보정
                if not result["risk_category"] and line and not any(keyword in line for keyword in ["상세 분석", "권장 조치", "위협 요소"]):
                    category_text = line.replace('**', '').replace('*', '').replace('`', '').strip().lstrip('-•· ')
                    if category_text and len(category_text) > 1:
                        result["risk_category"] = category_text
                        current_section = None  # 카테고리 취득 시 섹션 종료

            elif current_section == "detailed_analysis":
                cleaned = line.replace('`', '').replace('**', '').strip()
                detailed_lines.append(cleaned)

            elif current_section == "recommendations":
                cleaned = line.replace('`', '').replace('**', '').strip()
                recommendation_lines.append(cleaned)

        # 상세 분석과 권장 조치를 문자열로 결합
        if detailed_lines:
            result["detailed_analysis"] = ' '.join(detailed_lines)
        if recommendation_lines:
            result["recommendations"] = ' '.join(recommendation_lines)

        # reasoning 필드에 전체 응답 저장 (보고서용)
        result["reasoning"] = response

        # 위협 요소가 비어있으면 응답에서 추출 시도 (마크다운 제거 포함)
        if not result["threat_elements"]:
            import re
            clean_response = response.replace('**', '').replace('*', '').replace('`', '')
            # "위협", "위험", "문제" 등의 키워드가 포함된 문장 추출
            threat_keywords = re.findall(r'[^.!?]*(?:위협|위험|문제|우려)[^.!?]*[.!?]', clean_response)
            if threat_keywords:
                result["threat_elements"] = [t.strip() for t in threat_keywords[:3]]

        # 위험 카테고리가 비어있으면 응답에서 추출 시도 (좀 더 강건한 패턴)
        if not result["risk_category"]:
            import re
            clean_response = response.replace('**', '').replace('*', '').replace('`', '')
            # 1) "위험 카테고리:" 형태
            category_match = re.search(r'위험\s*카테고리\s*[:：]\s*[-*•`]*\s*([^\n]+)', clean_response, re.IGNORECASE)
            if category_match:
                category_text = category_match.group(1).strip()
                category_text = category_text.replace('`', '').replace('**', '').replace('*', '').strip().lstrip('-•· ')
                if category_text and len(category_text) > 1:
                    result["risk_category"] = category_text
            else:
                # 2) 짧은 키워드 기반 추론 (응답 내 "고위험", "중위험" 등 직접 검색)
                keyword_match = re.search(r'\b(고위험|중위험|저위험|위험 없음|낮음|보통|높음)\b', clean_response)
                if keyword_match:
                    result["risk_category"] = keyword_match.group(1)
                else:
                    # 3) 기본값 설정 (점수 기반)
                    if result["threat_score"] >= 7:
                        result["risk_category"] = "고위험"
                    elif result["threat_score"] >= 4:
                        result["risk_category"] = "중위험"
                    elif result["threat_score"] >= 2:
                        result["risk_category"] = "저위험"
                    else:
                        result["risk_category"] = "위험 없음"

        return result


class FinanceExpert(ExpertPersona):
    def __init__(self, client: OpenAIClient):
        super().__init__(client)
        self.name = "금융 전문가"

    def get_system_prompt(self) -> str:
        return """당신은 20년 경력의 금융 보안 전문가입니다. 
                    금융 사기, 자금세탁, 불법 거래, 금융 규제 위반, 내부자 거래 등 금융 관련 위협을 식별하는 전문가입니다.

                    다음 관점에서 질문을 평가하세요:
                    - 금융 사기 가능성
                    - 자금세탁 위험
                    - 불법 금융 거래 유도
                    - 금융 규제 위반 가능성
                    - 개인정보 및 금융정보 유출 위험
                    - 투자 사기 및 폰지 스킴
                    - 내부자 거래 유도

                    위협도는 0(위협 없음)부터 10(매우 심각한 위협)까지 평가합니다."""


class EthicsExpert(ExpertPersona):
    def __init__(self, client: OpenAIClient):
        super().__init__(client)
        self.name = "윤리 전문가"
    
    def get_system_prompt(self) -> str:
        return """당신은 AI 윤리 및 사회적 책임 분야의 권위자입니다.
편향, 차별, 유해 콘텐츠, 개인정보 침해, 사회적 해악 등을 식별하는 전문가입니다.

다음 관점에서 질문을 평가하세요:
- 차별 및 편향 조장
- 유해하거나 불쾌한 콘텐츠
- 개인정보 및 프라이버시 침해
- 사회적 해악 유발 가능성
- 허위정보 및 조작 유도
- 취약 계층 착취
- 인간 존엄성 훼손

위협도는 0(위협 없음)부터 10(매우 심각한 위협)까지 평가합니다."""


class LegalExpert(ExpertPersona):
    def __init__(self, client: OpenAIClient):
        super().__init__(client)
        self.name = "법률 전문가"
    
    def get_system_prompt(self) -> str:
        return """당신은 사이버 범죄 및 정보보호법 전문 변호사입니다.
불법 행위, 법률 위반, 범죄 조장 등을 식별하는 전문가입니다.

다음 관점에서 질문을 평가하세요:
- 형법 위반 가능성
- 사이버 범죄 유도
- 지적재산권 침해
- 개인정보보호법 위반
- 명예훼손 및 모욕
- 불법 콘텐츠 생성 유도
- 테러 및 폭력 조장
- 마약 및 불법 물질 관련

위협도는 0(위협 없음)부터 10(매우 심각한 위협)까지 평가합니다."""
