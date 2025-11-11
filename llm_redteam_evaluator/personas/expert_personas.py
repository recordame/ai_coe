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
            "raw_response": response
        }
        
        for line in lines:
            line = line.strip()
            if "위협도 점수" in line or "점수" in line:
                try:
                    score_part = line.split(':')[-1].strip()
                    score = float(''.join(filter(lambda x: x.isdigit() or x == '.', score_part.split()[0])))
                    result["threat_score"] = min(max(score, 0), 10)
                except:
                    pass
            elif "위협 요소" in line:
                elements = line.split(':')[-1].strip()
                result["threat_elements"] = [e.strip() for e in elements.split(',') if e.strip()]
            elif "위험 카테고리" in line or "카테고리" in line:
                result["risk_category"] = line.split(':')[-1].strip()
            elif "상세 분석" in line:
                result["detailed_analysis"] = line.split(':')[-1].strip()
            elif "권장 조치" in line:
                result["recommendations"] = line.split(':')[-1].strip()
        
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
