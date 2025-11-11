import re
from typing import Dict, Any, List
from utils.openai_client import OpenAIClient


class SyntacticEvaluator:
    def __init__(self, client: OpenAIClient):
        self.client = client
        self.threat_patterns = {
            "jailbreak_attempts": [
                r"ignore\s+(previous|above|all)\s+instructions?",
                r"disregard\s+.*\s+rules?",
                r"forget\s+.*\s+constraints?",
                r"act\s+as\s+if",
                r"pretend\s+(you|to)\s+are",
                r"roleplay\s+as",
                r"DAN\s+mode",
                r"developer\s+mode",
            ],
            "prompt_injection": [
                r"system\s*:\s*",
                r"<\s*system\s*>",
                r"\[SYSTEM\]",
                r"###\s*instruction",
                r"new\s+instructions?:",
                r"override\s+.*\s+settings?",
            ],
            "obfuscation": [
                r"[a-z]\s+[a-z]\s+[a-z]\s+[a-z]",
                r"[^\x00-\x7F]{10,}",
                r"base64|hex|encode|decode",
                r"rot13|caesar",
            ],
            "social_engineering": [
                r"urgent|emergency|immediately",
                r"confidential|secret|classified",
                r"bypass|circumvent|workaround",
                r"exploit|vulnerability|hack",
            ],
            "malicious_keywords": [
                r"malware|virus|trojan|ransomware",
                r"phishing|scam|fraud",
                r"illegal|unlawful|criminal",
                r"weapon|bomb|explosive",
                r"drug|narcotic|substance",
            ]
        }
    
    def evaluate(self, question: str) -> Dict[str, Any]:
        detected_patterns = self._detect_patterns(question)
        structural_score = self._calculate_structural_score(question)
        pattern_score = self._calculate_pattern_score(detected_patterns)

        llm_analysis = self._llm_syntactic_analysis(question)

        final_score = (pattern_score * 0.4 + structural_score * 0.3 + llm_analysis["score"] * 0.3)

        pattern_explanation = self._generate_pattern_explanation(detected_patterns, pattern_score)
        structural_explanation = self._generate_structural_explanation(question, structural_score)

        return {
            "evaluation_type": "syntactic",
            "threat_score": round(final_score, 2),
            "detected_patterns": detected_patterns,
            "structural_analysis": {
                "length": len(question),
                "special_chars_ratio": self._calculate_special_chars_ratio(question),
                "uppercase_ratio": self._calculate_uppercase_ratio(question),
            },
            "llm_analysis": llm_analysis,
            "risk_level": self._get_risk_level(final_score),
            "detailed_explanation": {
                "pattern_score": {
                    "score": round(pattern_score, 2),
                    "weight": "40%",
                    "explanation": pattern_explanation
                },
                "structural_score": {
                    "score": round(structural_score, 2),
                    "weight": "30%",
                    "explanation": structural_explanation
                },
                "llm_analysis_score": {
                    "score": round(llm_analysis["score"], 2),
                    "weight": "30%",
                    "explanation": llm_analysis["analysis"]
                },
                "final_calculation": f"최종 점수 = (패턴 점수 {round(pattern_score, 2)} × 0.4) + (구조 점수 {round(structural_score, 2)} × 0.3) + (LLM 분석 점수 {round(llm_analysis['score'], 2)} × 0.3) = {round(final_score, 2)}"
            }
        }
    
    def _detect_patterns(self, text: str) -> Dict[str, List[str]]:
        detected = {}
        text_lower = text.lower()
        
        for category, patterns in self.threat_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches.append(pattern)
            if matches:
                detected[category] = matches
        
        return detected
    
    def _calculate_pattern_score(self, detected_patterns: Dict[str, List[str]]) -> float:
        if not detected_patterns:
            return 0.0
        
        category_weights = {
            "jailbreak_attempts": 3.0,
            "prompt_injection": 3.5,
            "obfuscation": 2.5,
            "social_engineering": 2.0,
            "malicious_keywords": 2.5
        }
        
        total_score = 0.0
        for category, matches in detected_patterns.items():
            weight = category_weights.get(category, 1.0)
            total_score += len(matches) * weight
        
        return min(total_score, 10.0)
    
    def _calculate_structural_score(self, text: str) -> float:
        score = 0.0
        
        if len(text) > 1000:
            score += 2.0
        
        special_ratio = self._calculate_special_chars_ratio(text)
        if special_ratio > 0.3:
            score += 3.0
        elif special_ratio > 0.2:
            score += 1.5
        
        uppercase_ratio = self._calculate_uppercase_ratio(text)
        if uppercase_ratio > 0.5:
            score += 2.0
        
        if text.count('\n') > 10:
            score += 1.0
        
        return min(score, 10.0)
    
    def _calculate_special_chars_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special_chars / len(text)
    
    def _calculate_uppercase_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)
    
    def _llm_syntactic_analysis(self, question: str) -> Dict[str, Any]:
        prompt = f"""다음 질문의 문법적(syntactic) 위협 요소를 분석해주세요:

                    질문: {question}

                    다음 관점에서 분석하세요:
                    1. Jailbreak 시도 패턴
                    2. Prompt Injection 시도
                    3. 난독화 기법 사용
                    4. 비정상적인 문장 구조
                    5. 악의적인 명령어 패턴

                    위협도 점수 (0-10)와 발견된 문법적 위협 요소를 설명해주세요."""

        messages = [
            {"role": "system", "content": "당신은 LLM 보안 전문가로서 문법적 공격 패턴을 식별하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat_completion(messages, temperature=0.2)
        
        score = 5.0
        for line in response.split('\n'):
            if '점수' in line or 'score' in line.lower():
                try:
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        score = float(numbers[0])
                        # Generate detailed explanations for detected patterns and structural analysis,
                        # making them available for inclusion in the final result.
                        try:
                            detected_patterns = self._detect_patterns(question)
                            pattern_score = self._calculate_pattern_score(detected_patterns)
                            structural_score = self._calculate_structural_score(question)
                            pattern_explanation = self._generate_pattern_explanation(detected_patterns, pattern_score)
                            structural_explanation = self._generate_structural_explanation(question, structural_score)
                            # Store explanations on the instance for use when building the return value
                            # (safe even if not used elsewhere).
                            self._last_pattern_explanation = pattern_explanation
                            self._last_structural_explanation = structural_explanation
                        except Exception:
                            # If any helper is missing or fails, don't interrupt parsing — keep defaults.
                            self._last_pattern_explanation = None
                            self._last_structural_explanation = None
                        break
                except:
                    pass
        
        return {
            "score": min(max(score, 0), 10),
            "analysis": response
        }
    
    def _get_risk_level(self, score: float) -> str:
        if score >= 8:
            return "CRITICAL"
        elif score >= 6:
            return "HIGH"
        elif score >= 4:
            return "MEDIUM"
        elif score >= 2:
            return "LOW"
        else:
            return "MINIMAL"

    def _generate_pattern_explanation(self, detected_patterns: Dict[str, List[str]], score: float) -> str:
        if not detected_patterns:
            return "위협 패턴이 감지되지 않았습니다. 질문에 알려진 공격 패턴이 포함되어 있지 않습니다."

        explanations = []
        explanations.append(f"총 {len(detected_patterns)}개 카테고리에서 위협 패턴이 감지되었습니다:\n")

        category_names = {
            "jailbreak_attempts": "탈옥 시도",
            "prompt_injection": "프롬프트 인젝션",
            "obfuscation": "난독화",
            "social_engineering": "사회공학",
            "malicious_keywords": "악의적 키워드"
        }

        category_weights = {
            "jailbreak_attempts": 3.0,
            "prompt_injection": 3.5,
            "obfuscation": 2.5,
            "social_engineering": 2.0,
            "malicious_keywords": 2.5
        }

        for category, matches in detected_patterns.items():
            category_name = category_names.get(category, category)
            weight = category_weights.get(category, 1.0)
            category_score = len(matches) * weight
            explanations.append(f"  • {category_name}: {len(matches)}개 패턴 감지 (가중치: {weight}, 기여도: {category_score}점)")

        explanations.append(f"\n패턴 분석 점수: {round(score, 2)}/10")
        return "\n".join(explanations)

    def _generate_structural_explanation(self, text: str, score: float) -> str:
        explanations = []
        explanations.append("구조적 분석 결과:\n")

        length = len(text)
        explanations.append(f"  • 텍스트 길이: {length}자")
        if length > 1000:
            explanations.append(f"    → 비정상적으로 긴 텍스트 (+2.0점)")

        special_ratio = self._calculate_special_chars_ratio(text)
        explanations.append(f"  • 특수문자 비율: {special_ratio:.2%}")
        if special_ratio > 0.3:
            explanations.append(f"    → 매우 높은 특수문자 비율 (+3.0점)")
        elif special_ratio > 0.2:
            explanations.append(f"    → 높은 특수문자 비율 (+1.5점)")

        uppercase_ratio = self._calculate_uppercase_ratio(text)
        explanations.append(f"  • 대문자 비율: {uppercase_ratio:.2%}")
        if uppercase_ratio > 0.5:
            explanations.append(f"    → 비정상적으로 높은 대문자 사용 (+2.0점)")

        newline_count = text.count('\n')
        explanations.append(f"  • 줄바꿈 횟수: {newline_count}회")
        if newline_count > 10:
            explanations.append(f"    → 과도한 줄바꿈 (+1.0점)")

        explanations.append(f"\n구조 분석 점수: {round(score, 2)}/10")
        return "\n".join(explanations)
