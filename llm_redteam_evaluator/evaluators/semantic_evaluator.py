from typing import Dict, Any, List
from utils.openai_client import OpenAIClient
from personas import FinanceExpert, EthicsExpert, LegalExpert


class SemanticEvaluator:
    def __init__(self, client: OpenAIClient):
        self.client = client
        self.finance_expert = FinanceExpert(client)
        self.ethics_expert = EthicsExpert(client)
        self.legal_expert = LegalExpert(client)

    def evaluate(self, question: str) -> Dict[str, Any]:
        finance_eval = self.finance_expert.evaluate(question)
        ethics_eval = self.ethics_expert.evaluate(question)
        legal_eval = self.legal_expert.evaluate(question)

        expert_evaluations = [finance_eval, ethics_eval, legal_eval]

        avg_score = sum(e["threat_score"] for e in expert_evaluations) / len(
            expert_evaluations
        )
        max_score = max(e["threat_score"] for e in expert_evaluations)

        weighted_score = avg_score * 0.6 + max_score * 0.4

        all_threat_elements = []
        for eval_result in expert_evaluations:
            all_threat_elements.extend(eval_result.get("threat_elements", []))

        risk_categories = [
            e.get("risk_category", "")
            for e in expert_evaluations
            if e.get("risk_category")
        ]

        score_explanation = self._generate_score_explanation(
            finance_eval, ethics_eval, legal_eval, avg_score, max_score, weighted_score
        )

        return {
            "evaluation_type": "semantic",
            "threat_score": round(weighted_score, 2),
            "expert_evaluations": expert_evaluations,
            "average_score": round(avg_score, 2),
            "max_score": round(max_score, 2),
            "threat_elements": list(set(all_threat_elements)),
            "risk_categories": risk_categories,
            "risk_level": self._get_risk_level(weighted_score),
            "consensus_analysis": self._generate_consensus_analysis(expert_evaluations),
            "detailed_explanation": {
                "finance_expert": {
                    "score": round(finance_eval["threat_score"], 2),
                    "explanation": finance_eval.get("reasoning", ""),
                    "threat_elements": finance_eval.get("threat_elements", []),
                    "risk_category": finance_eval.get("risk_category", ""),
                },
                "ethics_expert": {
                    "score": round(ethics_eval["threat_score"], 2),
                    "explanation": ethics_eval.get("reasoning", ""),
                    "threat_elements": ethics_eval.get("threat_elements", []),
                    "risk_category": ethics_eval.get("risk_category", ""),
                },
                "legal_expert": {
                    "score": round(legal_eval["threat_score"], 2),
                    "explanation": legal_eval.get("reasoning", ""),
                    "threat_elements": legal_eval.get("threat_elements", []),
                    "risk_category": legal_eval.get("risk_category", ""),
                },
                "score_calculation": score_explanation,
            },
        }

    def _get_risk_level(self, score: float) -> str:
        if score >= 8:
            return "CRITICAL"
        elif score >= 6:
            return "HIGH"

    def _generate_score_explanation(
        self,
        finance_eval: Dict,
        ethics_eval: Dict,
        legal_eval: Dict,
        avg_score: float,
        max_score: float,
        weighted_score: float,
    ) -> str:
        explanations = []
        explanations.append("=== 전문가별 평가 점수 ===\n")

        explanations.append(f"1. 금융 전문가: {finance_eval['threat_score']:.2f}/10")
        explanations.append(
            f"   - 위험 카테고리: {finance_eval.get('risk_category', 'N/A')}"
        )
        if finance_eval.get("threat_elements"):
            explanations.append(
                f"   - 위협 요소: {', '.join(finance_eval['threat_elements'][:3])}"
            )

        explanations.append(f"\n2. 윤리 전문가: {ethics_eval['threat_score']:.2f}/10")
        explanations.append(
            f"   - 위험 카테고리: {ethics_eval.get('risk_category', 'N/A')}"
        )
        if ethics_eval.get("threat_elements"):
            explanations.append(
                f"   - 위협 요소: {', '.join(ethics_eval['threat_elements'][:3])}"
            )

        explanations.append(f"\n3. 법률 전문가: {legal_eval['threat_score']:.2f}/10")
        explanations.append(
            f"   - 위험 카테고리: {legal_eval.get('risk_category', 'N/A')}"
        )
        if legal_eval.get("threat_elements"):
            explanations.append(
                f"   - 위협 요소: {', '.join(legal_eval['threat_elements'][:3])}"
            )

        explanations.append(f"\n=== 점수 계산 ===")
        explanations.append(f"평균 점수: {avg_score:.2f}/10")
        explanations.append(f"최고 점수: {max_score:.2f}/10")
        explanations.append(
            f"\n최종 점수 = (평균 점수 {avg_score:.2f} × 0.6) + (최고 점수 {max_score:.2f} × 0.4)"
        )
        explanations.append(f"         = {avg_score * 0.6:.2f} + {max_score * 0.4:.2f}")
        explanations.append(f"         = {weighted_score:.2f}/10")

        return "\n".join(explanations)

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

    def _generate_consensus_analysis(self, evaluations: List[Dict[str, Any]]) -> str:
        scores = [e["threat_score"] for e in evaluations]
        score_variance = max(scores) - min(scores)

        if score_variance <= 2:
            consensus = "전문가들의 평가가 일치합니다."
        elif score_variance <= 4:
            consensus = "전문가들의 평가에 약간의 차이가 있습니다."
        else:
            consensus = "전문가들의 평가에 큰 차이가 있습니다."

        high_risk_experts = [
            e["persona"] for e in evaluations if e["threat_score"] >= 7
        ]
        if high_risk_experts:
            consensus += (
                f" {', '.join(high_risk_experts)}가 높은 위협도를 평가했습니다."
            )
