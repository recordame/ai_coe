"""
LLM 추론 능력 평가 파이프라인 (간소화 버전)

주요 기능:
1. 기본 데이터셋 로드
2. 영문 번역 및 셔플
3. 다양한 노이즈 생성
4. 노이즈 조합 데이터셋 생성
5. 평가 요약
"""

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Dict, List
import warnings

from openai import OpenAI

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ==================== API 클라이언트 ====================

client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc", base_url="https://api.upstage.ai/v1"
)


def call_llm(messages: List[Dict], temperature: float = 0.4) -> str:
    """LLM API 호출"""
    response = client.chat.completions.create(
        model="upstage/solar-1-mini-chat",
        messages=messages,
        temperature=temperature,
        max_tokens=16384,
        stream=False,
    )
    return response.choices[0].message.content


# ==================== 설정 ====================


@dataclass
class PipelineConfig:
    """파이프라인 설정"""

    # 필수 설정
    base_context_file: str
    output_dir: str = "./output"

    # 생성 설정
    num_unrelated_contexts: int = 70
    random_seed: int = 42

    # 난이도 설정
    difficulties: Dict = field(
        default_factory=lambda: {
            "easy": {
                "noise_types": ["unrelated", "english"],
                "positions": ["append"],
                "description": "적은 양의 무관한 정보를 끝에 추가",
            },
            "medium": {
                "noise_types": ["unrelated", "english", "similar"],
                "positions": ["prepend", "append", "sandwich"],
                "description": "유사 도메인 정보를 다양한 위치에 추가",
            },
            "hard": {
                "noise_types": ["unrelated", "english", "similar", "contradictory"],
                "positions": ["sandwich", "interleave"],
                "description": "모순 정보를 본문 사이에 삽입",
            },
            "extreme": {
                "noise_types": [
                    "unrelated",
                    "english",
                    "similar",
                    "contradictory",
                    "partial",
                ],
                "positions": ["interleave", "random"],
                "description": "다양한 혼란 정보를 무작위로 삽입",
            },
        }
    )


# ==================== 노이즈 생성 ====================


def generate_unrelated_contexts(num: int) -> List[str]:
    """무관한 법률 조항 생성"""
    print(f"\n📝 무관한 법률 조항 {num}개 생성 중...")

    contexts = []
    for i in range(num):
        print(f"  진행: {i + 1}/{num}", end="\r")

        messages = [
            {
                "role": "system",
                "content": "당신은 법률 전문가입니다. 금융 관련이 아닌 법률(헌법, 민법, 형법 등)의 "
                "조항을 30줄 이내로 생성해주세요. 설명 없이 법률 조항만 출력하세요.",
            },
            {"role": "user", "content": "아무 법조항이나 30줄 이내로 알려줘!"},
        ]

        context = call_llm(messages, temperature=1.0)
        contexts.append(context)

    print(f"\n✅ 완료: {len(contexts)}개 생성")
    return contexts


def generate_similar_domain(base_context: str) -> str:
    """유사 도메인 노이즈 생성"""
    messages = [
        {
            "role": "system",
            "content": "금융 법률 전문가로서, 주어진 본문과 유사하지만 질문에 도움이 되지 않는 "
            "금융 법률 조항을 30줄 이내로 생성하세요. 설명 없이 조항만 출력하세요.",
        },
        {
            "role": "user",
            "content": f"다음 본문과 유사하지만 무관한 금융 법률 조항을 생성하세요:\n\n{base_context}",
        },
    ]
    return call_llm(messages, temperature=0.8)


def generate_contradictory(base_context: str, answer: str) -> str:
    """모순 정보 생성"""
    messages = [
        {
            "role": "system",
            "content": "금융 법률 전문가로서, 주어진 정답과 모순되지만 그럴듯한 법률 조항을 "
            "30줄 이내로 생성하세요. 설명 없이 조항만 출력하세요.",
        },
        {
            "role": "user",
            "content": f"본문: {base_context}\n\n정답: {answer}\n\n정답과 모순되는 법률 조항을 생성하세요.",
        },
    ]
    return call_llm(messages, temperature=0.8)


def generate_partial_overlap(base_context: str) -> str:
    """부분 겹침 정보 생성"""
    messages = [
        {
            "role": "system",
            "content": "금융 법률 전문가로서, 주어진 본문과 일부 키워드는 겹치지만 의미는 다른 "
            "법률 조항을 30줄 이내로 생성하세요. 설명 없이 조항만 출력하세요.",
        },
        {
            "role": "user",
            "content": f"다음 본문과 일부 겹치지만 다른 내용의 법률 조항을 생성하세요:\n\n{base_context}",
        },
    ]
    return call_llm(messages, temperature=0.8)


def translate_to_english(text: str) -> str:
    """한국어를 영어로 번역"""
    messages = [
        {
            "role": "system",
            "content": "금융 법률 전문 번역가로서, 한국어 법 조항을 영어로 번역하세요. "
            "설명 없이 번역문만 출력하세요.",
        },
        {"role": "user", "content": text},
    ]
    return call_llm(messages, temperature=0)


# ==================== 노이즈 배치 ====================


class NoisePositioner:
    """노이즈 위치 조정"""

    @staticmethod
    def apply(base: str, noises: List[str], position: str) -> str:
        """노이즈를 지정된 위치에 배치"""
        if not noises:
            return base

        combined = "\n\n".join(noises)

        if position == "prepend":
            return f"{combined}\n\n{base}"

        elif position == "append":
            return f"{base}\n\n{combined}"

        elif position == "sandwich":
            if len(noises) >= 2:
                return f"{noises[0]}\n\n{base}\n\n{noises[1]}"
            return f"{combined}\n\n{base}\n\n{combined}"

        elif position == "interleave":
            return NoisePositioner._interleave(base, noises)

        elif position == "random":
            return NoisePositioner._random_insert(base, combined)

        return base

    @staticmethod
    def _interleave(base: str, noises: List[str]) -> str:
        """문장 사이사이에 노이즈 삽입"""
        base_sentences = [s.strip() for s in base.split(".") if s.strip()]
        noise_sentences = []
        for noise in noises:
            noise_sentences.extend([s.strip() for s in noise.split(".") if s.strip()])

        result = []
        noise_idx = 0

        for i, sentence in enumerate(base_sentences):
            result.append(sentence)
            if i % 2 == 1 and noise_idx < len(noise_sentences):
                result.append(noise_sentences[noise_idx])
                noise_idx += 1

        return ". ".join(result) + "."

    @staticmethod
    def _random_insert(base: str, noise: str, num_insertions: int = 3) -> str:
        """무작위 위치에 노이즈 삽입"""
        base_sentences = [s.strip() for s in base.split(".") if s.strip()]
        noise_sentences = [s.strip() for s in noise.split(".") if s.strip()]

        for _ in range(min(num_insertions, len(noise_sentences))):
            if noise_sentences:
                pos = random.randint(0, len(base_sentences))
                base_sentences.insert(pos, random.choice(noise_sentences))

        return ". ".join(base_sentences) + "."


# ==================== 메인 파이프라인 ====================


class InferenceBenchmarkPipeline:
    """LLM 추론 능력 평가 파이프라인"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.positioner = NoisePositioner()

        # 출력 디렉토리 설정
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.timestamp = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")

        # 시드 설정
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    def _load_or_create(self, filename: str, create_func, *args) -> pd.DataFrame:
        """파일이 있으면 로드, 없으면 생성"""
        filepath = self.output_path / filename

        try:
            df = pd.read_csv(filepath, index_col="id")
            print(f"✅ 파일에서 로드: {filename}")
            return df
        except FileNotFoundError:
            print(f"📝 새로 생성: {filename}")
            data = create_func(*args)
            df = pd.DataFrame({"context": data})
            df.to_csv(filepath, index_label="id")
            return df

    def prepare_english_contexts(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """영문 번역 및 셔플"""
        print("\n" + "=" * 80)
        print("📖 영문 번역 및 셔플")
        print("=" * 80)

        def create_english():
            contexts = []
            total = len(base_df)
            for idx, row in base_df.iterrows():
                print(f"  번역 중: {idx + 1}/{total}", end="\r")
                translated = translate_to_english(row["context"])
                contexts.append(translated)
            print(f"\n✅ 번역 완료: {total}개")
            return contexts

        df = self._load_or_create("shuffled_english_contexts.csv", create_english)

        # 셔플 (이미 저장된 경우 다시 셔플하지 않음)
        if len(df) == len(base_df):
            df = df.sample(frac=1, random_state=self.config.random_seed).reset_index(
                drop=True
            )
            df.to_csv(
                self.output_path / "shuffled_english_contexts.csv", index_label="id"
            )

        return df

    def prepare_noise_contexts(self, base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """모든 노이즈 컨텍스트 생성"""
        print("\n" + "=" * 80)
        print("🔊 노이즈 컨텍스트 생성")
        print("=" * 80)

        noise_dfs = {}
        sample_size = 20

        # 1. 무관한 법률
        noise_dfs["unrelated"] = self._load_or_create(
            "unrelated_contexts.csv",
            generate_unrelated_contexts,
            self.config.num_unrelated_contexts,
        )

        # 2. 유사 도메인
        def create_similar():
            contexts = []
            for idx, row in base_df.head(sample_size).iterrows():
                print(f"  유사 도메인 생성: {idx + 1}/{sample_size}", end="\r")
                contexts.append(generate_similar_domain(row["context"]))
            print(f"\n✅ 유사 도메인 완료: {len(contexts)}개")
            return contexts

        noise_dfs["similar"] = self._load_or_create(
            "similar_domain_contexts.csv", create_similar
        )

        # 3. 모순 정보
        def create_contradictory():
            contexts = []
            for idx, row in base_df.head(sample_size).iterrows():
                print(f"  모순 정보 생성: {idx + 1}/{sample_size}", end="\r")
                contexts.append(generate_contradictory(row["context"], row["answer"]))
            print(f"\n✅ 모순 정보 완료: {len(contexts)}개")
            return contexts

        noise_dfs["contradictory"] = self._load_or_create(
            "contradictory_contexts.csv", create_contradictory
        )

        # 4. 부분 겹침
        def create_partial():
            contexts = []
            for idx, row in base_df.head(sample_size).iterrows():
                print(f"  부분 겹침 생성: {idx + 1}/{sample_size}", end="\r")
                contexts.append(generate_partial_overlap(row["context"]))
            print(f"\n✅ 부분 겹침 완료: {len(contexts)}개")
            return contexts

        noise_dfs["partial"] = self._load_or_create(
            "partial_overlap_contexts.csv", create_partial
        )

        return noise_dfs

    def create_noisy_datasets(
        self,
        base_df: pd.DataFrame,
        english_df: pd.DataFrame,
        noise_dfs: Dict[str, pd.DataFrame],
    ) -> List[pd.DataFrame]:
        """노이즈 조합 데이터셋 생성"""
        print("\n" + "=" * 80)
        print("🎲 노이즈 조합 데이터셋 생성")
        print("=" * 80)

        datasets = []

        for difficulty_name, difficulty_config in self.config.difficulties.items():
            print(f"\n📊 난이도: {difficulty_name.upper()}")
            print(f"   {difficulty_config['description']}")

            for position in difficulty_config["positions"]:
                print(f"   위치: {position}")

                rows = []

                for idx, row in base_df.iterrows():
                    # 노이즈 선택
                    noises = []
                    for noise_type in difficulty_config["noise_types"]:
                        if noise_type == "english":
                            if idx < len(english_df):
                                noises.append(english_df.iloc[idx]["context"])
                        elif noise_type in noise_dfs:
                            noise_idx = random.randint(
                                0, len(noise_dfs[noise_type]) - 1
                            )
                            noises.append(
                                noise_dfs[noise_type].iloc[noise_idx]["context"]
                            )

                    # 노이즈 배치
                    noisy_context = self.positioner.apply(
                        row["context"], noises, position
                    )

                    rows.append(
                        {
                            "id": idx,
                            "difficulty": difficulty_name,
                            "position": position,
                            "noise_types": ",".join(difficulty_config["noise_types"]),
                            "original_context": row["context"],
                            "noisy_context": noisy_context,
                            "question": row["question"],
                            "answer": row["answer"],
                            "reasoning": row.get("reasoning", row.get("reason", "")),
                        }
                    )

                df = pd.DataFrame(rows)
                datasets.append(df)

                # 저장
                filename = f"noisy_{difficulty_name}_{position}_{self.timestamp}.csv"
                df.to_csv(self.output_path / filename, index=False)
                print(f"   ✅ 저장: {filename}")

        return datasets

    def create_summary(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """평가 요약 생성"""
        print("\n" + "=" * 80)
        print("📈 평가 요약 생성")
        print("=" * 80)

        rows = []
        for df in datasets:
            if len(df) == 0:
                continue

            rows.append(
                {
                    "difficulty": df.iloc[0]["difficulty"],
                    "position": df.iloc[0]["position"],
                    "num_samples": len(df),
                    "noise_types": df.iloc[0]["noise_types"],
                    "avg_noisy_length": df["noisy_context"].str.len().mean(),
                    "avg_original_length": df["original_context"].str.len().mean(),
                    "noise_ratio": df["noisy_context"].str.len().mean()
                    / df["original_context"].str.len().mean(),
                }
            )

        summary_df = pd.DataFrame(rows)
        filename = f"summary_{self.timestamp}.csv"
        summary_df.to_csv(self.output_path / filename, index=False)
        print(f"✅ 요약 저장: {filename}")

        return summary_df

    def run(self) -> Dict:
        """전체 파이프라인 실행"""
        print("\n" + "=" * 80)
        print("🚀 LLM 추론 능력 평가 파이프라인 시작")
        print(f"⏰ 시작: {pd.Timestamp.now(tz='Asia/Seoul')}")
        print("=" * 80)

        # 1. 기본 데이터셋 로드
        print(f"\n📂 기본 데이터셋 로드: {self.config.base_context_file}")
        base_df = pd.read_csv(self.config.base_context_file)
        if "id" not in base_df.columns:
            base_df = base_df.reset_index().rename(columns={"index": "id"})
        print(f"✅ {len(base_df)}개 샘플 로드")

        # 2. 영문 번역
        english_df = self.prepare_english_contexts(base_df)

        # 3. 노이즈 생성
        noise_dfs = self.prepare_noise_contexts(base_df)

        # 4. 노이즈 데이터셋 생성
        datasets = self.create_noisy_datasets(base_df, english_df, noise_dfs)

        # 5. 요약
        summary_df = self.create_summary(datasets)

        print("\n" + "=" * 80)
        print("✅ 파이프라인 완료!")
        print(f"⏰ 종료: {pd.Timestamp.now(tz='Asia/Seoul')}")
        print(f"📁 출력: {self.output_path}")
        print("=" * 80)

        return {
            "base_df": base_df,
            "english_df": english_df,
            "noise_dfs": noise_dfs,
            "datasets": datasets,
            "summary_df": summary_df,
        }


# ==================== 실행 ====================

if __name__ == "__main__":
    config = PipelineConfig(
        base_context_file="./dataset/base_contexts.csv",
        output_dir="./output",
        num_unrelated_contexts=70,
        random_seed=42,
    )

    pipeline = InferenceBenchmarkPipeline(config)
    results = pipeline.run()

    print("\n" + "=" * 80)
    print("📊 최종 결과")
    print("=" * 80)
    print(f"생성된 데이터셋: {len(results['datasets'])}개")
    print(f"\n평가 요약:\n{results['summary_df'].to_string()}")
