import random
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from openai import OpenAI

warnings.filterwarnings('ignore')

client = OpenAI(
    api_key="up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc",
    base_url="https://api.upstage.ai/v1"
)


# 언어 모델
def call_llm(messages, temp: float = 0.4) -> str:
    response = client.chat.completions.create(
        model="upstage/solar-1-mini-chat",
        messages=messages,
        stream=False,
        temperature=temp,
        max_tokens=16384,
    )

    return response.choices[0].message.content


# 노이즈 유형
class NoiseType(Enum):
    UNRELATED_LEGAL = "unrelated_legal"  # 무관한 법률 조항
    SIMILAR_DOMAIN = "similar_domain"  # 유사 도메인
    CONTRADICTORY = "contradictory"  # 모순 정보
    PARTIAL_OVERLAP = "partial_overlap"  # 부분 겹침
    SHUFFLED_ENGLISH = "shuffled_english"  # 셔플된 영문


# 노이즈 위치 전략
class NoisePosition(Enum):
    PREPEND = "prepend"  # 앞에 추가
    APPEND = "append"  # 뒤에 추가
    SANDWICH = "sandwich"  # 앞뒤로 감싸기
    INTERLEAVE = "interleave"  # 사이사이 삽입
    RANDOM_INSERT = "random"  # 무작위 삽입


# 난이도 설정
@dataclass
class DifficultyConfig:
    name: str
    noise_ratio: float
    noise_types: List[NoiseType]
    positions: List[NoisePosition]
    description: str


# 파이프라인 전체 설정
@dataclass
class PipelineConfig:
    # 파일 경로
    base_context_file: str
    output_dir: str = "./output"

    # 생성 설정
    num_unrelated_contexts: int = 70
    random_seed: int = 42

    # 난이도 설정
    difficulty_levels: List[DifficultyConfig] = field(default_factory=list)

    # 평가 설정
    num_consistency_runs: int = 5

    def __post_init__(self):
        if not self.difficulty_levels:
            self.difficulty_levels = [
                DifficultyConfig(
                    name="easy",
                    noise_ratio=0.2,
                    noise_types=[NoiseType.UNRELATED_LEGAL, NoiseType.SHUFFLED_ENGLISH],
                    positions=[NoisePosition.APPEND],
                    description="적은 양의 무관한 정보를 끝에 추가"
                ),
                DifficultyConfig(
                    name="medium",
                    noise_ratio=1.0,
                    noise_types=[NoiseType.UNRELATED_LEGAL, NoiseType.SHUFFLED_ENGLISH, NoiseType.SIMILAR_DOMAIN],
                    positions=[NoisePosition.PREPEND, NoisePosition.APPEND, NoisePosition.SANDWICH],
                    description="본문과 같은 양의 유사 도메인 정보를 다양한 위치에 추가"
                ),
                DifficultyConfig(
                    name="hard",
                    noise_ratio=2.0,
                    noise_types=[NoiseType.UNRELATED_LEGAL, NoiseType.SHUFFLED_ENGLISH, NoiseType.SIMILAR_DOMAIN, NoiseType.CONTRADICTORY],
                    positions=[NoisePosition.SANDWICH, NoisePosition.INTERLEAVE],
                    description="본문의 2배 분량의 모순 정보를 본문 사이에 삽입"
                ),
                DifficultyConfig(
                    name="extreme",
                    noise_ratio=5.0,
                    noise_types=[NoiseType.UNRELATED_LEGAL, NoiseType.SHUFFLED_ENGLISH, NoiseType.SIMILAR_DOMAIN, NoiseType.CONTRADICTORY, NoiseType.PARTIAL_OVERLAP],
                    positions=[NoisePosition.INTERLEAVE, NoisePosition.RANDOM_INSERT],
                    description="본문의 5배 분량의 다양한 혼란 정보를 무작위로 삽입"
                )
            ]


# 무관한 법률 조항 생성
def generate_unrelated_contexts(num: int = 70) -> List[str]:
    print(f"무관한 법률 조항 {num}개 생성 중...")

    unrelated_contexts = []

    for i in range(num):
        print(f"  진행: {i + 1}/{num}")

        messages = [
            {
                "role": "system",
                "content": "당신은 법률 전문가 입니다. 사용자가 '아무 법조항이나 알려줘' 라는 입력을 하면 "
                           "금융관련 법률 이 외의 법률(예, 헌법, 민법, 형법, 행정법 등)에서 법률과 하위 조, 항, 호, 목을 출력해줘! "
                           "그 어떤 법률 해설이나 마크다운 서식을 적용하지 말고 줄바꿈만 해줘. 단 매 생성마다 새로운 법안을 불러와줘! "
                           "생성에 대한 기준 또는 어떤한 생성관련 설명은 필요 없어!!!"
            },
            {
                "role": "user",
                "content": "아무 법조항이나 30줄 이내로 알려줘!"
            }
        ]

        context = call_llm(messages, 1)
        print(context)
        unrelated_contexts.append(context)

    return unrelated_contexts


# 유사 도메인 노이즈 생성""
def generate_similar_domain_contexts(base_context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "당신은 금융 법률 전문가입니다. 주어진 본문과 같은 금융 도메인이지만, "
                       "본문의 질문에 답하는 데 전혀 도움이 되지 않는 다른 금융 법률 조항을 생성해주세요."
                       "생성에 대한 기준 또는 어떤한 생성관련 설명은 필요 없어요!!!"
        },
        {
            "role": "user",
            "content": f"""다음 본문과 유사하지만 무관한 금융 법률 조항을 생성해주세요.
                        본문:
                        {base_context}

                        요구사항:
                        - 금융 관련 용어를 사용할 것
                        - 본문과 비슷한 구조를 가질 것
                        - 하지만 본문 과는 완전히 무관할 것
                        - 법률 조항 형식으로 작성할 것
                        - 30줄 이내로 생성할 것
                        """
        }
    ]

    return call_llm(messages, 0.8)


# 모순 정보 생성
def generate_contradictory_context(base_context: str, answer: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "당신은 금융 법률 전문가입니다. 주어진 본문과 정답을 보고, "
                       "정답과 모순되지만 그럴듯해 보이는 법률 조항을 생성해주세요."
                       "생성에 대한 기준 또는 어떤한 생성관련 설명은 필요 없어요!!!"
        },
        {
            "role": "user",
            "content": f"""다음 본문과 정답을 보고, 정답과 모순되는 정보를 생성해주세요.
                        본문:
                        {base_context}

                        정답:
                        {answer}

                        요구사항:
                        - 정답과 반대되는 내용
                        - 하지만 문맥상 자연스러울 것
                        - 법률 조항 형식을 유지할 것
                        - 30줄 이내로 생성 할 것
                        """
        }
    ]

    return call_llm(messages, 0.8)


# 부분 겹침 정보 생성
def generate_partial_overlap(base_context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "당신은 금융 법률 전문가입니다. 주어진 본문과 일부 단어나 개념은 겹치지만, "
                       "전체적으로는 다른 내용의 법률 조항을 생성해주세요."
                       "생성에 대한 기준 또는 어떤한 생성관련 설명은 필요 없어요!!!"
        },
        {
            "role": "user",
            "content": f"""다음 본문과 일부 겹치지만 다른 내용의 법률 조항을 생성해주세요.
                        본문:
                        {base_context}

                        요구사항:
                        - 본문의 주요 키워드 중 일부를 사용할 것
                        - 하지만 전체 의미는 완전히 다를 것
                        - 법률 조항 형식을 유지할 것
                        - 30줄 이내로 생성할 것
                        """
        }
    ]

    return call_llm(messages, 0.8)


# 노이즈 위치 조정
class NoisePositioner:
    # 노이즈를 앞에 배치
    @staticmethod
    def prepend(base: str, noise: str) -> str:
        return f"{noise}\n\n{base}"

    # 노이즈를 뒤에 배치
    @staticmethod
    def append(base: str, noise: str) -> str:
        return f"{base}\n\n{noise}"

    # 노이즈로 본문을 감싸기
    @staticmethod
    def sandwich(base: str, noise1: str, noise2: str) -> str:
        return f"{noise1}\n\n{base}\n\n{noise2}"

    # 본문을 문장 단위로 나누고 노이즈를 사이사이 삽입
    @staticmethod
    def interleave(base: str, noises: List[str]) -> str:
        base_sentences = [s.strip() for s in base.split('.') if s.strip()]
        noise_sentences = []

        for noise in noises:
            noise_sentences.extend([s.strip() for s in noise.split('.') if s.strip()])

        result = []
        noise_idx = 0

        for i, sentence in enumerate(base_sentences):
            result.append(sentence)

            # 매 2문장마다 노이즈 삽입
            if i % 2 == 1 and noise_idx < len(noise_sentences):
                result.append(noise_sentences[noise_idx])
                noise_idx += 1

        return '. '.join(result) + '.'

    # 무작위 위치에 노이즈 삽입
    @staticmethod
    def random_insert(base: str, noise: str, num_insertions: int = 3) -> str:
        base_sentences = [s.strip() for s in base.split('.') if s.strip()]
        noise_sentences = [s.strip() for s in noise.split('.') if s.strip()]

        for _ in range(min(num_insertions, len(noise_sentences))):
            if noise_sentences:
                pos = random.randint(0, len(base_sentences))
                base_sentences.insert(pos, random.choice(noise_sentences))

        return '. '.join(base_sentences) + '.'


# 번역
def translate_to_english(content: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "당신은 금융 법률용어 전문가로 한국어와 영어에 능통한 번역가이기도 합니다. 사용자가 입력한 법 조항을 친절하게 영어로 번역해주세요! 대답을 할 때는 이런저런 설명(예: Translate the following into English!) 붙이지 말고 번역문만 출력해주세요!!"
        },
        {
            "role": "user",
            "content": f"{content}"
        }
    ]

    english_statement = call_llm(messages, 0)

    return english_statement


# LLM 추론 능력 평가 통합 파이프라인
class InferenceBenchmarkPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.noise_positioner = NoisePositioner()

        # 결과 저장용
        self.timestamp = (pd.Timestamp.now(tz='Asia/Seoul')).strftime('%Y%m%d%H%M')
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 시드 설정
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    # 영어 문장 섞기
    def shuffled_english_contexts(self, df: pd.DataFrame) -> pd.DataFrame:
        print("=" * 80)
        print("본문 영어 번역 및 번호 섞은 후 저장")
        print("=" * 80)

        output_file = self.output_path / 'shuffled_english_contexts.csv'

        try:
            result_df = pd.read_csv(output_file, index_col='id')
            print("✓ 파일에서 로드")
        except:
            english_contexts = []

            for idx, row in df.iterrows():
                translated_content = translate_to_english(row['context'])
                print(translated_content, f'\n항목 {idx} 번역 완료')
                english_contexts.append(translated_content)

            # 결과 저장
            result_df = pd.DataFrame({'context': english_contexts})
            result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
            result_df.to_csv(output_file, index=True, index_label='id')

            print(f"✓ 번역 완료: {output_file}")

        return result_df

    # 다양한 노이즈 컨텍스트 생성
    def generate_noise_contexts(self, base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        print("=" * 80)
        print("노이즈 컨텍스트 생성")
        print("=" * 80)

        noise_dfs = {}
        sample_size = 20

        # 1. 무관한 법률 조항
        print("\n[1/4] 임의 컨택스트 생성")
        output_file = self.output_path / f'unrelated_contexts.csv'

        try:
            noise_dfs['unrelated_context'] = pd.read_csv(output_file, index_col='id')
            print("✓ 파일에서 로드")
        except:
            unrelated_contexts = generate_unrelated_contexts(self.config.num_unrelated_contexts)
            noise_dfs['unrelated_context'] = pd.DataFrame({'context': unrelated_contexts})
            noise_dfs['unrelated_context'].to_csv(output_file, index_label='id')
            print(f"✓ 저장: {output_file}")

        # 2. 유사 도메인 (샘플링하여 생성)
        print("\n[2/4] 유사 도메인 컨택스트 생성")
        output_file = self.output_path / f'similar_domain_contexts.csv'

        try:
            noise_dfs['similar_domain_context'] = pd.read_csv(output_file, index_col='id')
            print("✓ 파일에서 로드")
        except:
            similar_contexts = []

            for idx, row in base_df.head(sample_size).iterrows():
                print(f"  진행: {idx + 1}/{sample_size}")

                similar_context = generate_similar_domain_contexts(row['context'])
                print(similar_context)
                similar_contexts.append(similar_context)

            noise_dfs['similar_domain_context'] = pd.DataFrame({'context': similar_contexts})
            noise_dfs['similar_domain_context'].to_csv(output_file, index_label='id')
            print(f"✓ 저장: {output_file}")

        # 3. 모순 정보 (샘플링하여 생성)
        print("\n[3/4] 모순 정보 생성")
        output_file = self.output_path / f'contradictory_contexts.csv'

        try:
            noise_dfs['contradictory_context'] = pd.read_csv(output_file, index_col='id')
            print("✓ 파일에서 로드")
        except:
            contradictory_contexts = []

            for idx, row in base_df.head(sample_size).iterrows():
                print(f"  진행: {idx + 1}/{sample_size}")

                contradictory_context = generate_contradictory_context(row['context'], row['answer'])
                print(contradictory_context)
                contradictory_contexts.append(contradictory_context)

            noise_dfs['contradictory_context'] = pd.DataFrame({'context': contradictory_contexts})
            noise_dfs['contradictory_context'].to_csv(output_file, index_label='id')
            print(f"✓ 저장: {output_file}")

        # 4. 부분 겹침 (샘플링하여 생성)
        print("\n[4/4] 부분 겹침 정보 생성")
        output_file = self.output_path / 'partial_overlap_contexts.csv'

        try:
            noise_dfs['partial_overlap'] = pd.read_csv(output_file, index_col='id')
            print("✓ 파일에서 로드")
        except:
            partial_contexts = []

            for idx, row in base_df.head(sample_size).iterrows():
                partial_context = generate_partial_overlap(row['context'])
                print(partial_context)
                print(f"  진행: {idx}/{sample_size}")
                partial_contexts.append(partial_context)

            noise_dfs['partial_overlap'] = pd.DataFrame({'context': partial_contexts})
            noise_dfs['partial_overlap'].to_csv(output_file, index_label='id')
            print(f"✓ 저장: {output_file}")

        return noise_dfs

    # 노이즈가 추가된 데이터셋 생성
    def create_noisy_datasets(self, base_df: pd.DataFrame, shuffled_english_df: pd.DataFrame, noise_dfs: Dict[str, pd.DataFrame]) -> List[pd.DataFrame]:
        print("=" * 80)
        print("노이즈 데이터셋 생성")
        print("=" * 80)

        datasets = []

        for difficulty in self.config.difficulty_levels:
            print(f"\n난이도: {difficulty.name.upper()}")
            print(f"  설명: {difficulty.description}")

            for position in difficulty.positions:
                print(f"  위치 전략: {position.value}")

                dataset_rows = []

                for idx, row in base_df.iterrows():
                    # 노이즈 선택
                    noises = []

                    for noise_type in difficulty.noise_types:
                        if noise_type == NoiseType.SHUFFLED_ENGLISH:
                            # 영문 번역 사용
                            if idx < len(shuffled_english_df):
                                noises.append(shuffled_english_df.iloc[idx]['context'])
                        else:
                            # 다른 노이즈 타입
                            noise_key = noise_type.value

                            if noise_key in noise_dfs:
                                noise_df = noise_dfs[noise_key]
                                # 무작위 선택
                                noise_idx = random.randint(0, len(noise_df) - 1)
                                noises.append(noise_df.iloc[noise_idx]['context'])

                    # 노이즈 결합
                    combined_noise = '\n\n'.join([n for n in noises if n])

                    # 위치에 따라 배치
                    if position == NoisePosition.PREPEND:
                        noisy_context = self.noise_positioner.prepend(row['context'], combined_noise)

                    elif position == NoisePosition.APPEND:
                        noisy_context = self.noise_positioner.append(row['context'], combined_noise)

                    elif position == NoisePosition.SANDWICH:
                        if len(noises) >= 2:
                            noisy_context = self.noise_positioner.sandwich(row['context'], noises[0], noises[1])
                        else:
                            noisy_context = self.noise_positioner.sandwich(row['context'], combined_noise, combined_noise)

                    elif position == NoisePosition.INTERLEAVE:
                        noisy_context = self.noise_positioner.interleave(row['context'], noises)

                    else:  # RANDOM_INSERT
                        noisy_context = self.noise_positioner.random_insert(row['context'], combined_noise, num_insertions=3)

                    dataset_rows.append({
                        'id': idx,
                        'difficulty': difficulty.name,
                        'position': position.value,
                        'noise_types': ','.join([nt.value for nt in difficulty.noise_types]),
                        'original_context': row['context'],
                        'noisy_context': noisy_context,
                        'question': row['question'],
                        'answer': row['answer'],
                        'reason': row['reason']
                    })

                dataset_df = pd.DataFrame(dataset_rows)
                datasets.append(dataset_df)

                # 저장
                output_file = self.output_path / f'noisy_dataset_{difficulty.name}_{position.value}_{self.timestamp}.csv'
                dataset_df.to_csv(output_file, index=False)
                print(f"  ✓ 저장: {output_file}")

        return datasets

    # 평가 요약 데이터셋 생성
    def create_evaluation_summary(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        print("=" * 80)
        print("평가 요약 생성")
        print("=" * 80)

        summary_rows = []

        for dataset in datasets:
            if len(dataset) == 0:
                continue

            difficulty = dataset.iloc[0]['difficulty']
            position = dataset.iloc[0]['position']

            summary_rows.append({
                'difficulty': difficulty,
                'position': position,
                'num_samples': len(dataset),
                'noise_types': dataset.iloc[0]['noise_types'],
                'avg_context_length': dataset['noisy_context'].str.len().mean(),
                'avg_original_length': dataset['original_context'].str.len().mean(),
                'noise_ratio': (dataset['noisy_context'].str.len().mean() / dataset['original_context'].str.len().mean())
            })

        summary_df = pd.DataFrame(summary_rows)

        output_file = self.output_path / f'evaluation_summary_{self.timestamp}.csv'
        summary_df.to_csv(output_file, index=True, index_label='id')
        print(f"✓ 요약 저장: {output_file}")

        return summary_df

    # 전체 파이프라인 실행
    def run(self) -> Dict:
        print("=" * 80)
        print("LLM 추론 능력 평가 파이프라인 시작")
        print(f"시작 시간: {pd.Timestamp.now(tz='Asia/Seoul')}")
        print("=" * 80)

        #  데이터셋 준비
        base_df = pd.read_csv(self.config.base_context_file, index_col='id')

        # 영문 번역 및 순서 섞기
        shuffled_english_df = self.shuffled_english_contexts(base_df)

        # 노이즈 생성
        noise_dfs = self.generate_noise_contexts(base_df)

        # 노이즈 데이터셋 생성
        datasets = self.create_noisy_datasets(base_df, shuffled_english_df, noise_dfs)

        # 평가 요약
        summary_df = self.create_evaluation_summary(datasets)

        print("=" * 80)
        print("파이프라인 완료!")
        print(f"종료 시간: {pd.Timestamp.now(tz='Asia/Seoul')}")
        print(f"출력 디렉토리: {self.output_path}")
        print("=" * 80)

        return {
            'base_df': base_df,
            'english_df': shuffled_english_df,
            'noise_dfs': noise_dfs,
            'datasets': datasets,
            'summary_df': summary_df
        }


# ==================== 실행 ====================

# 설정
base_file_path = './dataset/base_contexts.csv'

config = PipelineConfig(
    base_context_file=base_file_path,
    output_dir='./output',
    num_unrelated_contexts=len(pd.read_csv(base_file_path)),
    random_seed=42
)

# 파이프라인 실행
pipeline = InferenceBenchmarkPipeline(config)
results = pipeline.run()

print("\n" + "=" * 80)
print("실행 완료!")
print("=" * 80)
print(f"\n생성된 데이터셋 수: {len(results['datasets'])}")
print(f"\n평가 요약:")
print(results['summary_df'].to_string())
