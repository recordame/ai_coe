# inference_benchmark_pipeline.py
"""
LLM 추론 능력 평가를 위한 통합 파이프라인

기능:
1. 기본 데이터셋 검증
2. 영문 번역
3. 다양한 노이즈 생성 (무관한 법률, 유사 도메인, 모순 정보)
4. 노이즈 위치 및 강도 조절
5. 평가 데이터셋 생성
6. LLM 평가 실행
7. 결과 분석 및 리포트 생성
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging
import json
import time
from enum import Enum
import random
from openai import OpenAI
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


# ==================== 설정 클래스 ====================

class NoiseType(Enum):
    """노이즈 유형"""
    UNRELATED_LEGAL = "unrelated_legal"      # 무관한 법률 조항
    SIMILAR_DOMAIN = "similar_domain"        # 유사 도메인
    CONTRADICTORY = "contradictory"          # 모순 정보
    PARTIAL_OVERLAP = "partial_overlap"      # 부분 겹침
    SHUFFLED_ENGLISH = "shuffled_english"    # 셔플된 영문


class NoisePosition(Enum):
    """노이즈 위치 전략"""
    PREPEND = "prepend"           # 앞에 추가
    APPEND = "append"             # 뒤에 추가
    SANDWICH = "sandwich"         # 앞뒤로 감싸기
    INTERLEAVE = "interleave"     # 사이사이 삽입
    RANDOM_INSERT = "random"      # 무작위 삽입


@dataclass
class DifficultyConfig:
    """난이도 설정"""
    name: str
    noise_ratio: float
    noise_types: List[NoiseType]
    positions: List[NoisePosition]
    description: str


@dataclass
class PipelineConfig:
    """파이프라인 전체 설정"""
    # 파일 경로
    base_context_file: str
    output_dir: str = "./output"
    
    # API 설정
    api_key: str = "up_ZDvIwLQKhlVuIrSdimyXmwdFwtSxc"
    base_url: str = "https://api.upstage.ai/v1"
    translation_model: str = "upstage/solar-1-mini-chat"
    generation_model: str = "solar-pro2"
    
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


# ==================== 유틸리티 클래스 ====================

class Logger:
    """로깅 유틸리티"""
    
    @staticmethod
    def setup(output_dir: str) -> logging.Logger:
        """로거 설정"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # log_file = Path(output_dir) / f"pipeline_{(pd.Timestamp.now(tz='Asia/Seoul')).strftime('%Y%m%d%H%M')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                # logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)


class APIClient:
    """API 클라이언트"""
    
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def call(self, model: str, messages: List[Dict], temperature: float = 0, max_tokens: int = 16384, retry: int = 3) -> str:
        """API 호출 with 재시도"""
        for attempt in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return response.choices[0].message.content
            except Exception as e:
                if attempt == retry - 1:
                    raise

                time.sleep(attempt)
        return ""


# ==================== 노이즈 생성 클래스 ====================

class NoiseGenerator:
    """다양한 유형의 노이즈 생성"""
    
    def __init__(self, api_client: APIClient, config: PipelineConfig, logger: logging.Logger):
        self.api = api_client
        self.config = config
        self.logger = logger
    
    def generate_unrelated_legal(self, num: int) -> List[str]:
        """무관한 법률 조항 생성"""
        self.logger.info(f"무관한 법률 조항 {num}개 생성 중...")
        
        contexts = []
        for i in range(num):
            self.logger.info(f"  진행: {i+1}/{num}")
            
            messages = [
                {
                    "role": "system",
                    "content": "당신은 법률 전문가 입니다. 사용자가 '아무 법조항이나 알려줘' 라는 입력을 하면 "
                              "금융관련 법률 이 외의 법률(예, 헌법, 민법, 형법, 행정법 등)에서 법률과 하위 조, 항, 호, 목을 출력해줘! "
                              "그 어떤 법률 해설이나 마크다운 서식을 적용하지 말고 줄바꿈만 해줘. 단 매 생성마다 새로운 법안을 불러와줘!"
                },
                {
                    "role": "user",
                    "content": "아무 법조항이나 알려줘"
                }
            ]
            
            try:
                context = self.api.call(self.config.generation_model, messages, temperature=1)
                contexts.append(context)
            except Exception as e:
                self.logger.error(f"  생성 실패 (index {i}): {e}")
                contexts.append("")
        
        return contexts
    
    def generate_similar_domain(self, base_context: str) -> str:
        """유사 도메인 노이즈 생성"""
        messages = [
            {
                "role": "system",
                "content": "당신은 금융 법률 전문가입니다. 주어진 본문과 같은 금융 도메인이지만, "
                          "본문의 질문에 답하는 데 전혀 도움이 되지 않는 다른 금융 법률 조항을 생성해주세요."
            },
            {
                "role": "user",
                "content": f"""다음 본문과 유사하지만 무관한 금융 법률 조항을 생성해주세요.
                            본문:
                            {base_context}

                            요구사항:
                            - 금융 관련 용어를 사용할 것
                            - 본문과 비슷한 구조를 가질 것
                            - 하지만 본문의 질문과는 완전히 무관할 것
                            - 법률 조항 형식으로 작성할 것
                            """
            }
        ]
        
        return self.api.call(self.config.generation_model, messages, temperature=0.8)
    
    def generate_contradictory(self, base_context: str, answer: str) -> str:
        """모순 정보 생성"""
        messages = [
            {
                "role": "system",
                "content": "당신은 금융 법률 전문가입니다. 주어진 본문과 정답을 보고, "
                          "정답과 모순되지만 그럴듯해 보이는 법률 조항을 생성해주세요."
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
                            """
            }
        ]
        
        return self.api.call(self.config.generation_model, messages, temperature=0.8)
    
    def generate_partial_overlap(self, base_context: str) -> str:
        """부분 겹침 정보 생성"""
        messages = [
            {
                "role": "system",
                "content": "당신은 금융 법률 전문가입니다. 주어진 본문과 일부 단어나 개념은 겹치지만, "
                          "전체적으로는 다른 내용의 법률 조항을 생성해주세요."
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
                            """
            }
        ]
        
        return self.api.call(self.config.generation_model, messages, temperature=0.8)


class NoisePositioner:
    """노이즈 위치 조정"""
    
    @staticmethod
    def prepend(base: str, noise: str) -> str:
        """노이즈를 앞에 배치"""
        return f"{noise}\n\n{base}"
    
    @staticmethod
    def append(base: str, noise: str) -> str:
        """노이즈를 뒤에 배치"""
        return f"{base}\n\n{noise}"
    
    @staticmethod
    def sandwich(base: str, noise1: str, noise2: str) -> str:
        """노이즈로 본문을 감싸기"""
        return f"{noise1}\n\n{base}\n\n{noise2}"
    
    @staticmethod
    def interleave(base: str, noises: List[str]) -> str:
        """본문을 문장 단위로 나누고 노이즈를 사이사이 삽입"""
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
    
    @staticmethod
    def random_insert(base: str, noise: str, num_insertions: int = 3) -> str:
        """무작위 위치에 노이즈 삽입"""
        base_sentences = [s.strip() for s in base.split('.') if s.strip()]
        noise_sentences = [s.strip() for s in noise.split('.') if s.strip()]
        
        for _ in range(min(num_insertions, len(noise_sentences))):
            if noise_sentences:
                pos = random.randint(0, len(base_sentences))
                base_sentences.insert(pos, random.choice(noise_sentences))
        
        return '. '.join(base_sentences) + '.'


# ==================== 번역 클래스 ====================

class Translator:
    """텍스트 번역"""
    
    def __init__(self, api_client: APIClient, config: PipelineConfig, logger: logging.Logger):
        self.api = api_client
        self.config = config
        self.logger = logger
    
    def translate_to_english(self, text: str) -> str:
        """한국어를 영어로 번역"""
        messages = [
            {
                "role": "system",
                "content": "당신은 금융 법률용어 전문가로 한국어와 영어에 능통한 번역가이기도 합니다. "
                          "사용자가 입력한 법 조항을 친절하게 영어로 번역해주세요! "
                          "대답을 할 때는 이런저런 설명(예: Translate the following into English!) 붙이지 말고 번역문만 출력해주세요!!"
            },
            {
                "role": "user",
                "content": f"{text}\n"
            }
        ]
        
        return self.api.call(self.config.translation_model, messages, temperature=0)
    
    def translate_batch(self, texts: pd.Series) -> pd.Series:
        """배치 번역"""
        self.logger.info(f"텍스트 {len(texts)}개 번역 중...")
        
        translated:pd.Series = texts.copy()
        
        for idx in texts.index:
            self.logger.info(f"  진행: {idx+1}/{len(texts)}")
            try:
                translated.loc[idx] = self.translate_to_english(texts.loc[idx])
            except Exception as e:
                self.logger.error(f"  번역 실패 (index {idx}): {e}")
                translated.loc[idx] = ""
        
        return translated


# ==================== 평가 클래스 ====================

class ReasoningEvaluator:
    """추론 능력 평가"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.vectorizer = TfidfVectorizer()
    
    def exact_match(self, text1: str, text2: str) -> float:
        """정확히 일치하는지 확인"""
        return 1.0 if text1.strip() == text2.strip() else 0.0
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산 (TF-IDF 기반)"""
        try:
            if not text1 or not text2:
                return 0.0
            
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def extract_concepts(self, text: str) -> set:
        """텍스트에서 주요 개념 추출 (간단한 키워드 기반)"""
        # 간단한 구현: 명사 추출 대신 단어 기반
        words = set(text.split())
        # 불용어 제거 (간단한 버전)
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로', '으로'}
        return words - stopwords
    
    def detect_hallucination(self, answer: str, reasoning: str) -> float:
        """환각 탐지: 답변이 추론 과정에 없는 정보를 포함하는지"""
        answer_concepts = self.extract_concepts(answer)
        reasoning_concepts = self.extract_concepts(reasoning)
        
        if not answer_concepts:
            return 0.0
        
        hallucinated = len(answer_concepts - reasoning_concepts)
        total = len(answer_concepts)
        
        return hallucinated / total
    
    def calculate_noise_resistance(self, base_answer: str, noisy_answer: str) -> float:
        """노이즈 저항성 계산"""
        return self.semantic_similarity(base_answer, noisy_answer)
    
    def evaluate(self, base_answer: str, noisy_answer: str, 
                 base_reasoning: str, noisy_reasoning: str) -> Dict:
        """종합 평가"""
        return {
            "answer_exact_match": self.exact_match(base_answer, noisy_answer),
            "answer_similarity": self.semantic_similarity(base_answer, noisy_answer),
            "reasoning_similarity": self.semantic_similarity(base_reasoning, noisy_reasoning),
            "noise_resistance": self.calculate_noise_resistance(base_answer, noisy_answer),
            "hallucination_score": self.detect_hallucination(noisy_answer, noisy_reasoning)
        }


# ==================== 메인 파이프라인 ====================

class InferenceBenchmarkPipeline:
    """LLM 추론 능력 평가 통합 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = Logger.setup(config.output_dir)
        self.api_client = APIClient(config.api_key, config.base_url)
        self.translator = Translator(self.api_client, config, self.logger)
        self.noise_generator = NoiseGenerator(self.api_client, config, self.logger)
        self.noise_positioner = NoisePositioner()
        self.evaluator = ReasoningEvaluator(self.logger)
        
        # 결과 저장용
        self.timestamp = (pd.Timestamp.now(tz='Asia/Seoul')).strftime('%Y%m%d%H%M')
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 시드 설정
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
    def shuffled_english_contexts(self, df: pd.DataFrame) -> pd.DataFrame:
        """본문 영어 번역"""
        self.logger.info("=" * 80)
        self.logger.info("Step 2: 본문 영어 번역")
        self.logger.info("=" * 80)
        
        english_contexts = self.translator.translate_batch(df['context'])
        
        # 결과 저장
        result_df = pd.DataFrame({'context': english_contexts})
        print(result_df)
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(result_df)
        output_file = self.output_path / f'shuffled_english_contexts_{self.timestamp}.csv'
        result_df.to_csv(output_file, index=True, index_label='id')

        self.logger.info(f"✓ 번역 완료: {output_file}")
        
        return result_df
    
    def generate_noise_contexts(self, base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """다양한 노이즈 컨텍스트 생성"""
        self.logger.info("=" * 80)
        self.logger.info("Step 3: 노이즈 컨텍스트 생성")
        self.logger.info("=" * 80)
        
        noise_dfs = {}
        
        # 1. 무관한 법률 조항
        self.logger.info("\n[1/4] 무관한 법률 조항 생성")
        unrelated_contexts = self.noise_generator.generate_unrelated_legal(self.config.num_unrelated_contexts)
        noise_dfs['unrelated_legal'] = pd.DataFrame({'context': unrelated_contexts})
        
        output_file = self.output_path / f'unrelated_contexts_{self.timestamp}.csv'
        noise_dfs['unrelated_legal'].to_csv(output_file, index=True, index_label='id')
        self.logger.info(f"✓ 저장: {output_file}")
        
        # 2. 유사 도메인 (샘플링하여 생성)
        self.logger.info("\n[2/4] 유사 도메인 노이즈 생성")
        sample_size = min(20, len(base_df))
        similar_contexts = []
        
        for idx, row in base_df.head(sample_size).iterrows():
            self.logger.info(f"  진행: {idx+1}/{sample_size}")
            try:
                similar = self.noise_generator.generate_similar_domain(row['context'])
                similar_contexts.append(similar)
            except Exception as e:
                self.logger.error(f"  생성 실패: {e}")
                similar_contexts.append("")
        
        noise_dfs['similar_domain'] = pd.DataFrame({'context': similar_contexts})
        
        output_file = self.output_path / f'similar_domain_contexts_{self.timestamp}.csv'
        noise_dfs['similar_domain'].to_csv(output_file, index=True, index_label='id')
        self.logger.info(f"✓ 저장: {output_file}")
        
        # 3. 모순 정보 (샘플링하여 생성)
        self.logger.info("\n[3/4] 모순 정보 생성")
        contradictory_contexts = []
        
        for idx, row in base_df.head(sample_size).iterrows():
            self.logger.info(f"  진행: {idx+1}/{sample_size}")
            try:
                contradictory = self.noise_generator.generate_contradictory(
                    row['context'], row['answer']
                )
                contradictory_contexts.append(contradictory)
            except Exception as e:
                self.logger.error(f"  생성 실패: {e}")
                contradictory_contexts.append("")
        
        noise_dfs['contradictory'] = pd.DataFrame({'context': contradictory_contexts})
        
        output_file = self.output_path / f'contradictory_contexts_{self.timestamp}.csv'
        noise_dfs['contradictory'].to_csv(output_file, index=True, index_label='id')
        self.logger.info(f"✓ 저장: {output_file}")
        
        # 4. 부분 겹침 (샘플링하여 생성)
        self.logger.info("\n[4/4] 부분 겹침 정보 생성")
        partial_contexts = []
        
        for idx, row in base_df.head(sample_size).iterrows():
            self.logger.info(f"  진행: {idx+1}/{sample_size}")
            try:
                partial = self.noise_generator.generate_partial_overlap(row['context'])
                partial_contexts.append(partial)
            except Exception as e:
                self.logger.error(f"  생성 실패: {e}")
                partial_contexts.append("")
        
        noise_dfs['partial_overlap'] = pd.DataFrame({'context': partial_contexts})
        
        output_file = self.output_path / f'partial_overlap_contexts_{self.timestamp}.csv'
        noise_dfs['partial_overlap'].to_csv(output_file, index=True, index_label='id')
        self.logger.info(f"✓ 저장: {output_file}")
        
        return noise_dfs
    
    def create_noisy_datasets(self, base_df: pd.DataFrame, shuffled_english_df: pd.DataFrame, noise_dfs: Dict[str, pd.DataFrame]) -> List[pd.DataFrame]:
        """노이즈가 추가된 데이터셋 생성"""
        self.logger.info("=" * 80)
        self.logger.info("Step 4: 노이즈 데이터셋 생성")
        self.logger.info("=" * 80)
        
        datasets = []
        
        for difficulty in self.config.difficulty_levels:
            self.logger.info(f"\n난이도: {difficulty.name.upper()}")
            self.logger.info(f"  설명: {difficulty.description}")
            
            for position in difficulty.positions:
                self.logger.info(f"  위치 전략: {position.value}")
                
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
                dataset_df.to_csv(output_file, index=True, index_label='id')
                self.logger.info(f"  ✓ 저장: {output_file}")
        
        return datasets
    
    def create_evaluation_summary(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """평가 요약 데이터셋 생성"""
        self.logger.info("=" * 80)
        self.logger.info("Step 5: 평가 요약 생성")
        self.logger.info("=" * 80)
        
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
                'noise_ratio': (dataset['noisy_context'].str.len().mean() / 
                               dataset['original_context'].str.len().mean())
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        output_file = self.output_path / f'evaluation_summary_{self.timestamp}.csv'
        summary_df.to_csv(output_file, index=True, index_label='id')
        self.logger.info(f"✓ 요약 저장: {output_file}")
        
        return summary_df
    
    def run(self) -> Dict:
        """전체 파이프라인 실행"""
        self.logger.info("=" * 80)
        self.logger.info("LLM 추론 능력 평가 파이프라인 시작")
        self.logger.info(f"시작 시간: {pd.Timestamp.now(tz='Asia/Seoul')}")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: 데이터셋 준비
            base_df = pd.read_csv(self.config.base_context_file)
            
            # Step 2: 영문 번역
            shuffled_english_df = self.shuffled_english_contexts(base_df)
            
            # Step 3: 노이즈 생성
            noise_dfs = self.generate_noise_contexts(base_df)
            
            # Step 4: 노이즈 데이터셋 생성
            datasets = self.create_noisy_datasets(base_df, shuffled_english_df, noise_dfs)
            
            # Step 5: 평가 요약
            summary_df = self.create_evaluation_summary(datasets)
            
            self.logger.info("=" * 80)
            self.logger.info("파이프라인 완료!")
            self.logger.info(f"종료 시간: {pd.Timestamp.now(tz='Asia/Seoul')}")
            self.logger.info(f"출력 디렉토리: {self.output_path}")
            self.logger.info("=" * 80)
            
            return {
                'base_df': base_df,
                'english_df': shuffled_english_df,
                'noise_dfs': noise_dfs,
                'datasets': datasets,
                'summary_df': summary_df
            }
            
        except Exception as e:
            self.logger.error(f"파이프라인 실패: {e}")
            raise


# ==================== 실행 예제 ====================

# 설정
base_file_path = './dataset/base_contexts_202511062030.csv'

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