#!/bin/bash
set -euo pipefail

# ============ 사용자 설정 ============
VISIBLE_GPUS="0,1,2,3" 
DOMAINS=("Finance")
MAX_BATCH_SIZE=256
# ====================================

mkdir -p logs

echo "LLM 단일 추론을 시작합니다."
echo "  - 사용할 GPU IDs: ${VISIBLE_GPUS}"
echo ""


for DOMAIN in "${DOMAINS[@]}"; do
  echo "================================================="
  echo "도메인 [${DOMAIN}]에 대한 추론을 시작합니다."
  echo "================================================="

  LOG_FILE="logs/${DOMAIN}_gpus${VISIBLE_GPUS//,/_}.log"

  CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} \
    python make_question.py \
      --domain "${DOMAIN}" \
      --max_batch_size "${MAX_BATCH_SIZE}" \
      2>&1 | tee "${LOG_FILE}"

  echo ""
  echo "도메인 [${DOMAIN}]에 대한 작업이 완료되었습니다."
  echo ""
done

echo "모든 도메인에 대한 추론이 최종 완료되었습니다."