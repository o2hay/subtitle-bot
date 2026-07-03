#!/bin/zsh

if [ $# -lt 1 ]; then
    echo "❌ 사용법: $0 [영상/음성파일_경로] [추가 옵션...]"
    echo "💡 예시: $0 lecture_01.mp4"
    echo "💡 예시: $0 lecture_01.mp4 --language Korean --device mps"
    exit 1
fi

INPUT_PATH=$1
shift # 첫 번째 인자 제거 후 나머지는 python 스크립트로 전달

if [ ! -f "$INPUT_PATH" ]; then
    echo "❌ 에러: 파일을 찾을 수 없습니다: $INPUT_PATH"
    exit 1
fi

CONDA_ENV_DIR="/opt/homebrew/Caskroom/miniconda/base/envs/subbot"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ 에러: conda 환경 'subbot'을 찾을 수 없습니다."
    echo "💡 '$CONDA_ENV_DIR' 경로를 확인하거나 환경이 올바르게 생성되었는지 확인하세요."
    exit 1
fi

echo "=================================================="
echo "🚀 Qwen3-ASR 0.6B & Forced Alignment 시작..."
echo "=================================================="
echo "📂 입력 파일: $INPUT_PATH"

# python 스크립트 실행 (나머지 인자 모두 전달)
$PYTHON_BIN qwen3_asr_align.py "$INPUT_PATH" "$@"

if [ $? -ne 0 ]; then
    echo "❌ 작업 중 오류가 발생했습니다."
    exit 1
fi

echo "=================================================="
echo "🎉 모든 작업 완료!"
echo "=================================================="
