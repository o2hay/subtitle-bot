#!/bin/zsh

if [ $# -lt 1 ]; then
    echo "❌ 사용법: $0 [영상파일_경로.mp4] ([출력파일명_확장자제외])"
    echo "💡 예시: $0 lecture_01.mp4"
    echo "💡 예시: $0 /path/to/video.mp4 my_lecture"
    exit 1
fi

INPUT_PATH=$1

if [ ! -f "$INPUT_PATH" ]; then
    echo "❌ 에러: 파일을 찾을 수 없습니다: $INPUT_PATH"
    exit 1
fi

# 출력용 파일명 접두사 설정
# 두 번째 인자가 있으면 사용하고, 없으면 입력 파일명에서 확장자를 제외한 이름을 사용
BASE_NAME=$(basename "$INPUT_PATH")
WHISPER_BASE_NAME="${BASE_NAME%.*}"

if [ -n "$2" ]; then
    FILE_NAME=$2
else
    FILE_NAME="$WHISPER_BASE_NAME"
fi

PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/subbot/bin/python"

echo "=================================================="
echo "🚀 1단계: Qwen3-ASR 영어 자막 생성 및 정렬 중..."
echo "=================================================="
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ 에러: conda 환경 'subbot'을 찾을 수 없습니다."
    exit 1
fi

$PYTHON_BIN qwen3_asr_align.py "$INPUT_PATH" --output "${FILE_NAME}_en.srt" --language English --device auto

if [ ! -f "${FILE_NAME}_en.srt" ]; then
    echo "❌ Qwen3-ASR 자막 생성 실패."
    exit 1
fi

echo "\n=================================================="
echo "🚀 2단계: LLM(translator.py) 호출하여 한글 번역 자막 생성 중..."
echo "=================================================="
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ 에러: OPENROUTER_API_KEY 환경변수가 정의되지 않았습니다."
    exit 1
fi
if [ -z "$OPENROUTER_MODELS" ]; then
    echo "❌ 에러: OPENROUTER_MODELS 환경변수가 정의되지 않았습니다."
    exit 1
fi

$PYTHON_BIN translator.py "$FILE_NAME"

if [ ! -f "${FILE_NAME}_kr.srt" ]; then
    echo "❌ 번역 실패: ${FILE_NAME}_kr.srt 파일이 생성되지 않았습니다."
    exit 1
fi

echo "\n=================================================="
echo "🚀 3단계: ffmpeg를 사용하여 영어 및 한국어 소프트 자막 트랙 추가 중..."
echo "=================================================="
ffmpeg -i "$INPUT_PATH" -i "${FILE_NAME}_en.srt" -i "${FILE_NAME}_kr.srt" \
  -map 0:v -map "0:a?" -map 1:s -map 2:s \
  -c copy -c:s mov_text \
  -metadata:s:s:0 language=eng -metadata:s:s:0 handler_name="English" \
  -metadata:s:s:1 language=kor -metadata:s:s:1 handler_name="Korean" \
  "${FILE_NAME}_final.mp4" -y

if [ $? -ne 0 ]; then
    echo "❌ 소프트 자막 합성 실패."
    exit 1
fi

echo "\n=================================================="
echo "🎉 모든 작업 완료!"
echo "📂 최종 결과물: ${FILE_NAME}_final.mp4"
echo "=================================================="
