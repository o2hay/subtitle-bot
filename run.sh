#!/bin/zsh

if [ $# -lt 2 ]; then
    echo "❌ 사용법: $0 [출력파일명_확장자제외] \"[m3u8_URL]\""
    echo "💡 예시: $0 lecture_02 \"https://example.com/master.m3u8\""
    exit 1
fi

FILE_NAME=$1
M3U8_URL=$2
PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/subbot/bin/python"

echo "=================================================="
echo "🚀 1단계: m3u8 스트리밍 영상 다운로드 중..."
echo "=================================================="
ffmpeg -i "$M3U8_URL" -c copy "${FILE_NAME}.mp4"

if [ $? -ne 0 ]; then
    echo "❌ 다운로드 실패. URL 혹은 네트워크를 확인하세요."
    exit 1
fi

echo "\n=================================================="
echo "🚀 2단계: Qwen3-ASR 영어 자막 생성 및 정렬 중..."
echo "=================================================="
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ 에러: conda 환경 'subbot'을 찾을 수 없습니다."
    exit 1
fi

$PYTHON_BIN qwen3_asr_align.py "${FILE_NAME}.mp4" --output "${FILE_NAME}_en.srt" --language English --device auto

if [ ! -f "${FILE_NAME}_en.srt" ]; then
    echo "❌ Qwen3-ASR 자막 생성 실패."
    exit 1
fi

echo "\n=================================================="
echo "🚀 3단계: LLM(translator.py) 호출하여 한글 번역 자막 생성 중..."
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
echo "🚀 4단계: ffmpeg를 사용하여 영어 및 한국어 소프트 자막 트랙 추가 중..."
echo "=================================================="
ffmpeg -i "${FILE_NAME}.mp4" -i "${FILE_NAME}_en.srt" -i "${FILE_NAME}_kr.srt" -c copy -c:s mov_text -map 0 -map 1:s -map 2:s -metadata:s:s:0 language=eng -metadata:s:s:0 handler_name="English" -metadata:s:s:1 language=kor -metadata:s:s:1 handler_name="Korean" "${FILE_NAME}_final.mp4" -y

if [ $? -ne 0 ]; then
    echo "❌ 소프트 자막 합성 실패."
    exit 1
fi

echo "\n=================================================="
echo "🎉 모든 작업 완료!"
echo "📂 최종 결과물: ${FILE_NAME}_final.mp4"
echo "=================================================="
