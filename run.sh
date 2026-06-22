#!/bin/zsh

if [ $# -lt 2 ]; then
    echo "❌ 사용법: $0 [출력파일명_확장자제외] \"[m3u8_URL]\""
    echo "💡 예시: $0 lecture_02 \"https://example.com/master.m3u8\""
    exit 1
fi

FILE_NAME=$1
M3U8_URL=$2
VENV_DIR=".venv"

echo "=================================================="
echo "🚀 1단계: m3u8 스트리밍 영상 다운로드 중..."
echo "=================================================="
ffmpeg -i "$M3U8_URL" -c copy "${FILE_NAME}.mp4"

if [ $? -ne 0 ]; then
    echo "❌ 다운로드 실패. URL 혹은 네트워크를 확인하세요."
    exit 1
fi

echo "\n=================================================="
echo "🚀 2단계: Python 가상환경 진입 및 Whisper 영어 자막 생성 중..."
echo "=================================================="
if [ -d "$VENV_DIR" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "⚠️ 가상환경이 없습니다. 새로 생성합니다."
    python3 -m venv "$VENV_DIR"
    source "${VENV_DIR}/bin/activate"
    pip install git+https://github.com/openai/whisper.git openai
fi

# MPS 오버플로우 에러 방지를 위해 CPU 모드 사용
whisper "${FILE_NAME}.mp4" --model base --output_format srt --language en --device cpu

if [ -f "${FILE_NAME}.srt" ]; then
    mv "${FILE_NAME}.srt" "${FILE_NAME}_en.srt"
else
    echo "❌ Whisper 자막 생성 실패."
    exit 1
fi

echo "\n=================================================="
echo "🚀 3단계: LLM(translate_srt.py) 호출하여 한글 번역 자막 생성 중..."
echo "=================================================="
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 에러: OPENAI_API_KEY 환경변수가 정의되지 않았습니다."
    exit 1
fi

python3 translator.py "$FILE_NAME"

if [ ! -f "${FILE_NAME}_kr.srt" ]; then
    echo "❌ 번역 실패: ${FILE_NAME}_kr.srt 파일이 생성되지 않았습니다."
    exit 1
fi

echo "\n=================================================="
echo "🚀 4단계: ffmpeg를 사용하여 한국어 자막 하드번 인코딩 중..."
echo "=================================================="
ffmpeg -i "${FILE_NAME}.mp4" -vf "subtitles=${FILE_NAME}_kr.srt" -c:a copy "${FILE_NAME}_final.mp4" -y

if [ $? -ne 0 ]; then
    echo "❌ 자막 하드번 합성 실패."
    exit 1
fi

echo "\n=================================================="
echo "🎉 모든 작업 완료!"
echo "📂 최종 결과물: ${FILE_NAME}_final.mp4"
echo "=================================================="
