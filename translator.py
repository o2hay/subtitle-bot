import os
import sys
import time
import json
import requests

# 1. 파일명 접두사 설정
file_prefix = sys.argv[1] if len(sys.argv) > 1 else "lecture_01"
input_file = f"{file_prefix}_en.srt"
output_file = f"{file_prefix}_kr.srt"

# 2. API 키 및 모델 검증
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODELS = os.environ.get("OPENROUTER_MODELS")

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
    raise ValueError("OPENROUTER_API_KEY가 설정되지 않았습니다.")
if not OPENROUTER_MODELS:
    raise ValueError("OPENROUTER_MODELS 환경 변수가 설정되지 않았습니다.")

if not os.path.exists(input_file):
    print(f"❌ 번역 대상 파일이 없습니다: {input_file}")
    sys.exit(1)

print(f"🔄 {input_file} 분석 및 분할 번역을 시작합니다...")

# 3. SRT 파일 파싱 함수 및 보조 함수 정의
import re

def parse_srt(srt_content):
    # Split blocks by double newlines, ignoring line ending differences
    raw_blocks = re.split(r'\n\s*\n', srt_content.replace('\r\n', '\n'))
    parsed_blocks = []
    
    for rb in raw_blocks:
        rb = rb.strip()
        if not rb:
            continue
        lines = rb.split('\n')
        if len(lines) >= 2:
            index_str = lines[0].strip()
            timestamp_str = lines[1].strip()
            text_str = '\n'.join(lines[2:])
            parsed_blocks.append({
                'index': index_str,
                'timestamp': timestamp_str,
                'text': text_str
            })
    return parsed_blocks

def extract_and_parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)

with open(input_file, "r", encoding="utf-8") as f:
    raw_content = f.read().strip()

blocks = parse_srt(raw_content)
total_blocks = len(blocks)
print(f"📊 총 {total_blocks}개의 자막 블록을 감지했습니다.")

# 4. 안전한 번역을 위해 15개 블록씩 쪼개서 번역 (누락 방지)
CHUNK_SIZE = 15
translated_blocks = []

system_prompt = (
    "You are an expert video translator specializing in English-to-Korean subtitles.\n"
    "Translate the English subtitle texts provided in JSON format into natural, fluent Korean.\n\n"
    "Guidelines:\n"
    "1. Translation Style: Translate in a natural, polite lecturing tone (존댓말, e.g., ~합니다, ~습니다, ~해요) that is easy for viewers to read.\n"
    "2. Flow: Avoid literal word-for-word translations. Rephrase idioms and sentences so they sound like natural Korean speech while preserving the original meaning.\n"
    "3. Format: You will receive a JSON object where the keys are subtitle IDs and the values are English text strings. You MUST return a JSON object with the EXACT SAME KEYS, where the values are the translated Korean text strings. Do not add, omit, or modify any keys.\n"
    "4. Newlines: Preserve newlines (\\n) in the text if they exist in the input string (e.g., to preserve line breaks within a subtitle block).\n"
    "5. Output: Return ONLY the raw JSON object. Do not include any introduction, explanations, markdown formatting blocks (like ```json), or notes."
)

# Parse models for payload
if OPENROUTER_MODELS.strip().startswith("["):
    try:
        models_payload = json.loads(OPENROUTER_MODELS)
    except Exception:
        models_payload = OPENROUTER_MODELS
else:
    models_payload = [m.strip() for m in OPENROUTER_MODELS.split(",") if m.strip()]

models_to_use = models_payload if models_payload else ["google/gemini-3.1-flash-lite"]

for i in range(0, total_blocks, CHUNK_SIZE):
    chunk_blocks = blocks[i:i + CHUNK_SIZE]
    
    # Create input JSON mapping subtitle indices to English texts
    chunk_dict = {block['index']: block['text'] for block in chunk_blocks}
    chunk_json_str = json.dumps(chunk_dict, ensure_ascii=False, indent=2)
    
    current_chunk_num = (i // CHUNK_SIZE) + 1
    total_chunks = (total_blocks + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"🔄 번역 진행 중: [{current_chunk_num}/{total_chunks}] 진행률: {int((i/total_blocks)*100)}%")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            headers_or = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"{system_prompt}\n\n[English Subtitles JSON]\n{chunk_json_str}"
            
            data = {
                "models": models_to_use if isinstance(models_to_use, list) else [models_to_use],
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            
            res_or = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers_or,
                json=data
            )
            
            # Auto-fallback: if response_format JSON mode is not supported by the model
            if res_or.status_code == 400 and "response_format" in res_or.text:
                print("⚠️ 선택된 모델이 response_format을 지원하지 않습니다. 일반 텍스트 모드로 재시도합니다...")
                del data["response_format"]
                res_or = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers_or,
                    json=data
                )
                
            if res_or.status_code != 200:
                raise ValueError(f"OpenRouter API 호출 에러 (Status: {res_or.status_code}): {res_or.text}")
                
            res_json = res_or.json()
            if "choices" in res_json and len(res_json["choices"]) > 0:
                translated_chunk_raw = res_json["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError(f"OpenRouter 응답에 choices가 없습니다: {res_or.text}")
                
            try:
                translated_dict = extract_and_parse_json(translated_chunk_raw)
            except Exception as e:
                raise ValueError(f"JSON 파싱 실패: {e}\nRaw Response: {translated_chunk_raw}")
            
            # Reconstruct the chunk SRT blocks
            chunk_translated_blocks = []
            missing_keys = []
            for block in chunk_blocks:
                idx = block['index']
                if idx not in translated_dict:
                    missing_keys.append(idx)
                # Fallback to English if translation is missing
                translated_text = translated_dict.get(idx, block['text'])
                
                # Clean up translation value (e.g. string strip)
                translated_text = str(translated_text).strip()
                
                # Format to a standard SRT block structure
                srt_block = f"{idx}\n{block['timestamp']}\n{translated_text}"
                chunk_translated_blocks.append(srt_block)
            
            if missing_keys:
                print(f"⚠️ 경고: 번역 결과에서 누락된 자막 키가 존재합니다 (영어 원문 사용): {missing_keys}")
                
            translated_blocks.extend(chunk_translated_blocks)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ API 호출 최종 실패: {e}")
                sys.exit(1)
            print(f"⚠️ 오류 발생으로 재시도 중... ({attempt + 1}/{max_retries}) | 에러: {e}")
            time.sleep(2)

# 5. 번역된 블록 합쳐서 저장
final_content = "\n\n".join(translated_blocks)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_content)

print(f"✅ 싱크 보정 번역 완료: {output_file}")

