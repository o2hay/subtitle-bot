import os
import sys
import time
from openai import OpenAI

# 1. 파일명 접두사 설정
file_prefix = sys.argv[1] if len(sys.argv) > 1 else "lecture_01"
input_file = f"{file_prefix}_en.srt"
output_file = f"{file_prefix}_kr.srt"

# 2. API 키 검증
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ 에러: OPENAI_API_KEY 환경변수를 찾을 수 없습니다.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

if not os.path.exists(input_file):
    print(f"❌ 번역 대상 파일이 없습니다: {input_file}")
    sys.exit(1)

print(f"🔄 {input_file} 분석 및 분할 번역을 시작합니다...")

# 3. SRT 파일을 자막 블록 단위로 파싱
with open(input_file, "r", encoding="utf-8") as f:
    raw_content = f.read().strip()

blocks = raw_content.split("\n\n")
total_blocks = len(blocks)
print(f"📊 총 {total_blocks}개의 자막 블록을 감지했습니다.")

# 4. 안전한 번역을 위해 15개 블록씩 쪼개서 번역 (누락 방지)
CHUNK_SIZE = 15
translated_blocks = []

system_prompt = (
    "You are a professional video translator. Translate the given English SRT subtitle chunk into Korean. "
    "Strictly maintain the EXACT number of subtitle blocks, sequence numbers, and timestamps. "
    "Do not omit, combine, or skip any timestamp lines. "
    "Only output the translated SRT content. Do not include any introduction, explanation, or notes."
)

for i in range(0, total_blocks, CHUNK_SIZE):
    chunk = blocks[i:i + CHUNK_SIZE]
    chunk_text = "\n\n".join(chunk)
    
    current_chunk_num = (i // CHUNK_SIZE) + 1
    total_chunks = (total_blocks + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"🔄 번역 진행 중: [{current_chunk_num}/{total_chunks}] 진행률: {int((i/total_blocks)*100)}%")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_text}
                ],
                temperature=0.1
            )
            
            translated_chunk = response.choices[0].message.content.strip()
            translated_blocks.append(translated_chunk)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ API 호출 최종 실패: {e}")
                sys.exit(1)
            print(f"⚠️ 오류 발생으로 재시도 중... ({attempt + 1}/{max_retries})")
            time.sleep(2)

# 5. 번역된 블록 합쳐서 저장
final_content = "\n\n".join(translated_blocks)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_content)

print(f"✅ 싱크 보정 번역 완료: {output_file}")
