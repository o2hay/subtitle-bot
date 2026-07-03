#!/usr/bin/env python3
import os
import sys

import argparse
import subprocess
import gc
import torch
import qwen_asr
from qwen_asr import Qwen3ASRModel

# --- Monkeypatch to actively free memory after every chunk ---
def optimized_infer_asr_transformers(self, contexts, wavs, languages):
    outs = []
    texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]
    batch_size = self.max_inference_batch_size
    if batch_size is None or batch_size < 0:
        batch_size = len(texts)

    for i in range(0, len(texts), batch_size):
        sub_text = texts[i : i + batch_size]
        sub_wavs = wavs[i : i + batch_size]
        inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        outs.extend(list(decoded))
        
        # Actively release intermediate tensors and empty MPS cache
        del inputs
        del text_ids
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    return outs

qwen_asr.Qwen3ASRModel._infer_asr_transformers = optimized_infer_asr_transformers

original_align = qwen_asr.Qwen3ForcedAligner.align

def optimized_align(self, audio, text, language):
    res = original_align(self, audio, text, language)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return res

qwen_asr.Qwen3ForcedAligner.align = optimized_align
# Reduce chunk size from 180s to 30s for optimal accuracy (Whisper/Qwen standard) and to prevent missing speech
qwen_asr.inference.qwen3_asr.MAX_FORCE_ALIGN_INPUT_SECONDS = 30
# -------------------------------------------------------------

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def extract_audio(video_path: str, temp_audio_path: str):
    print(f"🎬 Extracting audio from {video_path}...")
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"🎵 Audio extracted successfully.")

def clean_token(token: str) -> str:
    # Qwen/Llama tokens sometimes use special characters for spaces
    return token.replace('Ġ', ' ').replace(' ', ' ')

def group_timestamps_to_srt(time_stamps, language: str, max_chars: int = 40, max_gap: float = 1.0) -> str:
    is_cjk = language in ["Chinese", "Japanese", "Cantonese"]
    
    segments = []
    current_segment_words = []
    current_start = None
    current_end = None
    current_text_len = 0
    
    for word_info in time_stamps:
        text = word_info.text
        start = word_info.start_time
        end = word_info.end_time
        
        # Clean the word
        cleaned_text = clean_token(text)
        if not cleaned_text.strip() and not is_cjk:
            continue
            
        if current_start is None:
            current_start = start
            current_end = end
            current_segment_words.append(cleaned_text)
            current_text_len = len(cleaned_text)
        else:
            gap = start - current_end
            if gap > max_gap or (current_text_len + len(cleaned_text) > max_chars):
                # Finalize current segment
                # Join logic
                if is_cjk:
                    seg_text = "".join(current_segment_words).strip()
                else:
                    # Construct text with clean spacing
                    seg_text = " ".join([w.strip() for w in current_segment_words if w.strip()])
                
                segments.append((current_start, current_end, seg_text))
                
                # Start new segment
                current_start = start
                current_end = end
                current_segment_words = [cleaned_text]
                current_text_len = len(cleaned_text)
            else:
                current_end = end
                current_segment_words.append(cleaned_text)
                current_text_len += len(cleaned_text)
                
    # Add last segment
    if current_start is not None:
        if is_cjk:
            seg_text = "".join(current_segment_words).strip()
        else:
            seg_text = " ".join([w.strip() for w in current_segment_words if w.strip()])
        segments.append((current_start, current_end, seg_text))
        
    # Format to SRT string
    srt_output = []
    for idx, (start, end, text) in enumerate(segments, 1):
        srt_output.append(f"{idx}")
        srt_output.append(f"{format_time(start)} --> {format_time(end)}")
        srt_output.append(text)
        srt_output.append("")
        
    return "\n".join(srt_output)

def load_qwen_model_with_fallback(model_name: str, aligner_name: str, device: str, max_batch_size: int = 1) -> Qwen3ASRModel:
    dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
    try:
        # 1. Try local files only on preferred device
        print(f"🚀 Loading models from local cache on {device} (dtype: {dtype}, max_batch_size: {max_batch_size})...")
        return Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=max_batch_size,
            forced_aligner=aligner_name,
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=device,
                local_files_only=True
            ),
            local_files_only=True
        )
    except Exception:
        try:
            # 2. Try online check/download on preferred device
            print(f"🌐 Local cache not found/incomplete. Checking/downloading from Hugging Face on {device} (dtype: {dtype}, max_batch_size: {max_batch_size})...")
            return Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=max_batch_size,
                forced_aligner=aligner_name,
                forced_aligner_kwargs=dict(
                    dtype=dtype,
                    device_map=device,
                    local_files_only=False
                ),
                local_files_only=False
            )
        except Exception as e_pref:
            print(f"⚠️ Loading on {device} failed: {e_pref}")
            if device != "cpu":
                print("🔄 Falling back to CPU...")
                try:
                    # 3. Try local files only on CPU
                    print(f"🚀 Loading models from local cache on cpu (max_batch_size: {max_batch_size})...")
                    return Qwen3ASRModel.from_pretrained(
                        model_name,
                        dtype=torch.float32,
                        device_map="cpu",
                        max_inference_batch_size=max_batch_size,
                        forced_aligner=aligner_name,
                        forced_aligner_kwargs=dict(
                            dtype=torch.float32,
                            device_map="cpu",
                            local_files_only=True
                        ),
                        local_files_only=True
                    )
                except Exception:
                    # 4. Try online check/download on CPU
                    print(f"🌐 Local cache not found/incomplete. Checking/downloading from Hugging Face on cpu (max_batch_size: {max_batch_size})...")
                    return Qwen3ASRModel.from_pretrained(
                        model_name,
                        dtype=torch.float32,
                        device_map="cpu",
                        max_inference_batch_size=max_batch_size,
                        forced_aligner=aligner_name,
                        forced_aligner_kwargs=dict(
                            dtype=torch.float32,
                            device_map="cpu",
                            local_files_only=False
                        ),
                        local_files_only=False
                    )
            raise e_pref

def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Speech-to-Text & Forced Alignment")
    parser.add_argument("input", type=str, help="Input video or audio file path")
    parser.add_argument("--output", type=str, default=None, help="Output SRT subtitle file path")
    parser.add_argument("--language", type=str, default="English", help="Audio language (e.g. English, Korean, Chinese)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"], help="Hardware device to run on")
    parser.add_argument("--max_chars", type=int, default=40, help="Max characters per subtitle line")
    parser.add_argument("--max_gap", type=float, default=1.0, help="Max silence gap (seconds) to split lines")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Max batch size for model inference (lower saves memory, e.g. 1 or 2)")
    
    args = parser.parse_args()
    
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
        
    # Determine output file path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_qwen_en.srt"
        
    # Audio extraction logic
    temp_audio_path = "temp_qwen3_input.wav"
    is_video = input_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.flv'))
    
    if is_video:
        try:
            extract_audio(input_path, temp_audio_path)
            audio_for_model = temp_audio_path
        except Exception as e:
            print(f"❌ Audio extraction failed: {e}")
            sys.exit(1)
    else:
        audio_for_model = input_path
        
    # Device setup
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
        
    # Load models
    try:
        model = load_qwen_model_with_fallback(
            model_name="Qwen/Qwen3-ASR-0.6B",
            aligner_name="Qwen/Qwen3-ForcedAligner-0.6B",
            device=device,
            max_batch_size=args.max_batch_size
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        sys.exit(1)
            
    print(f"🎙️ Transcribing and aligning (Language: {args.language})...")
    try:
        with torch.inference_mode():
            results = model.transcribe(
                audio=audio_for_model,
                language=args.language,
                return_time_stamps=True
            )
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        sys.exit(1)
        
    # Process transcription result
    if not results or not results[0].time_stamps:
        print("❌ Transcription completed but no timestamps were returned.")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        sys.exit(1)
        
    transcription = results[0]
    time_stamps = transcription.time_stamps
    
    # If time_stamps is list of lists (e.g. batch or nested), flatten it
    if isinstance(time_stamps, list) and len(time_stamps) > 0 and isinstance(time_stamps[0], list):
        time_stamps = time_stamps[0]
        
    print(f"📝 Transcribed text: {transcription.text}")
    print("⏳ Aligning and generating SRT...")
    
    srt_content = group_timestamps_to_srt(time_stamps, language=args.language, max_chars=args.max_chars, max_gap=args.max_gap)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
        
    print(f"✅ SRT Subtitles saved to: {output_path}")
    
    # Cleanup
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

if __name__ == "__main__":
    main()
