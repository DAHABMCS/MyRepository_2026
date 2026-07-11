"""
exe_capture_srt_driven.py

Purpose:
  1. Parse an existing .srt subtitle file (with timing + text already defined).
  2. Launch the target .exe program.
  3. Record the screen for the SRT's total duration.
  4. Generate TTS voice for each subtitle line, placed EXACTLY at its SRT timestamp.
  5. Burn in the subtitle text on screen at the same timestamps.
  6. Produce one final video where video, voice, and on-screen text are all
     driven by the SRT timing -- the SRT is the single source of truth.

Runs on Windows (needs a real display).

Install dependencies:
    pip install mss opencv-python numpy pyttsx3 moviepy
"""

import re
import os
import time
import subprocess
import numpy as np
import cv2
import mss
import pyttsx3
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------

SRT_PATH = r"Episode1_Subtitles.srt"  # <-- path to your .srt file (place it next to this script,
                                       #     or give the full path e.g. C:\path\to\Episode1_Subtitles.srt)
EXE_PATH = r"C:\Program Files\Pyinstall Pro\dist\ProfessionalTradingPlatformV9.exe"  # <-- set this to your actual program
EXE_ARGS = []

FPS = 10
OUTPUT_VIDEO_RAW = "capture_raw.mp4"
FINAL_VIDEO = "capture_final.mp4"
CAPTURE_REGION = None                # None = full primary monitor

TTS_RATE = 165
TTS_VOICE_INDEX = None               # None = default system voice

BURN_IN_TEXT = True                  # also draw subtitle text on the video itself


# ----------------------------------------------------------------------
# 2. PARSE THE SRT FILE
# ----------------------------------------------------------------------

def srt_time_to_seconds(t):
    # Format: HH:MM:SS,mmm
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path):
    """
    Returns a list of dicts: [{"start": float, "end": float, "text": str}, ...]
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []

    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip() != ""]
        if len(lines) < 2:
            continue

        time_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                time_line = line
            elif not line.strip().isdigit():
                text_lines.append(line.strip())

        if not time_line:
            continue

        start_str, end_str = [p.strip() for p in time_line.split("-->")]
        start = srt_time_to_seconds(start_str)
        end = srt_time_to_seconds(end_str)
        text = " ".join(text_lines)

        entries.append({"start": start, "end": end, "text": text})

    return entries


# ----------------------------------------------------------------------
# 3. LAUNCH THE EXE
# ----------------------------------------------------------------------

def launch_exe():
    print(f"Launching: {EXE_PATH} {EXE_ARGS}")
    return subprocess.Popen([EXE_PATH] + EXE_ARGS)


# ----------------------------------------------------------------------
# 4. RECORD SCREEN FOR THE SRT'S TOTAL DURATION, BURNING IN TEXT LIVE
# ----------------------------------------------------------------------

def record_screen(duration, fps, output_path, region, subtitles, burn_in_text):
    with mss.mss() as sct:
        monitor = region if region else sct.monitors[1]
        width, height = monitor["width"], monitor["height"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_interval = 1.0 / fps
        start = time.time()
        next_frame_time = start

        print(f"Recording for {duration:.1f}s -> {output_path}")
        while time.time() - start < duration:
            now = time.time() - start
            frame = np.array(sct.grab(monitor))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if burn_in_text:
                caption = next(
                    (s["text"] for s in subtitles if s["start"] <= now < s["end"]), None
                )
                if caption:
                    overlay = frame_bgr.copy()
                    bar_height = 50
                    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
                    frame_bgr = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)
                    cv2.putText(
                        frame_bgr, caption, (20, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                    )

            out.write(frame_bgr)

            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        out.release()
    print("Recording finished.")


# ----------------------------------------------------------------------
# 5. GENERATE TTS VOICE FOR EACH SRT LINE, AT ITS EXACT TIMESTAMP
# ----------------------------------------------------------------------

def generate_voice_clips(subtitles, tmp_dir="tts_tmp"):
    os.makedirs(tmp_dir, exist_ok=True)
    engine = pyttsx3.init()
    engine.setProperty("rate", TTS_RATE)
    if TTS_VOICE_INDEX is not None:
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[TTS_VOICE_INDEX].id)

    timed_clips = []
    for i, sub in enumerate(subtitles):
        wav_path = os.path.join(tmp_dir, f"line_{i:03d}.wav")
        engine.save_to_file(sub["text"], wav_path)
        timed_clips.append((sub["start"], wav_path))

    engine.runAndWait()
    return timed_clips


def build_final_video(video_path, timed_clips, output_path):
    video = VideoFileClip(video_path)
    audio_clips = [
        AudioFileClip(wav_path).set_start(start_time)
        for start_time, wav_path in timed_clips
    ]
    composite_audio = CompositeAudioClip(audio_clips).set_duration(video.duration)
    final = video.set_audio(composite_audio)
    final.write_videofile(output_path, fps=video.fps, codec="libx264", audio_codec="aac")


# ----------------------------------------------------------------------
# 6. MAIN WORKFLOW
# ----------------------------------------------------------------------

def main():
    subtitles = parse_srt(SRT_PATH)
    if not subtitles:
        print("No subtitles parsed -- check SRT_PATH and file format.")
        return

    total_duration = max(s["end"] for s in subtitles) + 1.0  # +1s tail buffer
    print(f"Parsed {len(subtitles)} subtitle lines. Total duration: {total_duration:.1f}s")

    launch_exe()
    time.sleep(2)  # let the program's window open before recording

    record_screen(total_duration, FPS, OUTPUT_VIDEO_RAW, CAPTURE_REGION, subtitles, BURN_IN_TEXT)

    timed_clips = generate_voice_clips(subtitles)
    build_final_video(OUTPUT_VIDEO_RAW, timed_clips, FINAL_VIDEO)

    print(f"Done. Final video: {FINAL_VIDEO}")


if __name__ == "__main__":
    main()