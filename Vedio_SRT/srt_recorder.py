"""
SRT Voice & Screen Recorder - FIXED audio pipeline
---------------------------------------------------
Root causes of "voice not recorded" in the original script:

1. The original captured audio via a *microphone input stream* while TTS
   played out the speakers. Unless the system has a loopback/"Stereo Mix"
   device set as the default mic, nothing is actually captured - the
   recorded WAV is silence/room noise, not the TTS voice.

2. Even when audio WAS captured, `embed_audio()` ran ffmpeg with
   `-shortest` on the FULL accumulated video but only the CURRENT
   segment's audio clip every single segment. That trims the whole
   video down to one segment's length each time, destroying every
   previously recorded segment.

Fix:
- Generate each segment's voice directly to a WAV file with
  pyttsx3's `save_to_file()` - no mic, no loopback dependency, and the
  audio is guaranteed to be the actual TTS voice.
- Pad/trim that WAV to match the segment's exact SRT duration with
  ffmpeg so it lines up with the recorded video segment.
- Only concatenate + embed audio ONCE, at final save, instead of
  re-running ffmpeg -shortest destructively after every segment.
"""

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pyttsx3
import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import srt
import sys
import warnings
import traceback
import shutil
import subprocess
import tempfile

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Windows-only: prevent floating panel from stealing focus ------
try:
    import ctypes
    from ctypes import wintypes

    _HAS_CTYPES = True
except Exception:
    _HAS_CTYPES = False

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020

_user32 = None
if _HAS_CTYPES and sys.platform == 'win32':
    try:
        _user32 = ctypes.windll.user32
        _user32.GetWindowLongW.restype = wintypes.LONG
        _user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        _user32.SetWindowLongW.restype = wintypes.LONG
        _user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
        _user32.SetWindowPos.restype = wintypes.BOOL
    except Exception:
        _user32 = None


class SRTPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("SRT Voice & Screen Recorder - Fixed Audio Pipeline")
        self.root.geometry("950x800")
        self.root.minsize(800, 600)

        # Core variables
        self.srt_file_path = None
        self.subtitles = []
        self.current_index = 0
        self.is_playing = False
        self.is_recording = False

        # Recording state
        self.video_writer = None
        self.segment_count = 0
        self.total_frames_recorded = 0
        self.total_video_duration = 0

        # Per-segment synthesized audio clips, in order: list of wav paths
        self.segment_audio_files = []

        # Voice settings
        self.speech_engine = None
        self.voice_id = None
        self.available_voices = []

        # Current subtitle state
        self.current_audio_text = ""
        self.current_subtitle_duration = 0
        self.subtitle_start_time = 0
        self.subtitle_end_time = 0

        # Floating panel state
        self.is_minimized = False
        self.floating_panel = None
        self.floating_status_var = None
        self._drag_data = {"x": 0, "y": 0}
        self.global_hotkeys_active = False

        # Settings
        self.freeze_frame_var = tk.BooleanVar(value=True)

        # Check for ffmpeg - REQUIRED now for padding/concatenating audio
        self._ffmpeg_path = shutil.which('ffmpeg')
        self._ffprobe_path = shutil.which('ffprobe')
        if not self._ffmpeg_path:
            print("WARNING: ffmpeg not found. Audio cannot be embedded without it.")

        # Per-segment sync diagnostics: (segment_number, target_duration,
        # frames_written, frames_expected, padded_frame_count)
        self.segment_sync_log = []

        # Setup GUI
        self.setup_gui()
        self.init_tts()
        self.setup_shortcuts()
        self.update_status("Ready - Load an SRT file to begin", "blue")

        # Start global hotkey listener
        self.start_global_hotkeys()

    # ------------------------------------------------------------------
    # TTS setup
    # ------------------------------------------------------------------
    def init_tts(self):
        try:
            self.speech_engine = pyttsx3.init()
            self.available_voices = self.speech_engine.getProperty('voices')
            self.populate_voice_list()
            self.set_default_voice()
            self.speech_engine.setProperty('rate', 150)
            self.speech_engine.setProperty('volume', 1.0)
            self.update_status("✅ TTS engine initialized", "green")
        except Exception as e:
            self.update_status(f"❌ TTS Error: {str(e)[:50]}", "red")
            self.speech_engine = None

    def populate_voice_list(self):
        if not self.available_voices:
            return
        voice_names = []
        for voice in self.available_voices:
            name = voice.name
            if 'female' in name.lower():
                name = f"👩 {name}"
            elif 'male' in name.lower():
                name = f"👨 {name}"
            voice_names.append(name)
        self.voice_combo['values'] = voice_names
        if voice_names:
            self.voice_combo.current(0)

    def set_default_voice(self):
        if not self.available_voices:
            return
        for voice in self.available_voices:
            name_lower = voice.name.lower()
            if 'male' in name_lower or 'david' in name_lower or 'mark' in name_lower:
                self.voice_id = voice.id
                self.speech_engine.setProperty('voice', voice.id)
                self.update_status(f"👨 Using male voice: {voice.name}", "green")
                return
        self.voice_id = self.available_voices[0].id
        self.speech_engine.setProperty('voice', self.available_voices[0].id)
        self.update_status(f"Using voice: {self.available_voices[0].name}", "green")

    def change_voice(self, event=None):
        if not self.speech_engine or not self.available_voices:
            return
        selection = self.voice_combo.current()
        if 0 <= selection < len(self.available_voices):
            voice = self.available_voices[selection]
            self.voice_id = voice.id
            self.speech_engine.setProperty('voice', voice.id)
            self.update_status(f"✅ Changed to voice: {voice.name}", "green")

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def setup_gui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        main_frame = ttk.Frame(scrollable_frame, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(title_frame, text="🎙️ SRT Recorder - Fixed Audio Pipeline",
                  font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT)
        ttk.Label(title_frame, text="v4.0",
                  font=("Segoe UI", 10), foreground="gray").pack(side=tk.RIGHT)

        file_frame = ttk.LabelFrame(main_frame, text="📁 File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(file_frame, text="SRT File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_srt).grid(row=0, column=2, padx=5)

        workflow_frame = ttk.LabelFrame(main_frame, text="📋 Workflow", padding="10")
        workflow_frame.pack(fill=tk.X, pady=(0, 10))
        instructions = """🎬 EXACT SRT DURATION RECORDING WITH VOICE (fixed pipeline):
1. Load SRT file
2. Select voice from dropdown
3. Press '▶ Preview' → hear the subtitle before recording
4. Press '🔴 Record' → synthesizes the voice straight to a WAV file
   (no mic/loopback needed) AND records the screen for the exact
   SRT segment duration
5. Repeat for each subtitle
6. Press '⏹ Stop & Save' → all voice clips are concatenated and
   embedded into the final video ONCE at the end"""
        ttk.Label(workflow_frame, text=instructions, font=("Segoe UI", 10),
                  justify=tk.LEFT).pack(anchor=tk.W)

        subtitle_frame = ttk.LabelFrame(main_frame, text="📝 Current Subtitle", padding="10")
        subtitle_frame.pack(fill=tk.X, pady=(0, 10))
        self.subtitle_display = tk.Text(subtitle_frame, height=6, font=("Segoe UI", 12),
                                        wrap=tk.WORD, bg="#f0f0f0", relief=tk.FLAT)
        self.subtitle_display.pack(fill=tk.X)
        self.subtitle_display.config(state=tk.DISABLED)

        progress_frame = ttk.LabelFrame(main_frame, text="📊 Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        progress_top = ttk.Frame(progress_frame)
        progress_top.pack(fill=tk.X)
        self.progress_var = tk.StringVar(value="0 / 0")
        ttk.Label(progress_top, textvariable=self.progress_var,
                  font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.percentage_var = tk.StringVar(value="0%")
        ttk.Label(progress_top, textvariable=self.percentage_var,
                  font=("Segoe UI", 11), foreground="gray").pack(side=tk.RIGHT)
        self.progress_bar = ttk.Progressbar(progress_frame, length=500, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        segment_frame = ttk.Frame(progress_frame)
        segment_frame.pack(fill=tk.X, pady=5)
        self.segment_var = tk.StringVar(value="Segments: 0")
        ttk.Label(segment_frame, textvariable=self.segment_var,
                  font=("Segoe UI", 10), foreground="blue").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="⏱ Duration: 0.0s")
        ttk.Label(segment_frame, textvariable=self.duration_var,
                  font=("Segoe UI", 10), foreground="green").pack(side=tk.LEFT, padx=(20, 0))
        self.status_indicator_var = tk.StringVar(value="⏸ Waiting")
        ttk.Label(segment_frame, textvariable=self.status_indicator_var,
                  font=("Segoe UI", 10, "bold"), foreground="orange").pack(side=tk.RIGHT)

        control_frame = ttk.LabelFrame(main_frame, text="🎮 Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()

        self.start_btn = ttk.Button(button_frame, text="▶ Preview (Ctrl+S)",
                                    command=self.preview_voice, width=16)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.record_btn = ttk.Button(button_frame, text="🔴 Record (Ctrl+R)",
                                     command=self.record_segment, width=18)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop & Save (S)",
                                   command=self.stop_all, width=16)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.reset_btn = ttk.Button(button_frame, text="↺ Reset",
                                    command=self.reset, width=12)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        self.minimize_btn = ttk.Button(button_frame, text="🗕 Mini (M)",
                                       command=self.toggle_minimize, width=12)
        self.minimize_btn.pack(side=tk.LEFT, padx=5)

        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(settings_frame, text="Select Voice:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.voice_combo = ttk.Combobox(settings_frame, width=40, state="readonly")
        self.voice_combo.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        self.voice_combo.bind('<<ComboboxSelected>>', self.change_voice)
        ttk.Button(settings_frame, text="🔊 Test Voice",
                   command=self.test_voice).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(settings_frame, text="Voice Speed:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(settings_frame, from_=0.5, to=2.0, variable=self.speed_var,
                  orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, padx=10, pady=5)
        self.speed_label = ttk.Label(settings_frame, textvariable=self.speed_var, width=5)
        self.speed_label.grid(row=1, column=2, pady=5)

        ttk.Label(settings_frame, text="Video FPS:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.fps_var = tk.IntVar(value=30)
        ttk.Spinbox(settings_frame, from_=10, to=60, textvariable=self.fps_var,
                    width=10).grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)

        ttk.Label(settings_frame, text="Output Video:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.output_var = tk.StringVar(value="recording_output.mp4")
        ttk.Entry(settings_frame, textvariable=self.output_var, width=40).grid(row=3, column=1, padx=10, pady=5)
        ttk.Button(settings_frame, text="Browse", command=self.browse_output).grid(row=3, column=2, pady=5)

        ttk.Checkbutton(settings_frame, text="🧊 Freeze Frame (one screenshot per segment)",
                        variable=self.freeze_frame_var).grid(row=4, column=1, sticky=tk.W, padx=10, pady=5)

        status_frame = ttk.LabelFrame(main_frame, text="📌 Status", padding="10")
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                      font=("Segoe UI", 11), foreground="blue")
        self.status_label.pack(anchor=tk.W)

        shortcut_frame = ttk.Frame(main_frame)
        shortcut_frame.pack(pady=10)
        ttk.Label(shortcut_frame, text="⌨️ Ctrl+S=Preview | Ctrl+R=Record | S=Stop & Save | M=Mini | ESC=Exit",
                  font=("Segoe UI", 9), foreground="gray").pack()

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def test_voice(self):
        if not self.speech_engine:
            messagebox.showerror("Error", "TTS engine not initialized!")
            return
        try:
            test_text = "This is a test of the selected voice."
            rate = int(150 * self.speed_var.get())
            self.speech_engine.setProperty('rate', rate)
            self.speech_engine.say(test_text)
            self.speech_engine.runAndWait()
            self.update_status("✅ Voice test completed", "green")
        except Exception as e:
            self.update_status(f"❌ Voice test failed: {str(e)[:50]}", "red")

    def setup_shortcuts(self):
        self.root.bind('<Control-s>', lambda e: self.preview_voice())
        self.root.bind('<Control-r>', lambda e: self.record_segment())
        self.root.bind('<S>', lambda e: self.stop_all())
        self.root.bind('<s>', lambda e: self.stop_all())
        self.root.bind('<m>', lambda e: self.toggle_minimize())
        self.root.bind('<M>', lambda e: self.toggle_minimize())
        self.root.bind('<Escape>', lambda e: self.on_closing())

    def start_global_hotkeys(self):
        if self.global_hotkeys_active:
            return
        self.global_hotkeys_active = True

        def hotkey_listener():
            try:
                import keyboard
                keyboard.add_hotkey('ctrl+s', self._safe_preview)
                keyboard.add_hotkey('ctrl+r', self._safe_record)
                keyboard.add_hotkey('s', self._safe_stop_all)
                keyboard.add_hotkey('m', self._safe_toggle_minimize)
                keyboard.add_hotkey('esc', self._safe_exit)
                while self.global_hotkeys_active:
                    time.sleep(0.1)
            except ImportError:
                print("Keyboard module not installed. Global hotkeys disabled.")
                self.update_status("Install 'keyboard' for global hotkeys", "orange")
            except Exception as e:
                print(f"Global hotkey error: {e}")

        threading.Thread(target=hotkey_listener, daemon=True).start()

    def _safe_preview(self):
        try:
            if self.floating_panel is not None or self.root.winfo_exists():
                self.root.after(0, self.preview_voice)
        except Exception:
            pass

    def _safe_record(self):
        try:
            if self.floating_panel is not None or self.root.winfo_exists():
                self.root.after(0, self.record_segment)
        except Exception:
            pass

    def _safe_stop_all(self):
        try:
            if self.floating_panel is not None or self.root.winfo_exists():
                self.root.after(0, self.stop_all)
        except Exception:
            pass

    def _safe_toggle_minimize(self):
        try:
            if self.floating_panel is not None or self.root.winfo_exists():
                self.root.after(0, self.toggle_minimize)
        except Exception:
            pass

    def _safe_exit(self):
        try:
            if self.floating_panel is not None or self.root.winfo_exists():
                self.root.after(0, self.on_closing)
        except Exception:
            pass

    def toggle_minimize(self, event=None):
        if self.is_minimized:
            self.restore_main_panel()
        else:
            self.minimize_to_floating_panel()

    def minimize_to_floating_panel(self):
        if self.is_minimized:
            return
        try:
            main_x = self.root.winfo_x()
            main_y = self.root.winfo_y()
        except Exception:
            main_x, main_y = 100, 100

        self.floating_panel = tk.Toplevel(self.root)
        self.floating_panel.withdraw()
        self.floating_panel.overrideredirect(True)
        self.floating_panel.attributes('-topmost', True)
        self.floating_panel.configure(bg="#222222", takefocus=0)
        self.floating_panel.geometry(f"+{max(main_x, 0)}+{max(main_y, 0)}")

        outer = tk.Frame(self.floating_panel, bg="#222222", bd=2, relief=tk.RAISED, takefocus=0)
        outer.pack(fill=tk.BOTH, expand=True)

        grip = tk.Label(outer, text="⠿", bg="#222222", fg="#aaaaaa",
                        font=("Segoe UI", 12, "bold"), cursor="fleur", padx=6, takefocus=0)
        grip.pack(side=tk.LEFT, fill=tk.Y)

        btn_style = {"bg": "#333333", "fg": "white", "activebackground": "#555555",
                     "activeforeground": "white", "relief": tk.FLAT, "bd": 0,
                     "font": ("Segoe UI", 10), "padx": 8, "pady": 4, "takefocus": 0,
                     "highlightthickness": 0}

        tk.Button(outer, text="▶ Preview", command=self.preview_voice, **btn_style).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(outer, text="🔴 Record", command=self.record_segment, **btn_style).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(outer, text="⏹ Stop", command=self.stop_all, **btn_style).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(outer, text="↺ Reset", command=self.reset, **btn_style).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(outer, text="🗖", command=self.restore_main_panel, **btn_style).pack(side=tk.LEFT, padx=(2, 6), pady=2)

        self.floating_status_var = tk.StringVar(value=self.status_var.get())
        status_lbl = tk.Label(outer, textvariable=self.floating_status_var,
                              bg="#222222", fg="#7fdc7f", font=("Segoe UI", 9),
                              padx=8, takefocus=0)
        status_lbl.pack(side=tk.LEFT)

        for w in [self.floating_panel, outer, grip, status_lbl]:
            w.bind("<ButtonPress-1>", self._start_drag_floating)
            w.bind("<B1-Motion>", self._do_drag_floating)

        self.floating_panel.bind('<Control-s>', lambda e: self.preview_voice())
        self.floating_panel.bind('<Control-r>', lambda e: self.record_segment())
        self.floating_panel.bind('<s>', lambda e: self.stop_all())
        self.floating_panel.bind('<S>', lambda e: self.stop_all())
        self.floating_panel.bind('<m>', lambda e: self.toggle_minimize())
        self.floating_panel.bind('<M>', lambda e: self.toggle_minimize())
        self.floating_panel.bind('<Escape>', lambda e: self.on_closing())

        self._make_noactivate(self.floating_panel)
        self.floating_panel.update_idletasks()
        self.floating_panel.deiconify()
        self.floating_panel.lift()

        self.root.withdraw()
        self.is_minimized = True

        if not self.global_hotkeys_active:
            self.start_global_hotkeys()

    def restore_main_panel(self):
        if not self.is_minimized:
            return
        if self.floating_panel is not None:
            try:
                self.floating_panel.destroy()
            except Exception:
                pass
            self.floating_panel = None
            self.floating_status_var = None
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.is_minimized = False

    def _make_noactivate(self, window):
        if _user32 is None or sys.platform != 'win32':
            return
        try:
            hwnd = wintypes.HWND(window.winfo_id())
            current_style = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            new_style = current_style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            _user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            _user32.SetWindowPos(
                hwnd, None, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED
            )
        except Exception as e:
            print(f"Could not apply no-activate window style: {e}")

    def _start_drag_floating(self, event):
        self._drag_data["x"] = event.x_root - self.floating_panel.winfo_x()
        self._drag_data["y"] = event.y_root - self.floating_panel.winfo_y()

    def _do_drag_floating(self, event):
        if self.floating_panel is None:
            return
        new_x = event.x_root - self._drag_data["x"]
        new_y = event.y_root - self._drag_data["y"]
        self.floating_panel.geometry(f"+{new_x}+{new_y}")

    def update_status(self, message, color="blue"):
        try:
            self.status_var.set(message)
            self.status_label.config(foreground=color)
            if self.floating_status_var is not None:
                self.floating_status_var.set(message)
            self.root.update_idletasks()
        except Exception:
            pass

    def browse_srt(self):
        file_path = filedialog.askopenfilename(
            title="Select SRT File",
            filetypes=[("SRT files", "*.srt"), ("All files", "*.*")]
        )
        if file_path:
            self.srt_file_path = file_path
            self.file_path_var.set(file_path)
            self.load_srt()

    def browse_output(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=".mp4",
            initialfile=f"recording_{timestamp}.mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if file_path:
            self.output_var.set(file_path)

    def load_srt(self):
        try:
            with open(self.srt_file_path, 'r', encoding='utf-8') as file:
                srt_content = file.read()
                self.subtitles = list(srt.parse(srt_content))
                self.current_index = 0
                self.segment_count = 0
                self.total_video_duration = 0
                self.segment_audio_files = []
                self.update_display()
                self.progress_bar['maximum'] = len(self.subtitles)
                self.update_progress()
                self.update_status(f"✅ Loaded {len(self.subtitles)} subtitles", "green")
                messagebox.showinfo("Success", f"Loaded {len(self.subtitles)} subtitles!")
        except Exception as e:
            self.update_status(f"❌ Failed to load SRT: {str(e)}", "red")
            messagebox.showerror("Error", str(e))

    def update_display(self):
        self.subtitle_display.config(state=tk.NORMAL)
        self.subtitle_display.delete(1.0, tk.END)

        if self.current_index < len(self.subtitles):
            sub = self.subtitles[self.current_index]
            start_sec = sub.start.total_seconds()
            end_sec = sub.end.total_seconds()
            duration = end_sec - start_sec
            self.current_subtitle_duration = duration
            self.subtitle_start_time = start_sec
            self.subtitle_end_time = end_sec

            info = f"[{self.current_index + 1}/{len(self.subtitles)}]\n"
            info += f"⏱ Start: {start_sec:.2f}s | End: {end_sec:.2f}s | Duration: {duration:.2f}s\n"
            info += "─" * 50 + "\n"
            info += sub.content
            self.subtitle_display.insert(tk.END, info)
            self.current_audio_text = sub.content
            self.duration_var.set(f"⏱ Duration: {duration:.2f}s")
        else:
            self.subtitle_display.insert(tk.END, "🏁 All subtitles processed!")
            self.current_audio_text = ""
            self.current_subtitle_duration = 0

        self.subtitle_display.config(state=tk.DISABLED)

    def speak_text(self, text):
        if not self.speech_engine or not text:
            return False
        try:
            if self.voice_id:
                self.speech_engine.setProperty('voice', self.voice_id)
            rate = int(150 * self.speed_var.get())
            self.speech_engine.setProperty('rate', rate)
            self.speech_engine.say(text)
            self.speech_engine.runAndWait()
            return True
        except Exception as e:
            print(f"TTS Error: {e}")
            return False

    def preview_voice(self):
        if not self.subtitles:
            messagebox.showwarning("Warning", "Please load an SRT file first")
            return
        if self.current_index >= len(self.subtitles):
            messagebox.showinfo("Complete", "All subtitles processed!")
            return
        if self.is_playing:
            return
        if self.is_recording:
            messagebox.showinfo("Info", "Currently recording. Please wait.")
            return

        self.is_playing = True
        self.start_btn.config(text="🔊 Previewing...")
        self.update_status(f"🔊 Previewing subtitle {self.current_index + 1}", "green")

        def preview():
            sub = self.subtitles[self.current_index]
            success = self.speak_text(sub.content)
            self.is_playing = False
            self.start_btn.config(text="▶ Preview (Ctrl+S)")
            if success:
                self.update_status("✅ Preview complete - Ready to Record", "green")
                self.record_btn.config(text=f"🔴 RECORD NOW! ({self.current_subtitle_duration:.1f}s)")
                self.record_btn.config(style="Accent.TButton")
                self.root.after(3000, lambda: self.record_btn.config(text="🔴 Record (Ctrl+R)"))
                self.root.after(3000, lambda: self.record_btn.config(style=""))
            else:
                self.update_status("❌ Preview failed", "red")

        threading.Thread(target=preview, daemon=True).start()

    # ------------------------------------------------------------------
    # Recording (fixed pipeline)
    # ------------------------------------------------------------------
    def record_segment(self):
        if self.is_recording:
            self.stop_recording()
            return
        if self.current_index >= len(self.subtitles):
            messagebox.showinfo("Complete", "All subtitles processed!")
            return
        if not self.output_var.get():
            messagebox.showwarning("Warning", "Please specify output file path!")
            return
        if not self._ffmpeg_path:
            messagebox.showerror("Error", "ffmpeg not found - it's required to embed audio into the video.")
            return

        self.start_recording()

    def _synthesize_segment_wav(self, text, target_duration):
        """Render TTS speech straight to a WAV file (no mic involved),
        then pad/trim it with ffmpeg so its length exactly matches the
        video segment's duration."""
        raw_wav = tempfile.NamedTemporaryFile(delete=False, suffix='_raw.wav')
        raw_wav.close()
        fixed_wav = tempfile.NamedTemporaryFile(delete=False, suffix='_fixed.wav')
        fixed_wav.close()

        try:
            if self.voice_id:
                self.speech_engine.setProperty('voice', self.voice_id)
            rate = int(150 * self.speed_var.get())
            self.speech_engine.setProperty('rate', rate)
            self.speech_engine.save_to_file(text, raw_wav.name)
            self.speech_engine.runAndWait()
        except Exception as e:
            print(f"TTS synthesis error: {e}")

        # Pad with silence if too short, trim if too long, so the audio
        # clip lines up exactly with the recorded video segment length.
        try:
            cmd = [
                self._ffmpeg_path, '-y',
                '-i', raw_wav.name,
                '-af', 'apad',
                '-t', f"{max(target_duration, 0.05):.3f}",
                fixed_wav.name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ffmpeg pad/trim error: {result.stderr}")
                shutil.copy(raw_wav.name, fixed_wav.name)
        finally:
            try:
                os.remove(raw_wav.name)
            except Exception:
                pass

        return fixed_wav.name

    def start_recording(self):
        try:
            screen = pyautogui.size()
            width, height = screen.width, screen.height

            if self.video_writer is None:
                output_path = self.output_var.get()
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    output_path, fourcc, self.fps_var.get(), (width, height)
                )
                if not self.video_writer.isOpened():
                    raise Exception("Failed to open video writer")

            screenshot = None
            if self.freeze_frame_var.get():
                try:
                    screenshot = ImageGrab.grab()
                except Exception:
                    screenshot = None

            self.is_recording = True
            self.record_btn.config(text="⏹ Stop Recording")
            self.record_btn.config(style="Accent.TButton")
            self.status_indicator_var.set("🔴 Recording with Voice...")

            segment_duration = self.current_subtitle_duration
            segment_text = self.current_audio_text
            self.segment_count += 1
            self.segment_var.set(f"Segments: {self.segment_count}")

            self.update_status(f"🔴 Recording segment {self.segment_count} - {segment_duration:.2f}s", "red")

            # Synthesize this segment's voice straight to a WAV file
            # (this is the actual fix - no mic capture involved at all)
            wav_path = self._synthesize_segment_wav(segment_text, segment_duration)
            self.segment_audio_files.append(wav_path)

            # Optionally play the voice out loud too, purely so you can
            # monitor it live - this playback is NOT what gets recorded.
            def play_voice_for_monitoring():
                try:
                    if self.voice_id:
                        self.speech_engine.setProperty('voice', self.voice_id)
                    rate = int(150 * self.speed_var.get())
                    self.speech_engine.setProperty('rate', rate)
                    self.speech_engine.say(segment_text)
                    self.speech_engine.runAndWait()
                except Exception as e:
                    print(f"Voice monitor playback error: {e}")

            threading.Thread(target=play_voice_for_monitoring, daemon=True).start()

            def record():
                try:
                    start_time = time.time()
                    frame_count = 0
                    fps = self.fps_var.get()
                    frame_interval = 1.0 / fps
                    # This is the number of frames this segment MUST end up
                    # with so its playback duration (frame_count / fps)
                    # exactly matches segment_duration - and therefore the
                    # exact-length audio clip already synthesized for it.
                    target_frames = max(1, round(segment_duration * fps))
                    last_frame = None
                    # Safety cap: if screen capture is badly lagging, don't
                    # let this segment run forever - bail after 3x its
                    # intended duration and pad the rest synthetically.
                    hard_deadline = segment_duration * 3 + 1.0

                    while self.is_recording and frame_count < target_frames:
                        elapsed = time.time() - start_time
                        if elapsed >= hard_deadline:
                            break
                        if elapsed >= frame_count * frame_interval:
                            if screenshot is not None:
                                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                            else:
                                frame = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)

                            self._draw_overlay(frame, segment_text, frame_count)
                            self.video_writer.write(frame)
                            last_frame = frame
                            frame_count += 1
                            self.total_frames_recorded += 1

                            progress = frame_count / target_frames * 100
                            if frame_count % (fps * 2) == 0:
                                self.root.after(0, lambda p=progress: self.update_status(
                                    f"🔴 Recording {p:.0f}% of segment {self.segment_count}", "red"))

                    # If real-time capture couldn't keep up (e.g. slow
                    # screen grabs at high FPS/resolution), pad out the
                    # remaining frames instantly by duplicating the last
                    # captured frame. This guarantees the segment's encoded
                    # duration == segment_duration exactly, matching the
                    # already-exact-length audio clip, instead of silently
                    # drifting shorter every segment.
                    padded = 0
                    if frame_count < target_frames and last_frame is not None:
                        while frame_count < target_frames:
                            self.video_writer.write(last_frame)
                            frame_count += 1
                            self.total_frames_recorded += 1
                            padded += 1
                        print(f"Segment {self.segment_count}: padded {padded} frame(s) "
                              f"to correct for slow capture")

                    self.segment_sync_log.append({
                        "segment": self.segment_count,
                        "target_duration": segment_duration,
                        "target_frames": target_frames,
                        "frames_written": frame_count,
                        "padded_frames": padded,
                    })

                    if self.is_recording:
                        self.root.after(0, self.stop_recording)

                except Exception as e:
                    print(f"Recording error: {e}")
                    self.root.after(0, lambda: self.update_status(f"❌ Recording error: {str(e)}", "red"))
                    self.root.after(0, self.stop_recording)

            threading.Thread(target=record, daemon=True).start()

        except Exception as e:
            self.update_status(f"❌ Failed to start recording: {str(e)}", "red")
            messagebox.showerror("Error", str(e))
            self.is_recording = False
            self.record_btn.config(text="🔴 Record (Ctrl+R)")

    def _draw_overlay(self, frame, text, frame_count):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 120), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h - 120), (w, h), (50, 50, 50), 1)

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            max_width = w - 100
            words = text.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                (text_width, _) = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
                if text_width > max_width:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))

            y_offset = h - 90 - (len(lines) - 1) * 35
            for i, line in enumerate(lines):
                y_pos = y_offset + i * 35
                cv2.putText(frame, line, (50, y_pos + 2), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(frame, line, (50, y_pos), font, font_scale, (255, 255, 255), thickness)

        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"🔴 REC {timestamp} | Seg: {self.segment_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    def stop_recording(self):
        """Stop recording the current segment. Audio is NOT embedded
        here anymore - it's just queued up in self.segment_audio_files
        and embedded once, at the very end, in stop_all()."""
        self.is_recording = False

        self.record_btn.config(text="🔴 Record (Ctrl+R)")
        self.status_indicator_var.set("⏸ Ready for Next")

        duration = self.current_subtitle_duration
        self.total_video_duration += duration

        self.update_status(f"✅ Segment {self.segment_count} recorded! ({duration:.2f}s)", "green")

        self.current_index += 1
        self.update_display()
        self.update_progress()

        if self.current_index < len(self.subtitles):
            self.update_status(f"✅ Ready for subtitle {self.current_index + 1} - Press Preview then Record", "green")
        else:
            self.update_status("✅ All subtitles recorded! Press Stop & Save", "green")

    def _finalize_audio_and_embed(self, video_path):
        """Concatenate every segment's WAV (in recording order) into one
        audio track, then embed it into the finished video in a single
        ffmpeg pass - this is what avoids the old bug where every
        segment's embed re-trimmed the whole video down to that
        segment's audio length."""
        if not self.segment_audio_files:
            return
        if not self._ffmpeg_path:
            self.update_status("⚠️ ffmpeg not found - audio not embedded", "orange")
            return
        if not os.path.exists(video_path):
            return

        concat_list = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8')
        combined_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        combined_audio.close()
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.close()

        try:
            for wav_path in self.segment_audio_files:
                escaped = wav_path.replace("'", "'\\''")
                concat_list.write(f"file '{escaped}'\n")
            concat_list.close()

            # Concatenate all segment WAVs into one continuous audio track
            concat_cmd = [
                self._ffmpeg_path, '-y',
                '-f', 'concat', '-safe', '0',
                '-i', concat_list.name,
                '-c', 'copy',
                combined_audio.name
            ]
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Audio concat error: {result.stderr}")
                self.update_status("⚠️ Failed to combine segment audio", "orange")
                return

            # Embed the single combined audio track into the full video ONCE
            embed_cmd = [
                self._ffmpeg_path, '-y',
                '-i', video_path,
                '-i', combined_audio.name,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                temp_video.name
            ]
            result = subprocess.run(embed_cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(temp_video.name):
                os.replace(temp_video.name, video_path)
                self.update_status("✅ Voice embedded into final video", "green")
            else:
                print(f"Final embed error: {result.stderr}")
                self.update_status("⚠️ Failed to embed audio into final video", "orange")

        finally:
            for f in [concat_list.name, combined_audio.name]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass
            try:
                if os.path.exists(temp_video.name):
                    os.remove(temp_video.name)
            except Exception:
                pass

    def _get_media_duration(self, path):
        """Return duration in seconds via ffprobe, or None if unavailable."""
        if not self._ffprobe_path or not os.path.exists(path):
            return None
        try:
            cmd = [
                self._ffprobe_path, '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            print(f"ffprobe error: {e}")
        return None

    def _get_stream_durations(self, path):
        """Return (video_duration, audio_duration) in seconds via ffprobe,
        so we can confirm they actually match each other, not just report
        one combined container duration."""
        if not self._ffprobe_path or not os.path.exists(path):
            return None, None
        video_dur, audio_dur = None, None
        try:
            for stream_type in ('v', 'a'):
                cmd = [
                    self._ffprobe_path, '-v', 'quiet',
                    '-select_streams', f'{stream_type}:0',
                    '-show_entries', 'stream=duration',
                    '-of', 'csv=p=0',
                    path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    val = float(result.stdout.strip())
                    if stream_type == 'v':
                        video_dur = val
                    else:
                        audio_dur = val
        except Exception as e:
            print(f"ffprobe stream error: {e}")
        return video_dur, audio_dur

    def stop_all(self):
        """Stop everything, embed the combined audio ONCE, and save."""
        self.is_recording = False
        time.sleep(0.3)

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.speech_engine:
            try:
                self.speech_engine.stop()
            except Exception:
                pass

        self.is_playing = False
        self.start_btn.config(text="▶ Preview (Ctrl+S)")
        self.record_btn.config(text="🔴 Record (Ctrl+R)")
        self.status_indicator_var.set("⏹ Stopped")

        output_path = self.output_var.get()

        if output_path and os.path.exists(output_path) and self.segment_count > 0:
            self.update_status("🔧 Combining and embedding voice audio...", "blue")
            self._finalize_audio_and_embed(output_path)

        # Clean up the per-segment wav files now that they're merged in
        for f in self.segment_audio_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        self.segment_audio_files = []

        if output_path and os.path.exists(output_path) and self.segment_count > 0:
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            summary = "🎬 Complete Video Saved!\n\n"
            summary += f"📁 File: {os.path.basename(output_path)}\n"
            summary += f"📊 Size: {file_size:.2f} MB\n"
            summary += f"🎙 Segments: {self.segment_count}\n"
            summary += f"⏱ Expected Duration: {self.total_video_duration:.2f}s\n"

            # Actually measure the finished file instead of assuming it worked
            video_dur, audio_dur = self._get_stream_durations(output_path)
            if video_dur is not None and audio_dur is not None:
                drift = abs(video_dur - audio_dur)
                summary += f"🎞 Measured video stream: {video_dur:.2f}s\n"
                summary += f"🔊 Measured audio stream: {audio_dur:.2f}s\n"
                if drift <= 0.15:
                    summary += f"📝 Sync check: ✓ In sync (drift {drift:.3f}s)\n"
                else:
                    summary += f"📝 Sync check: ⚠️ Drift of {drift:.2f}s detected\n"
            else:
                summary += "📝 Sync check: could not verify (ffprobe unavailable)\n"

            padded_total = sum(s["padded_frames"] for s in self.segment_sync_log)
            if padded_total > 0:
                summary += f"⚠️ Note: {padded_total} frame(s) were duplicated across " \
                           f"segments because screen capture couldn't keep up with the " \
                           f"target FPS - consider lowering FPS or enabling Freeze Frame.\n"

            self.update_status(f"✅ Video saved: {os.path.basename(output_path)}", "green")
            messagebox.showinfo("Recording Complete", summary)
        else:
            self.update_status("⏹ Stopped - No video saved", "orange")
            messagebox.showinfo("Stopped", "No video was saved.")

        if self.is_minimized:
            self.restore_main_panel()

    def reset(self):
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        for f in self.segment_audio_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        self.segment_audio_files = []

        self.current_index = 0
        self.segment_count = 0
        self.total_video_duration = 0
        self.total_frames_recorded = 0

        self.update_display()
        self.update_progress()
        self.segment_var.set("Segments: 0")
        self.duration_var.set("⏱ Duration: 0.0s")
        self.status_indicator_var.set("⏸ Waiting")
        self.update_status("↺ Reset to start", "blue")

        output_path = self.output_var.get()
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass

    def update_progress(self):
        total = len(self.subtitles)
        progress = min(self.current_index, total)
        self.progress_var.set(f"{progress} / {total}")
        self.percentage_var.set(f"{int((progress / total) * 100) if total > 0 else 0}%")
        self.progress_bar['value'] = progress

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.global_hotkeys_active = False
            self.is_recording = False
            if self.video_writer:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
            if self.speech_engine:
                try:
                    self.speech_engine.stop()
                except Exception:
                    pass
            self.root.destroy()
            sys.exit(0)


def main():
    try:
        root = tk.Tk()
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Accent.TButton", foreground="white", background="#0078D4")
        style.map("Accent.TButton", background=[('active', '#005A9E')])

        app = SRTPlayer(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()