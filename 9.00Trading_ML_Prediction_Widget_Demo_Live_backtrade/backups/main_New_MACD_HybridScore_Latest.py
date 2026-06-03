"""
Main entry point for Trading ML Application
Welcome screen persists until loading is fully complete.
"""

import multiprocessing
import os
import queue
import sys
import threading
import time

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

# ═══════════════════════════════════════════════════════════════════════════
# MOVE ALL HEAVY IMPORTS HERE (outside the background thread)
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import mplfinance as mpf
import ccxt

# Handle PyInstaller paths
if hasattr(sys, "_MEIPASS"):
    os.environ["PATH"] = sys._MEIPASS + os.pathsep + os.environ.get("PATH", "")


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED WELCOME + LOADING SPLASH SCREEN
# ═══════════════════════════════════════════════════════════════════════════

class SplashScreen:
    """
    Displays welcome.jpg and keeps the window open with a progress bar
    at the bottom until loading is fully complete.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Professional Trading with ML & AI v8.0")
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='#1a1a2e')

        # ── Load & display welcome image ───────────────────────────────────
        if hasattr(sys, "_MEIPASS"):
            image_path = os.path.join(sys._MEIPASS, 'images/WELCOME.jpg')
        else:
            image_path = os.path.join(os.path.dirname(__file__), 'images/WELCOME.jpg')

        self.img_width = 640
        self.img_height = 360

        try:
            img = Image.open(image_path)
            self.img_width, self.img_height = img.size

            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            max_w = int(screen_w * 0.6)
            max_h = int(screen_h * 0.6)

            if self.img_width > max_w or self.img_height > max_h:
                ratio = min(max_w / self.img_width, max_h / self.img_height)
                self.img_width = int(self.img_width * ratio)
                self.img_height = int(self.img_height * ratio)
                img = img.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(self.root, image=photo, bd=0, bg='#1a1a2e')
            img_label.image = photo  # keep reference
            img_label.pack()

        except Exception as e:
            print(f"Welcome image error: {e}")
            tk.Label(
                self.root,
                text="Professional Trading with ML & AI v8.0",
                font=('Arial', 16, 'bold'),
                bg='#1a1a2e', fg='white'
            ).pack(pady=40, padx=40)

        # ── Progress widgets below the image ──────────────────────────────
        footer = tk.Frame(self.root, bg='#1a1a2e')
        footer.pack(fill='x', padx=20, pady=(6, 14))

        self.status_label = tk.Label(
            footer, text="Initializing...",
            font=('Arial', 10), bg='#1a1a2e', fg='#00d4ff'
        )
        self.status_label.pack()

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(
            "Splash.Horizontal.TProgressbar",
            troughcolor='#16213e', background='#00d4ff', bordercolor='#1a1a2e'
        )
        self.progress = ttk.Progressbar(
            footer, style="Splash.Horizontal.TProgressbar",
            length=self.img_width - 40, mode='determinate'
        )
        self.progress.pack(pady=(4, 2))

        self.percent_label = tk.Label(
            footer, text="0%",
            font=('Arial', 9), bg='#1a1a2e', fg='#888888'
        )
        self.percent_label.pack()

        # ── Centre window on screen ────────────────────────────────────────
        self.root.update_idletasks()
        total_h = self.root.winfo_reqheight()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - self.img_width) // 2
        y = (screen_h - total_h) // 2
        self.root.geometry(f"{self.img_width}x{total_h}+{x}+{y}")

    def update(self, value: float, status: str = None):
        self.progress['value'] = value
        self.percent_label.config(text=f"{int(value)}%")
        if status:
            self.status_label.config(text=status)
        self.root.update()

    def close(self):
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════
# BACKGROUND LOADING (Now only does the final import and progress updates)
# ═══════════════════════════════════════════════════════════════════════════

def background_load_task(res_queue: queue.Queue):
    """Background task - only the final app import happens here."""
    try:
        time.sleep(0.1)

        # Send progress updates (imports already happened at top level)
        res_queue.put(("progress", (25, "Data libraries loaded")))
        time.sleep(0.2)

        res_queue.put(("progress", (45, "Visualization loaded")))
        time.sleep(0.2)

        res_queue.put(("progress", (60, "Exchange APIs loaded")))
        time.sleep(0.2)

        # Import the main TradingApp class (this is the heavy one)
        from App_MACD_AI_HybridScore_Latest import TradingApp
        res_queue.put(("progress", (90, "AI Engine loaded")))

        res_queue.put(("complete", TradingApp))

    except Exception as e:
        res_queue.put(("error", str(e)))


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Step 1: Show splash (welcome image + progress bar — stays open)
    splash = SplashScreen()

    # Step 2: Start background loading
    res_queue: queue.Queue = queue.Queue()
    load_thread = threading.Thread(
        target=background_load_task, args=(res_queue,), daemon=True
    )
    load_thread.start()

    # Step 3: Poll queue, keep UI alive until loading completes
    TradingAppClass = None
    while TradingAppClass is None:
        try:
            msg_type, data = res_queue.get_nowait()
            if msg_type == "progress":
                splash.update(data[0], data[1])
            elif msg_type == "complete":
                splash.update(100, "Ready!")
                TradingAppClass = data
            elif msg_type == "error":
                print(f"Loading error: {data}")
                sys.exit(1)
        except queue.Empty:
            # Animate progress bar slightly between 80–89% while AI loads
            current = splash.progress['value']
            if 80 <= current < 89:
                splash.update(current + 0.1)
            splash.root.update()
            time.sleep(0.05)

    time.sleep(0.6)  # Let user see "Ready! 100%" briefly
    splash.close()

    # Step 4: Launch main application
    root = tk.Tk()
    root.title("Professional Trading Platform with ML & AI v8.0")

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    w, h = 1280, 800
    root.geometry(f"{w}x{h}+{(screen_w - w) // 2}+{(screen_h - h) // 2}")

    print("Launching interface...")
    app = TradingAppClass(root)

    try:
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        os._exit(0)