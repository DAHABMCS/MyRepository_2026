import time
import math
import tkinter as tk
from tkinter import ttk
from datetime import datetime

# ==========================================
# GLOBAL CONTROL VARIABLE
# ==========================================
RUN_TIMER = False  # Set to True to run, False to pause


class CircleTimer:
    def __init__(self, parent, size=200, bg_color='#2c3e50', fg_color='white'):
        self.parent = parent
        self.size = size
        self.bg_color = bg_color
        self.fg_color = fg_color

        # State Variables
        self.active = False
        self.angle = 0
        self.current_interval = 60  # Default 1 minute
        self.start_time = None
        self.elapsed_at_pause = 0
        self.total_elapsed_seconds = 0

        self.interval_strings = ['1m', '5m', '15m', '30m', '1H', '1D']
        self.interval_seconds_map = {
            '1m': 60, '5m': 300, '15m': 900,
            '30m': 1800, '1H': 3600, '1D': 86400
        }
        self.current_interval_index = 0

        # UI Elements
        self.canvas = tk.Canvas(parent, width=size, height=size, bg=bg_color, highlightthickness=0)
        self.canvas.pack(pady=10)

        self.draw_background()

        # The Progress "Pie" and Hand
        self.sector = self.canvas.create_arc(20, 20, size - 20, size - 20, start=90, extent=0,
                                             fill='#3498db', outline='#2980b9', style=tk.PIESLICE)
        self.hand = self.canvas.create_line(size // 2, size // 2, size // 2, 20, width=3, fill='red', arrow=tk.LAST)

        self.interval_text = self.canvas.create_text(size // 2, size // 2 + 40, text="1m",
                                                     fill="white", font=("Arial", 14, "bold"))

        self.digital_timer = tk.Label(
            parent,
            text="00:00:00",
            font=("Courier", 24, "bold"),
            bg="black",
            fg="#00FF00",
            padx=10,
            pady=5
        )
        self.digital_timer.pack(pady=10)

        # Start the background loops
        self.update_animation()
        self.update_digital_clock()

    def draw_background(self):
        center = self.size // 2
        radius = self.size // 2 - 10
        # Outer Ring
        self.canvas.create_oval(center - radius, center - radius, center + radius, center + radius,
                                outline=self.fg_color, width=3)
        # Center Dot
        self.canvas.create_oval(center - 4, center - 4, center + 4, center + 4,
                                fill='red', outline='white')

        # Hour Ticks
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            inner = center + (radius - 12) * math.cos(angle), center + (radius - 12) * math.sin(angle)
            outer = center + radius * math.cos(angle), center + radius * math.sin(angle)
            self.canvas.create_line(*inner, *outer, fill=self.fg_color, width=2)

    def update_animation(self):
        """Updates the visual circular progress IF RUN_TIMER is True."""
        if RUN_TIMER:
            if self.start_time is None:
                self.start_time = time.time() - self.elapsed_at_pause

            elapsed = time.time() - self.start_time
            progress = (elapsed % self.current_interval) / self.current_interval
            self.angle = progress * 360

            center = self.size // 2
            radius = self.size // 2 - 20

            # Update Pie Slice
            self.canvas.itemconfigure(self.sector, extent=-self.angle)

            # Update Needle Hand
            angle_rad = math.radians(self.angle - 90)
            end_x = center + radius * math.cos(angle_rad)
            end_y = center + radius * math.sin(angle_rad)
            self.canvas.coords(self.hand, center, center, end_x, end_y)
        else:
            # If paused, record where we were so we don't 'jump' when resuming
            if self.start_time is not None:
                self.elapsed_at_pause = time.time() - self.start_time
                self.start_time = None

        self.parent.after(50, self.update_animation)

    def update_digital_clock(self):
        """Increments the digital timer IF RUN_TIMER is True."""
        if RUN_TIMER:
            self.total_elapsed_seconds += 1

            hours, remainder = divmod(self.total_elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.digital_timer.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        self.parent.after(1000, self.update_digital_clock)


# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================
def toggle_timer():
    global RUN_TIMER
    RUN_TIMER = not RUN_TIMER
    btn_text.set("PAUSE TIMER" if RUN_TIMER else "START TIMER")
    status_label.config(text="Status: RUNNING" if RUN_TIMER else "Status: PAUSED",
                        fg="#00FF00" if RUN_TIMER else "#FF3333")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Python Variable-Controlled Clock")
    root.geometry("300x450")
    root.configure(bg='#2c3e50')

    # Header
    header = tk.Label(root, text="Master Control Clock", bg='#2c3e50', fg='white', font=("Arial", 12, "italic"))
    header.pack(pady=10)

    # Instantiate Timer
    timer_ui = CircleTimer(root)

    # Control Button
    btn_text = tk.StringVar(value="START TIMER")
    control_btn = tk.Button(root, textvariable=btn_text, command=toggle_timer,
                            font=("Arial", 10, "bold"), bg="#ecf0f1", width=15)
    control_btn.pack(pady=20)

    # Variable Status Display
    status_label = tk.Label(root, text="Status: PAUSED", bg='#2c3e50', fg='#FF3333', font=("Arial", 10))
    status_label.pack()

    root.mainloop()