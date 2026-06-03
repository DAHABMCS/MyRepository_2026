"""
PyInstaller Builder Pro
Supports full JSON and SPEC file workflows:
  JSON: Load JSON → Edit UI → Save JSON → Build (generates spec automatically)
  SPEC: Load Spec → Edit UI → Save Spec → Build from Spec (uses spec directly)
  DIRECT: Pick any existing .spec → Build from Spec (skips UI entirely)
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import json, os, re, subprocess, threading, shutil
from datetime import datetime
import sys


def minimize_console():
    try:
        import ctypes
        hwnd = ctypes.WinDLL('kernel32').GetConsoleWindow()
        if hwnd:
            ctypes.WinDLL('user32').ShowWindow(hwnd, 6)
    except Exception:
        pass

if sys.platform == 'win32':
    minimize_console()


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN LINK SANITISER
# Strips  [text](url)  →  text
# ─────────────────────────────────────────────────────────────────────────────
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]*\)')

def _strip_md(s: str) -> str:
    return _MD_LINK.sub(r'\1', s)


class ModernPyInstallerGUI:

    COLORS = {
        'bg':       '#f0f2f5',
        'primary':  '#1a56db',
        'secondary':'#3f37c9',
        'success':  '#057a55',
        'danger':   '#e02424',
        'dark':     '#111928',
        'light':    '#ffffff',
        'muted':    '#6b7280',
        'panel':    '#ffffff',
        'purple':   '#7c3aed',
        'teal':     '#0694a2',
        'orange':   '#c05621',
    }

    DEFAULT_CONFIG = {
        "script_path": "", "name": "MyApp",
        "onefile": True, "console": True, "icon": "",
        "collect_all": [], "collect_data": [],
        "hidden_imports": [], "binaries": [],
        "datas": [], "exclude_modules": [],
        "upx": False, "clean": True, "debug": False, "noconfirm": True,
    }

    def __init__(self, root):
        self.root = root
        self.root.title("PyInstaller Builder Pro")
        self.root.state('zoomed')
        try:
            self.root.attributes('-zoomed', True)
        except Exception:
            pass
        self.root.configure(bg=self.COLORS['bg'])
        self.config_file  = "pyinstaller_config.json"
        self._loaded_spec_path = None   # tracks last loaded/saved spec path
        self._build_ui()
        self._auto_load()

    # =========================================================================
    # UI BUILD
    # =========================================================================
    def _build_ui(self):
        self._make_header()
        self._make_body()
        self._make_footer()

    def _make_header(self):
        bar = tk.Frame(self.root, bg=self.COLORS['primary'], height=80)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="🚀  PyInstaller Builder Pro",
                 font=('Segoe UI', 26, 'bold'),
                 bg=self.COLORS['primary'], fg='white').pack(pady=(14, 0))
        tk.Label(bar, text="JSON · Spec · collect_all · DLL-safe builds",
                 font=('Segoe UI', 10),
                 bg=self.COLORS['primary'], fg='#bfdbfe').pack()

    def _make_body(self):
        body = tk.Frame(self.root, bg=self.COLORS['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(body, bg=self.COLORS['panel'], relief=tk.SOLID, bd=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        right = tk.Frame(body, bg=self.COLORS['panel'], relief=tk.SOLID, bd=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._make_settings(left)
        self._make_output(right)

    def _make_settings(self, parent):
        tb = tk.Frame(parent, bg=self.COLORS['primary'])
        tb.pack(fill=tk.X)
        tk.Label(tb, text="⚙️  Configuration",
                 font=('Segoe UI', 15, 'bold'),
                 bg=self.COLORS['primary'], fg='white', pady=10).pack()
        canvas = tk.Canvas(parent, bg=self.COLORS['panel'], highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        sf = tk.Frame(canvas, bg=self.COLORS['panel'])
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._populate(sf)

    def _populate(self, P):
        # Script
        self._sec(P, "📄  Script File  *")
        row = tk.Frame(P, bg=self.COLORS['panel'])
        row.pack(fill=tk.X, padx=20, pady=4)
        self.script_entry = self._entry(row, 52)
        self.script_entry.pack(side=tk.LEFT, ipady=7, padx=(0, 8))
        self._btn(row, "Browse", self.browse_script).pack(side=tk.LEFT)

        # Name
        self._sec(P, "🏷️  Application Name  *")
        self.name_entry = self._entry(P, 40)
        self.name_entry.pack(anchor='w', padx=20, pady=4, ipady=7)
        self.name_entry.insert(0, "MyApp")

        # Icon
        self._sec(P, "🎨  Icon  (optional)")
        row = tk.Frame(P, bg=self.COLORS['panel'])
        row.pack(fill=tk.X, padx=20, pady=4)
        self.icon_entry = self._entry(row, 52)
        self.icon_entry.pack(side=tk.LEFT, ipady=7, padx=(0, 8))
        self._btn(row, "Browse", self.browse_icon, self.COLORS['secondary']).pack(side=tk.LEFT)

        # Options
        self._sec(P, "⚡  Build Options")
        opts = tk.Frame(P, bg=self.COLORS['panel'])
        opts.pack(fill=tk.X, padx=20, pady=6)
        self.onefile_var   = tk.BooleanVar(value=True)
        self.console_var   = tk.BooleanVar(value=True)
        self.upx_var       = tk.BooleanVar(value=False)
        self.clean_var     = tk.BooleanVar(value=True)
        self.debug_var     = tk.BooleanVar(value=False)
        self.noconfirm_var = tk.BooleanVar(value=True)
        self._chk(opts, "📦 One File",       self.onefile_var,   "Bundle into a single .exe")
        self._chk(opts, "🖥️ Console Window",  self.console_var,   "Show terminal window")
        self._chk(opts, "🗜️ UPX Compression", self.upx_var,       "Keep OFF to avoid DLL corruption")
        self._chk(opts, "🧹 Clean Build",      self.clean_var,     "Remove temp files before build")
        self._chk(opts, "🐛 Debug Mode",       self.debug_var,     "Verbose PyInstaller output")
        self._chk(opts, "✅ No Confirm",       self.noconfirm_var, "Skip overwrite prompts")

        # collect_all
        self._sec(P, "🔬  collect_all  ← fixes numpy / scipy / sklearn DLL errors")
        tk.Label(P, text="One package per line  (numpy  scipy  pandas  sklearn  tensorflow  keras  talib  backtesting ...)",
                 font=('Segoe UI', 9), bg=self.COLORS['panel'],
                 fg=self.COLORS['muted']).pack(anchor='w', padx=20, pady=(0, 3))
        self.collect_all_text = self._textbox(P, 5)
        self.collect_all_text.pack(anchor='w', padx=20, pady=4)
        self._std_btns(P, self.collect_all_text, label="Package:")

        # collect_data
        self._sec(P, "📦  collect_data  (data + hidden imports, no DLLs — e.g. xgboost)")
        self.collect_data_text = self._textbox(P, 3)
        self.collect_data_text.pack(anchor='w', padx=20, pady=4)
        self._std_btns(P, self.collect_data_text, label="Package:", show_clear=False)

        # Hidden imports
        self._sec(P, "📦  Hidden Imports")
        self.hidden_text = self._textbox(P, 7)
        self.hidden_text.pack(anchor='w', padx=20, pady=4)
        self._std_btns(P, self.hidden_text, label="Module:")

        # Binaries
        self._sec(P, "🔗  Binaries  (explicit DLLs — format: src_path;dest_folder)")
        tk.Label(P, text="Example:  C:/Python311/site-packages/numpy/.libs;numpy/.libs",
                 font=('Segoe UI', 9), bg=self.COLORS['panel'],
                 fg=self.COLORS['muted']).pack(anchor='w', padx=20, pady=(0, 3))
        self.binaries_text = self._textbox(P, 4)
        self.binaries_text.pack(anchor='w', padx=20, pady=4)
        row = tk.Frame(P, bg=self.COLORS['panel'])
        row.pack(anchor='w', padx=20, pady=(0, 8))
        self._btn(row, "📁 Folder", self._add_binary_folder).pack(side=tk.LEFT, padx=3)
        self._btn(row, "📄 File",   self._add_binary_file).pack(side=tk.LEFT, padx=3)
        self._btn(row, "❌ Remove", lambda: self._remove_line(self.binaries_text)).pack(side=tk.LEFT, padx=3)

        # Datas
        self._sec(P, "📁  Data Files  (format: src_path;dest_folder)")
        self.datas_text = self._textbox(P, 6)
        self.datas_text.pack(anchor='w', padx=20, pady=4)
        row = tk.Frame(P, bg=self.COLORS['panel'])
        row.pack(anchor='w', padx=20, pady=(0, 8))
        self._btn(row, "📄 Files",   self._add_data_files).pack(side=tk.LEFT, padx=3)
        self._btn(row, "📁 Folder",  self._add_data_folder).pack(side=tk.LEFT, padx=3)
        self._btn(row, "❌ Remove",  lambda: self._remove_line(self.datas_text)).pack(side=tk.LEFT, padx=3)
        self._btn(row, "🗑 Clear",   lambda: self.datas_text.delete('1.0', tk.END)).pack(side=tk.LEFT, padx=3)

        # Excludes
        self._sec(P, "🚫  Exclude Modules")
        self.exclude_text = self._textbox(P, 4)
        self.exclude_text.pack(anchor='w', padx=20, pady=4)
        self._std_btns(P, self.exclude_text, label="Module:", show_clear=False)

        tk.Frame(P, height=20, bg=self.COLORS['panel']).pack()

    def _std_btns(self, parent, widget, label="Value:", show_clear=True):
        row = tk.Frame(parent, bg=self.COLORS['panel'])
        row.pack(anchor='w', padx=20, pady=(0, 8))
        self._btn(row, "➕ Add",    lambda: self._dialog_append(widget, label)).pack(side=tk.LEFT, padx=3)
        self._btn(row, "❌ Remove", lambda: self._remove_line(widget)).pack(side=tk.LEFT, padx=3)
        if show_clear:
            self._btn(row, "🗑 Clear", lambda: widget.delete('1.0', tk.END)).pack(side=tk.LEFT, padx=3)

    def _make_output(self, parent):
        tb = tk.Frame(parent, bg=self.COLORS['success'])
        tb.pack(fill=tk.X)
        tk.Label(tb, text="📊  Build Output",
                 font=('Segoe UI', 15, 'bold'),
                 bg=self.COLORS['success'], fg='white', pady=10).pack()
        self.output_text = scrolledtext.ScrolledText(
            parent, font=('Consolas', 11),
            bg='#1a1d23', fg='#d4d4d4',
            relief=tk.FLAT, padx=10, pady=10)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._log("═"*55 + "\n  PyInstaller Builder Pro\n  JSON + Spec full workflow support\n" + "═"*55 + "\n\n")
        row = tk.Frame(parent, bg=self.COLORS['panel'])
        row.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._btn(row, "🗑 Clear",      lambda: self.output_text.delete('1.0', tk.END),
                  self.COLORS['muted']).pack(side=tk.LEFT, padx=3)
        self._btn(row, "📂 Dist",   self._open_dist,     self.COLORS['primary']).pack(side=tk.LEFT, padx=3)
        self._btn(row, "📂 Build",  self._open_build,    self.COLORS['secondary']).pack(side=tk.LEFT, padx=3)
        self._btn(row, "🧹 Clean",  self._clean_folders, self.COLORS['danger']).pack(side=tk.RIGHT, padx=3)

    def _make_footer(self):
        footer = tk.Frame(self.root, bg=self.COLORS['panel'], relief=tk.SOLID, bd=1)
        footer.pack(fill=tk.X, padx=10, pady=(0, 10))

        # ── Row 1: labels ──────────────────────────────────────────────────────
        lbl_row = tk.Frame(footer, bg=self.COLORS['panel'])
        lbl_row.pack(fill=tk.X, padx=12, pady=(6, 0))

        tk.Label(lbl_row, text="── JSON workflow ──────────────────────",
                 font=('Segoe UI', 8), bg=self.COLORS['panel'],
                 fg=self.COLORS['muted']).pack(side=tk.LEFT)
        tk.Label(lbl_row, text="── Spec workflow ─────────────────────────────────────────",
                 font=('Segoe UI', 8), bg=self.COLORS['panel'],
                 fg=self.COLORS['muted']).pack(side=tk.LEFT, padx=(20, 0))

        # ── Row 2: buttons ─────────────────────────────────────────────────────
        btn_row = tk.Frame(footer, bg=self.COLORS['panel'])
        btn_row.pack(fill=tk.X, padx=10, pady=(2, 8))

        # JSON
        self._fbtn(btn_row, "💾 Save JSON",   self._save_json,       self.COLORS['secondary']).pack(side=tk.LEFT, padx=3)
        self._fbtn(btn_row, "📂 Load JSON",   self._load_json,       self.COLORS['secondary']).pack(side=tk.LEFT, padx=3)

        ttk.Separator(btn_row, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Spec
        self._fbtn(btn_row, "📋 Load Spec",   self._load_spec,       self.COLORS['purple']).pack(side=tk.LEFT, padx=3)
        self._fbtn(btn_row, "💾 Save Spec",   self._save_spec,       self.COLORS['teal']).pack(side=tk.LEFT, padx=3)
        self._fbtn(btn_row, "📝 Preview Spec",self._preview_spec,    '#6f42c1').pack(side=tk.LEFT, padx=3)
        self._fbtn(btn_row, "🚀 Build from Spec", self._build_from_spec, self.COLORS['orange']).pack(side=tk.LEFT, padx=3)

        ttk.Separator(btn_row, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Build from UI
        self._fbtn(btn_row, "🔨 Build EXE",   self._build,           self.COLORS['success'], large=True).pack(side=tk.LEFT, padx=3)

    # =========================================================================
    # WIDGET HELPERS
    # =========================================================================
    def _sec(self, p, t):
        tk.Label(p, text=t, font=('Segoe UI', 12, 'bold'),
                 bg=self.COLORS['panel'], fg=self.COLORS['dark'],
                 anchor='w').pack(fill=tk.X, padx=20, pady=(14, 3))

    def _entry(self, p, w=50):
        return tk.Entry(p, font=('Segoe UI', 11), bg='white',
                        relief=tk.SOLID, bd=1, width=w)

    def _textbox(self, p, h=6):
        return scrolledtext.ScrolledText(p, height=h, width=64,
                                          font=('Consolas', 10),
                                          bg='white', relief=tk.SOLID, bd=1)

    def _chk(self, p, t, v, d):
        f = tk.Frame(p, bg=self.COLORS['panel']); f.pack(fill=tk.X, pady=3)
        tk.Checkbutton(f, text=t, variable=v, font=('Segoe UI', 11, 'bold'),
                       bg=self.COLORS['panel'],
                       activebackground=self.COLORS['panel'],
                       cursor='hand2').pack(anchor='w')
        tk.Label(f, text=d, font=('Segoe UI', 9), bg=self.COLORS['panel'],
                 fg=self.COLORS['muted']).pack(anchor='w', padx=(26, 0))

    def _btn(self, p, t, c, color='#4b5563', **kw):
        b = tk.Button(p, text=t, command=c, font=('Segoe UI', 10),
                      bg=color, fg='white', relief=tk.FLAT,
                      cursor='hand2', padx=8, pady=3, **kw)
        b.bind('<Enter>', lambda e: b.config(bg=self._dk(color)))
        b.bind('<Leave>', lambda e: b.config(bg=color))
        return b

    def _fbtn(self, p, t, c, color, large=False):
        b = tk.Button(p, text=t, command=c,
                      font=('Segoe UI', 12 if large else 10, 'bold'),
                      bg=color, fg='white', relief=tk.FLAT, cursor='hand2',
                      padx=18 if large else 10, pady=9 if large else 6)
        b.bind('<Enter>', lambda e: b.config(bg=self._dk(color)))
        b.bind('<Leave>', lambda e: b.config(bg=color))
        return b

    @staticmethod
    def _dk(c, a=25):
        c = c.lstrip('#')
        r, g, b = (int(c[i:i+2], 16) for i in (0, 2, 4))
        return f'#{max(0,r-a):02x}{max(0,g-a):02x}{max(0,b-a):02x}'

    def _log(self, t):
        self.output_text.insert(tk.END, t)
        self.output_text.see(tk.END)
        self.output_text.update()

    # =========================================================================
    # TEXTBOX UTILITIES
    # =========================================================================
    def _lines(self, w):
        return [_strip_md(l.strip()) for l in w.get('1.0', tk.END).split('\n') if l.strip()]

    def _append(self, w, v):
        if not v or not v.strip(): return
        v = _strip_md(v.strip())
        c = w.get('1.0', tk.END).strip()
        w.insert(tk.END, ('\n' if c else '') + v)
        w.see(tk.END)

    def _remove_line(self, w):
        try:
            idx = w.index("insert linestart")
            if w.get(idx, f"{idx} lineend").strip():
                w.delete(idx, f"{idx} lineend+1c")
        except Exception:
            pass

    def _dialog_append(self, widget, label):
        win = tk.Toplevel(self.root); win.title("Add"); win.resizable(False, False)
        win.transient(self.root); win.grab_set()
        tk.Label(win, text=label, font=('Segoe UI', 11)).pack(padx=14, pady=(10, 4))
        e = tk.Entry(win, width=48, font=('Segoe UI', 11))
        e.pack(padx=14, pady=4); e.focus_set()
        result = {}
        def ok(): result['v'] = e.get(); win.destroy()
        tk.Button(win, text="OK", command=ok, bg=self.COLORS['primary'], fg='white',
                  font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
                  padx=18, pady=5).pack(pady=10)
        win.bind('<Return>', lambda _: ok()); win.wait_window()
        self._append(widget, result.get('v', ''))

    def _add_binary_folder(self):
        folder = filedialog.askdirectory(title="Select Binary Folder")
        if folder:
            dst = os.path.basename(folder.rstrip('/\\'))
            self._append(self.binaries_text, f"{folder};{dst}")

    def _add_binary_file(self):
        for f in filedialog.askopenfilenames(title="Select DLL / .pyd",
                                              filetypes=[("DLL/PYD","*.dll *.pyd"),("All","*.*")]):
            self._append(self.binaries_text,
                         f"{f};{os.path.basename(os.path.dirname(f)) or '.'}")

    def _add_data_files(self):
        for f in filedialog.askopenfilenames(title="Select Data Files"):
            self._append(self.datas_text,
                         f"{f};{os.path.basename(os.path.dirname(f)) or '.'}")

    def _add_data_folder(self):
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            dst = os.path.basename(folder.rstrip('/\\'))
            self._append(self.datas_text, f"{folder};{dst}")

    def browse_script(self):
        f = filedialog.askopenfilename(title="Select Script",
                                       filetypes=[("Python","*.py"),("All","*.*")])
        if f:
            self.script_entry.delete(0, tk.END)
            self.script_entry.insert(0, _strip_md(f))
            self._log(f"✅ Script: {f}\n")

    def browse_icon(self):
        f = filedialog.askopenfilename(title="Select Icon",
                                       filetypes=[("Icon","*.ico"),("All","*.*")])
        if f:
            self.icon_entry.delete(0, tk.END)
            self.icon_entry.insert(0, _strip_md(f))

    # =========================================================================
    # CONFIG  GET / APPLY
    # =========================================================================
    def get_config(self):
        return {
            "script_path":     _strip_md(self.script_entry.get().strip()),
            "name":            _strip_md(self.name_entry.get().strip()) or "MyApp",
            "onefile":         self.onefile_var.get(),
            "console":         self.console_var.get(),
            "icon":            _strip_md(self.icon_entry.get().strip()),
            "collect_all":     self._lines(self.collect_all_text),
            "collect_data":    self._lines(self.collect_data_text),
            "hidden_imports":  self._lines(self.hidden_text),
            "binaries":        [[_strip_md(p.split(';',1)[0].strip()),
                                  _strip_md(p.split(';',1)[1].strip())]
                                 for p in self._lines(self.binaries_text) if ';' in p],
            "datas":           self._lines(self.datas_text),
            "exclude_modules": self._lines(self.exclude_text),
            "upx":             self.upx_var.get(),
            "clean":           self.clean_var.get(),
            "debug":           self.debug_var.get(),
            "noconfirm":       self.noconfirm_var.get(),
        }

    def _apply(self, cfg):
        def e(w, v): w.delete(0, tk.END); w.insert(0, _strip_md(v or ''))
        def t(w, v):
            w.delete('1.0', tk.END)
            if isinstance(v, list):
                lines = []
                for item in v:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        lines.append(f"{_strip_md(str(item[0]))};{_strip_md(str(item[1]))}")
                    else:
                        lines.append(_strip_md(str(item)))
                w.insert('1.0', '\n'.join(lines))
            elif v:
                w.insert('1.0', _strip_md(str(v)))

        e(self.script_entry, cfg.get("script_path", ""))
        e(self.name_entry,   cfg.get("name", "MyApp"))
        e(self.icon_entry,   cfg.get("icon", ""))
        self.onefile_var.set(cfg.get("onefile", True))
        self.console_var.set(cfg.get("console", True))
        self.upx_var.set(cfg.get("upx", False))
        self.clean_var.set(cfg.get("clean", True))
        self.debug_var.set(cfg.get("debug", False))
        self.noconfirm_var.set(cfg.get("noconfirm", True))
        t(self.collect_all_text,  cfg.get("collect_all", []))
        t(self.collect_data_text, cfg.get("collect_data", []))
        t(self.hidden_text,       cfg.get("hidden_imports", []))
        t(self.binaries_text,     cfg.get("binaries", []))
        t(self.datas_text,        cfg.get("datas", []))
        t(self.exclude_text,      cfg.get("exclude_modules", []))

    # =========================================================================
    # JSON  SAVE / LOAD
    # =========================================================================
    def _save_json(self):
        fn = filedialog.asksaveasfilename(title="Save JSON Config",
                                          defaultextension=".json",
                                          filetypes=[("JSON","*.json")],
                                          initialfile="build_config.json")
        if not fn: return
        try:
            with open(fn, 'w', encoding='utf-8') as f:
                json.dump(self.get_config(), f, indent=4)
            self._log(f"✅ JSON saved: {fn}\n")
            messagebox.showinfo("Saved", fn)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    def _load_json(self):
        fn = filedialog.askopenfilename(title="Load JSON Config",
                                         filetypes=[("JSON","*.json"),("All","*.*")])
        if not fn: return
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self._apply(cfg)
            self._loaded_spec_path = None
            self._log(f"✅ JSON loaded: {fn}\n")
            messagebox.showinfo("Loaded", fn)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    def _auto_load(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._apply(json.load(f))
                self._log("✅ Auto-loaded last session\n\n")
            except Exception:
                pass

    # =========================================================================
    # SPEC  LOAD
    # =========================================================================
    def _load_spec(self):
        fn = filedialog.askopenfilename(title="Load .spec File",
                                         filetypes=[("Spec","*.spec"),("All","*.*")])
        if not fn: return
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                spec = f.read()
            spec = _strip_md(spec)
            cfg = self._parse_spec(spec, spec_file_path=fn)
            self._apply(cfg)
            self._loaded_spec_path = fn
            self._log(f"✅ Spec loaded: {fn}\n")
            self._log(f"   Script : {cfg.get('script_path')}\n")
            messagebox.showinfo("Spec Loaded",
                                f"Parsed: {fn}\n\n"
                                f"Script       : {cfg.get('script_path')}\n"
                                f"collect_all  : {cfg.get('collect_all')}\n"
                                f"collect_data : {cfg.get('collect_data')}\n"
                                f"Hidden imports: {len(cfg.get('hidden_imports', []))}\n\n"
                                f"Tip: click '💾 Save Spec' to save edits back to spec,\n"
                                f"or '🚀 Build from Spec' to build without regenerating.")
        except Exception as ex:
            messagebox.showerror("Parse Error", str(ex))

    # =========================================================================
    # SPEC  SAVE  (new — writes UI → .spec file of your choice)
    # =========================================================================
    def _save_spec(self):
        cfg = self.get_config()
        if not cfg["script_path"]:
            messagebox.showerror("Error", "Select a script first."); return

        # Default filename: beside the script
        default_dir  = os.path.dirname(cfg["script_path"]) if cfg["script_path"] else "."
        default_name = f"{cfg['name']}.spec"

        fn = filedialog.asksaveasfilename(
            title="Save .spec File",
            defaultextension=".spec",
            filetypes=[("Spec","*.spec"),("All","*.*")],
            initialdir=default_dir,
            initialfile=default_name
        )
        if not fn: return
        try:
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(self._build_spec_text(cfg))
            self._loaded_spec_path = fn
            self._log(f"✅ Spec saved: {fn}\n")
            messagebox.showinfo("Spec Saved",
                                f"Saved: {fn}\n\n"
                                f"You can now click '🚀 Build from Spec' to build directly\n"
                                f"from this file without going through the UI again.")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    # =========================================================================
    # SPEC  BUILD DIRECTLY  (new — picks any .spec and runs pyinstaller on it)
    # =========================================================================
    def _build_from_spec(self):
        """Build directly from an existing .spec file — no UI regeneration."""
        # Default to last loaded/saved spec if available
        initial = self._loaded_spec_path or ""
        initial_dir  = os.path.dirname(initial) if initial else "."
        initial_file = os.path.basename(initial) if initial else ""

        fn = filedialog.askopenfilename(
            title="Select .spec File to Build",
            filetypes=[("Spec","*.spec"),("All","*.*")],
            initialdir=initial_dir,
            initialfile=initial_file
        )
        if not fn: return

        # Quick preflight — check spec exists
        if not os.path.exists(fn):
            messagebox.showerror("Not Found", f"Spec file not found:\n{fn}"); return

        # Scrub markdown links from spec before building
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                raw = f.read()
            clean = _strip_md(raw)
            if clean != raw:
                with open(fn, 'w', encoding='utf-8') as f:
                    f.write(clean)
                self._log(f"⚠️  Stripped markdown links from spec before building.\n")
        except Exception:
            pass

        cfg = self.get_config()
        clean_flag    = cfg.get("clean", True)
        noconfirm_flag= cfg.get("noconfirm", True)

        cwd = os.path.dirname(os.path.abspath(fn))

        cmd = ["pyinstaller"]
        if clean_flag:     cmd.append("--clean")
        if noconfirm_flag: cmd.append("--noconfirm")
        cmd.append(fn)

        app_name = os.path.splitext(os.path.basename(fn))[0]

        self._log("\n" + "═"*55 + "\n")
        self._log(f"🚀  Building from spec directly\n")
        self._log(f"    Spec : {fn}\n")
        self._log(f"    CWD  : {cwd}\n")
        self._log(f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log("═"*55 + "\n\n")

        threading.Thread(
            target=self._run_build,
            args=(cmd, app_name, cwd),
            daemon=True
        ).start()

    # =========================================================================
    # SPEC PARSING
    # =========================================================================
    def _parse_spec(self, spec, spec_file_path=None):
        cfg = dict(self.DEFAULT_CONFIG)

        m = re.search(r"Analysis\s*\(\s*\[r?['\"](.+?)['\"]", spec)
        if m:
            script = _strip_md(m.group(1))
            if not os.path.isabs(script) and spec_file_path:
                script = os.path.join(os.path.dirname(spec_file_path), script)
            cfg["script_path"] = os.path.normpath(script)

        m = re.search(r"name\s*=\s*['\"](.+?)['\"]", spec)
        if m: cfg["name"] = _strip_md(m.group(1))

        m = re.search(r"console\s*=\s*(True|False)", spec)
        if m: cfg["console"] = m.group(1) == 'True'

        cfg["onefile"] = 'runtime_tmpdir' in spec

        m = re.search(r"debug\s*=\s*(True|False)", spec)
        if m: cfg["debug"] = m.group(1) == 'True'

        m = re.search(r"\bupx\s*=\s*(True|False)", spec)
        if m: cfg["upx"] = m.group(1) == 'True'

        m = re.search(r"icon\s*=\s*r?['\"](.+?)['\"]", spec)
        if m: cfg["icon"] = _strip_md(m.group(1))

        cfg["collect_all"]  = [_strip_md(x) for x in re.findall(r"collect_all\(['\"](.+?)['\"]\)", spec)]
        cfg["collect_data"] = [_strip_md(x) for x in re.findall(r"collect_data_files\(['\"](.+?)['\"]\)", spec)]

        m = re.search(r"hiddenimports\s*=.*?\[(.+?)\]", spec, re.DOTALL)
        if m:
            cfg["hidden_imports"] = [_strip_md(x)
                                     for x in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))]

        m = re.search(r"excludes\s*=\s*\[(.+?)\]", spec, re.DOTALL)
        if m:
            cfg["exclude_modules"] = [_strip_md(x)
                                      for x in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))]

        datas = re.findall(r"\(r?['\"](.+?)['\"]\s*,\s*r?['\"](.+?)['\"]\)", spec)
        cfg["datas"] = [f"{_strip_md(s)};{_strip_md(d)}" for s, d in datas]

        return cfg

    # =========================================================================
    # SPEC TEXT GENERATION
    # =========================================================================
    def _build_spec_text(self, cfg):
        script    = _strip_md(cfg["script_path"].replace("\\", "/"))
        app_name  = _strip_md(cfg["name"])
        icon_path = _strip_md(cfg.get("icon", "").replace("\\", "/"))
        icon_val  = f"r'{icon_path}'" if icon_path and os.path.exists(icon_path) else "None"
        dbg = "True" if cfg.get("debug")   else "False"
        upx = "True" if cfg.get("upx")     else "False"
        con = "True" if cfg.get("console") else "False"
        NL  = "\n"

        collect_lines, safe_names = [], []
        for pkg in [_strip_md(p) for p in cfg.get("collect_all", [])]:
            safe = pkg.replace("-","_").replace(".","_")
            collect_lines.append(f"{safe}_d, {safe}_b, {safe}_h = collect_all('{pkg}')")
            safe_names.append(safe)
        for pkg in [_strip_md(p) for p in cfg.get("collect_data", [])]:
            safe = pkg.replace("-","_").replace(".","_")
            collect_lines.append(
                f"{safe}_d = collect_data_files('{pkg}')\n"
                f"{safe}_b = []\n"
                f"{safe}_h = collect_submodules('{pkg}')"
            )
            safe_names.append(safe)

        md = " + ".join(f"{n}_d" for n in safe_names) if safe_names else "[]"
        mb = " + ".join(f"{n}_b" for n in safe_names) if safe_names else "[]"
        mh = " + ".join(f"{n}_h" for n in safe_names) if safe_names else "[]"

        dl = []
        for entry in cfg.get("datas", []):
            entry = _strip_md(entry)
            if ";" in entry:
                s, d = entry.split(";", 1)
                s, d = s.strip().replace("\\","/"), d.strip()
                if os.path.exists(s):
                    dl.append(f"    (r'{s}', r'{d}'),")

        bl = []
        for entry in cfg.get("binaries", []):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                s = _strip_md(str(entry[0])).replace("\\","/")
                d = _strip_md(str(entry[1]))
                if os.path.exists(s):
                    bl.append(f"    (r'{s}', r'{d}'),")

        hl = [f"    '{_strip_md(h)}'," for h in cfg.get("hidden_imports", [])]
        xl = [f"    '{_strip_md(x)}'," for x in cfg.get("exclude_modules", [])]

        if cfg.get("onefile", True):
            exe_block = (
                f"exe = EXE(\n"
                f"    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],\n"
                f"    name='{app_name}', debug={dbg},\n"
                f"    bootloader_ignore_signals=False, strip=False,\n"
                f"    upx={upx}, upx_exclude=[], runtime_tmpdir=None,\n"
                f"    console={con}, icon={icon_val},\n"
                f")\n"
            )
        else:
            exe_block = (
                f"exe = EXE(\n"
                f"    pyz, a.scripts, [], exclude_binaries=True,\n"
                f"    name='{app_name}', debug={dbg}, strip=False,\n"
                f"    upx={upx}, console={con}, icon={icon_val},\n"
                f")\n"
                f"coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas,\n"
                f"               strip=False, upx={upx}, name='{app_name}')\n"
            )

        return (
            f"# -*- mode: python ; coding: utf-8 -*-\n"
            f"# Generated by PyInstaller Builder Pro  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules\n\n"
            f"# collect_all / collect_data  —  bundles C-extension DLLs\n"
            f"{NL.join(collect_lines) if collect_lines else '# (none)'}\n\n"
            f"_collected_datas    = {md}\n"
            f"_collected_binaries = {mb}\n"
            f"_collected_hidden   = {mh}\n\n"
            f"block_cipher = None\n\n"
            f"a = Analysis(\n"
            f"    [r'{script}'],\n"
            f"    pathex=[],\n"
            f"    binaries=_collected_binaries + [\n"
            f"{NL.join(bl)}\n"
            f"    ],\n"
            f"    datas=_collected_datas + [\n"
            f"{NL.join(dl)}\n"
            f"    ],\n"
            f"    hiddenimports=_collected_hidden + [\n"
            f"{NL.join(hl)}\n"
            f"    ],\n"
            f"    hookspath=[], hooksconfig={{}}, runtime_hooks=[],\n"
            f"    excludes=[\n"
            f"{NL.join(xl)}\n"
            f"    ],\n"
            f"    cipher=block_cipher, noarchive=False,\n"
            f")\n\n"
            f"pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)\n\n"
            f"{exe_block}"
        )

    # =========================================================================
    # PREVIEW SPEC
    # =========================================================================
    def _preview_spec(self):
        cfg = self.get_config()
        if not cfg["script_path"]:
            messagebox.showerror("Error", "Select a script first."); return
        txt = self._build_spec_text(cfg)
        win = tk.Toplevel(self.root)
        win.title(f"{cfg['name']}.spec — Preview")
        win.geometry("900x700")
        st = scrolledtext.ScrolledText(win, font=('Consolas', 11), wrap=tk.NONE)
        st.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        st.insert('1.0', txt)

        def save_it():
            default_dir = os.path.dirname(cfg["script_path"]) if cfg["script_path"] else "."
            fn = filedialog.asksaveasfilename(
                title="Save Spec", defaultextension=".spec",
                filetypes=[("Spec","*.spec")],
                initialdir=default_dir,
                initialfile=f"{cfg['name']}.spec"
            )
            if fn:
                with open(fn, 'w', encoding='utf-8') as f: f.write(txt)
                self._loaded_spec_path = fn
                messagebox.showinfo("Saved", fn)

        tk.Button(win, text="💾 Save Spec", command=save_it,
                  bg=self.COLORS['teal'], fg='white',
                  font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
                  padx=20, pady=6).pack(pady=6)

    # =========================================================================
    # BUILD FROM UI  (generates spec → builds)
    # =========================================================================
    def _build(self):
        cfg = self.get_config()
        script = cfg["script_path"]
        if not script:
            messagebox.showerror("Error", "Select a Python script first!"); return
        if not os.path.isabs(script):
            script = os.path.abspath(script); cfg["script_path"] = script
        if not os.path.exists(script):
            messagebox.showerror("Error", f"Script not found:\n{script}"); return

        local_np = os.path.join(os.path.dirname(script), "numpy")
        if os.path.exists(local_np):
            messagebox.showerror("Numpy Conflict",
                f"Local numpy/ folder found:\n{local_np}\n\n"
                "Rename or delete it — this causes the DLL import error.")
            return

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=4)
        except Exception:
            pass

        script_dir = os.path.dirname(script)
        spec_path  = os.path.join(script_dir, f"{cfg['name']}.spec")
        with open(spec_path, 'w', encoding='utf-8') as f:
            f.write(self._build_spec_text(cfg))
        self._loaded_spec_path = spec_path

        cmd = ["pyinstaller"]
        if cfg.get("clean"):     cmd.append("--clean")
        if cfg.get("noconfirm"): cmd.append("--noconfirm")
        cmd.append(spec_path)

        self._log("\n" + "═"*55 + "\n")
        self._log(f"🔨  Building : {cfg['name']}\n")
        self._log(f"    Script   : {script}\n")
        self._log(f"    Spec     : {spec_path}\n")
        self._log(f"    CWD      : {script_dir}\n")
        self._log(f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log("═"*55 + "\n\n")

        threading.Thread(
            target=self._run_build,
            args=(cmd, cfg['name'], script_dir),
            daemon=True
        ).start()

    # =========================================================================
    # SHARED BUILD RUNNER
    # =========================================================================
    def _run_build(self, cmd, app_name, cwd=None):
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=cwd
            )
            for line in proc.stdout:
                self._log(line)
            proc.wait()
            self._log("\n" + "═"*55 + "\n")
            if proc.returncode == 0:
                self._log(f"✅  BUILD SUCCESSFUL — dist/{app_name}\n")
                messagebox.showinfo("Done", f"Build complete!\n\ndist/{app_name}")
            else:
                self._log(f"❌  BUILD FAILED (exit {proc.returncode})\n")
                messagebox.showerror("Failed", "Build failed — check the output panel.")
            self._log("═"*55 + "\n")
        except Exception as ex:
            self._log(f"\n❌ {ex}\n")
            messagebox.showerror("Error", str(ex))

    # =========================================================================
    # FOLDER UTILITIES
    # =========================================================================
    def _open_folder(self, path):
        if not os.path.exists(path):
            messagebox.showwarning("Not Found", f"{path} does not exist yet."); return
        if sys.platform == 'win32':    os.startfile(path)
        elif sys.platform == 'darwin': subprocess.run(['open', path])
        else:                          subprocess.run(['xdg-open', path])

    def _open_dist(self):  self._open_folder(os.path.abspath("dist"))
    def _open_build(self): self._open_folder(os.path.abspath("build"))

    def _clean_folders(self):
        if not messagebox.askyesno("Confirm",
                                   "Delete build/ and dist/ folders?", icon='warning'):
            return
        self._log("\n🧹 Cleaning...\n")
        cfg = self.get_config()
        for p in ['build', 'dist', f"{cfg['name']}.spec"]:
            ap = os.path.abspath(p)
            if os.path.isdir(ap):    shutil.rmtree(ap); self._log(f"  ✅ {ap}\n")
            elif os.path.isfile(ap): os.remove(ap);     self._log(f"  ✅ {ap}\n")
        messagebox.showinfo("Done", "Cleaned.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    ModernPyInstallerGUI(root)
    root.mainloop()
