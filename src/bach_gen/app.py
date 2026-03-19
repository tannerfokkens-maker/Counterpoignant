"""Tk GUI for live Bach generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import queue
import threading
from pathlib import Path
import traceback

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - platform dependent
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

import mido

from bach_gen.cli import _maybe_cast_generation_model_for_mps
from bach_gen.data.tokenizer import load_tokenizer
from bach_gen.generation.live import (
    autodetect_channel_base,
    autodetect_midi_output_port,
    list_midi_output_ports,
    run_live_generation,
)
from bach_gen.model.trainer import Trainer
from bach_gen.utils.constants import FORM_DEFAULTS, VALID_FORMS


APP_TITLE = "Bach Gen Studio"
SETTINGS_PATH = Path.home() / ".bach-gen-studio.json"
DEFAULT_SUBJECT_DURATIONS = ["e", "q", "q.", "h", "h.", "w", "s"]
DEFAULT_SUBJECT_PITCHES = [
    "Rest",
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb",
    "G", "G#", "Ab", "A", "A#", "Bb", "B",
]
PALETTE = {
    "paper": "#111417",
    "paper_deep": "#171b20",
    "panel": "#1b2025",
    "panel_alt": "#252c33",
    "ink": "#efe6d7",
    "muted": "#b2a798",
    "accent": "#cc7757",
    "accent_soft": "#6d4338",
    "forest": "#2f7a66",
    "forest_soft": "#223932",
    "gold": "#c59638",
    "gold_soft": "#3d3322",
    "danger": "#b65b4e",
    "border": "#39424b",
    "log_bg": "#0d0f12",
    "log_fg": "#efe3d1",
}


@dataclass
class SubjectBuilderEvent:
    note: str = "Rest"
    octave: int = 4
    duration: str = "q"


def build_subject_string(events: list[SubjectBuilderEvent]) -> str:
    """Convert GUI builder rows into a subject string."""
    parts: list[str] = []
    for event in events:
        duration = event.duration.strip()
        if not duration:
            continue
        if event.note == "Rest":
            parts.append(f"r:{duration}")
        else:
            parts.append(f"{event.note}{int(event.octave)}:{duration}")
    return " ".join(parts)


def _default_workspace_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "data" / "tokenizer.json").exists():
        return cwd
    return cwd


def _default_model_path(workspace_root: Path) -> Path | None:
    candidates = [
        workspace_root / "models" / "best.pt",
        workspace_root / "models" / "latest.pt",
        workspace_root / "models" / "final.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class SubjectBuilderFrame(ttk.LabelFrame):
    """Small event-based subject editor."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="Subject Builder", style="Card.TLabelframe")
        self._rows: list[tuple[tk.StringVar, tk.IntVar, tk.StringVar]] = []

        header = ttk.Frame(self, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        ttk.Label(header, text="Pitch", style="FieldLabel.TLabel").grid(row=0, column=0, padx=(0, 8))
        ttk.Label(header, text="Oct", style="FieldLabel.TLabel").grid(row=0, column=1, padx=(0, 8))
        ttk.Label(header, text="Dur", style="FieldLabel.TLabel").grid(row=0, column=2)

        self.rows_frame = ttk.Frame(self, style="Card.TFrame")
        self.rows_frame.grid(row=1, column=0, sticky="ew", padx=8)

        buttons = ttk.Frame(self, style="Card.TFrame")
        buttons.grid(row=2, column=0, sticky="w", padx=8, pady=(4, 6))
        ttk.Button(buttons, text="Add Event", style="Secondary.TButton", command=self.add_row).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(buttons, text="Remove Event", style="Secondary.TButton", command=self.remove_row).grid(row=0, column=1)

        for seed in [
            SubjectBuilderEvent("D", 4, "q"),
            SubjectBuilderEvent("A", 4, "e"),
            SubjectBuilderEvent("Rest", 4, "e"),
            SubjectBuilderEvent("Bb", 4, "q"),
        ]:
            self.add_row(seed)

    def add_row(self, seed: SubjectBuilderEvent | None = None) -> None:
        seed = seed or SubjectBuilderEvent()
        row_idx = len(self._rows)
        note_var = tk.StringVar(value=seed.note)
        octave_var = tk.IntVar(value=seed.octave)
        duration_var = tk.StringVar(value=seed.duration)

        note_box = ttk.Combobox(
            self.rows_frame,
            textvariable=note_var,
            values=DEFAULT_SUBJECT_PITCHES,
            width=8,
            state="readonly",
        )
        octave_box = ttk.Spinbox(self.rows_frame, from_=0, to=8, textvariable=octave_var, width=4)
        duration_box = ttk.Combobox(
            self.rows_frame,
            textvariable=duration_var,
            values=DEFAULT_SUBJECT_DURATIONS,
            width=6,
            state="readonly",
        )
        note_box.grid(row=row_idx, column=0, padx=(0, 8), pady=2, sticky="w")
        octave_box.grid(row=row_idx, column=1, padx=(0, 8), pady=2, sticky="w")
        duration_box.grid(row=row_idx, column=2, pady=2, sticky="w")
        self._rows.append((note_var, octave_var, duration_var))

    def remove_row(self) -> None:
        if not self._rows:
            return
        row_idx = len(self._rows) - 1
        for widget in self.rows_frame.grid_slaves(row=row_idx):
            widget.destroy()
        self._rows.pop()

    def get_subject_string(self) -> str:
        return build_subject_string(
            [
                SubjectBuilderEvent(
                    note=note_var.get(),
                    octave=octave_var.get(),
                    duration=duration_var.get(),
                )
                for note_var, octave_var, duration_var in self._rows
            ]
        )

    def set_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        spin_state = "normal" if enabled else "disabled"
        for row_idx, (note_var, _octave_var, _duration_var) in enumerate(self._rows):
            widgets = {widget.grid_info()["column"]: widget for widget in self.rows_frame.grid_slaves(row=row_idx)}
            if 0 in widgets:
                widgets[0].configure(state=state)
            if 1 in widgets:
                widgets[1].configure(state=spin_state)
            if 2 in widgets:
                widgets[2].configure(state=state)


class BachGenStudio:
    """Desktop GUI for live generation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1060x860")
        self.root.minsize(900, 760)
        self.root.configure(bg=PALETTE["paper"])

        self.event_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.session_thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.cached_model_path: str | None = None
        self.cached_model = None
        self.cached_tokenizer = None
        self.cached_tokenizer_path: str | None = None

        self.workspace_var = tk.StringVar(value=str(_default_workspace_root()))
        default_model = _default_model_path(Path(self.workspace_var.get()))
        self.model_var = tk.StringVar(value=str(default_model) if default_model else "")
        self.port_var = tk.StringVar(value="")
        self.key_var = tk.StringVar(value="D minor")
        self.mode_var = tk.StringVar(value="fugue")
        self.voices_var = tk.IntVar(value=3)
        self.style_var = tk.StringVar(value="bach")
        self.length_var = tk.StringVar(value="")
        self.meter_var = tk.StringVar(value="alla_breve")
        self.texture_var = tk.StringVar(value="polyphonic")
        self.imitation_var = tk.StringVar(value="high")
        self.harmonic_rhythm_var = tk.StringVar(value="moderate")
        self.tension_var = tk.StringVar(value="moderate")
        self.chromaticism_var = tk.StringVar(value="high")
        self.temperature_var = tk.DoubleVar(value=0.94)
        self.min_p_var = tk.DoubleVar(value=0.03)
        self.tempo_var = tk.DoubleVar(value=110.0)
        self.prebuffer_var = tk.IntVar(value=4)
        self.low_water_var = tk.IntVar(value=2)
        self.single_channel_var = tk.BooleanVar(value=True)
        self.auto_channel_var = tk.BooleanVar(value=True)
        self.channel_base_var = tk.IntVar(value=0)
        self.velocity_var = tk.IntVar(value=80)
        self.cadence_density_var = tk.StringVar(value="")
        self.subject_remind_var = tk.StringVar(value="")
        self.mps_bf16_var = tk.BooleanVar(value=False)
        self.subject_mode_var = tk.StringVar(value="none")
        self.subject_text_var = tk.StringVar(value="")
        self.subject_midi_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")

        self._configure_styles()
        self._build_ui()
        self._load_settings()
        self.refresh_ports()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background=PALETTE["paper"])
        style.configure("Card.TFrame", background=PALETTE["panel"])
        style.configure(
            "Card.TLabelframe",
            background=PALETTE["panel"],
            bordercolor=PALETTE["border"],
            borderwidth=1,
            relief="solid",
            padding=10,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=PALETTE["panel"],
            foreground=PALETTE["ink"],
            font=("SF Pro Display", 13, "bold"),
        )
        style.configure(
            "Title.TLabel",
            background=PALETTE["paper"],
            foreground=PALETTE["ink"],
            font=("Iowan Old Style", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=PALETTE["paper"],
            foreground=PALETTE["muted"],
            font=("SF Pro Text", 11),
        )
        style.configure(
            "FieldLabel.TLabel",
            background=PALETTE["panel"],
            foreground=PALETTE["muted"],
            font=("SF Pro Text", 10, "bold"),
        )
        style.configure(
            "Note.TLabel",
            background=PALETTE["panel"],
            foreground=PALETTE["muted"],
            font=("SF Pro Text", 10),
        )
        style.configure(
            "Accent.TButton",
            background=PALETTE["forest"],
            foreground=PALETTE["panel"],
            borderwidth=0,
            focusthickness=0,
            focuscolor=PALETTE["forest"],
            padding=(14, 10),
            font=("SF Pro Text", 11, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#356f60"), ("disabled", PALETTE["forest_soft"])],
            foreground=[("disabled", PALETTE["panel"])],
        )
        style.configure(
            "Secondary.TButton",
            background=PALETTE["paper_deep"],
            foreground=PALETTE["ink"],
            bordercolor=PALETTE["border"],
            padding=(12, 9),
            font=("SF Pro Text", 10, "bold"),
        )
        style.map("Secondary.TButton", background=[("active", PALETTE["gold_soft"])])
        style.configure(
            "Danger.TButton",
            background=PALETTE["danger"],
            foreground=PALETTE["panel"],
            borderwidth=0,
            padding=(12, 9),
            font=("SF Pro Text", 10, "bold"),
        )
        style.map("Danger.TButton", background=[("active", "#9b4b40"), ("disabled", "#c7a39c")])
        style.configure(
            "TEntry",
            fieldbackground=PALETTE["panel_alt"],
            background=PALETTE["panel_alt"],
            foreground=PALETTE["ink"],
            insertcolor=PALETTE["ink"],
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            padding=7,
        )
        style.configure(
            "TCombobox",
            fieldbackground=PALETTE["panel_alt"],
            background=PALETTE["panel_alt"],
            foreground=PALETTE["ink"],
            bordercolor=PALETTE["border"],
            arrowsize=14,
            padding=6,
        )
        style.map("TCombobox", fieldbackground=[("readonly", PALETTE["panel_alt"])])
        style.configure(
            "TSpinbox",
            fieldbackground=PALETTE["panel_alt"],
            background=PALETTE["panel_alt"],
            foreground=PALETTE["ink"],
            arrowsize=13,
            padding=5,
        )
        style.configure(
            "TCheckbutton",
            background=PALETTE["panel"],
            foreground=PALETTE["ink"],
            font=("SF Pro Text", 10),
        )
        style.configure(
            "TRadiobutton",
            background=PALETTE["panel"],
            foreground=PALETTE["ink"],
            font=("SF Pro Text", 10, "bold"),
        )

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(
            outer,
            bg=PALETTE["paper"],
            highlightthickness=0,
            relief="flat",
            bd=0,
        )
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = ttk.Frame(canvas, padding=16, style="App.TFrame")
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )
        canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.scroll_canvas = canvas

        header = tk.Frame(body, bg=PALETTE["paper"], highlightthickness=0)
        header.pack(fill="x", pady=(0, 12))

        title_col = tk.Frame(header, bg=PALETTE["paper"])
        title_col.pack(side="left", fill="x", expand=True)
        ttk.Label(title_col, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_col,
            text="Live endless fugue playback, subject sketching, and hardware synth streaming.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        header_right = tk.Frame(header, bg=PALETTE["paper"])
        header_right.pack(side="right", anchor="ne")
        self.status_chip = tk.Label(
            header_right,
            text="Idle",
            bg=PALETTE["gold_soft"],
            fg=PALETTE["ink"],
            padx=12,
            pady=7,
            font=("SF Pro Text", 10, "bold"),
            relief="flat",
        )
        self.status_chip.pack(anchor="e")
        tk.Label(
            header_right,
            text="Pinned prompt memory • Live MIDI • Subject builder",
            bg=PALETTE["paper"],
            fg=PALETTE["muted"],
            font=("SF Pro Text", 10),
        ).pack(anchor="e", pady=(8, 0))

        quickstrip = tk.Frame(
            body,
            bg=PALETTE["panel_alt"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            bd=0,
            padx=14,
            pady=12,
        )
        quickstrip.pack(fill="x", pady=(0, 12))
        for idx, text in enumerate(
            [
                "1. Pick model and MIDI port",
                "2. Sketch a subject or load one",
                "3. Start the stream and shape the synth",
            ]
        ):
            pill = tk.Label(
                quickstrip,
                text=text,
                bg=PALETTE["paper"],
                fg=PALETTE["ink"],
                padx=10,
                pady=6,
                font=("SF Pro Text", 10, "bold"),
            )
            pill.grid(row=0, column=idx, sticky="w", padx=(0 if idx == 0 else 10, 0))

        top = ttk.Frame(body, style="App.TFrame")
        top.pack(fill="x", expand=False)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self._build_session_frame(top).grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._build_generation_frame(top).grid(row=0, column=1, sticky="nsew")

        middle = ttk.Frame(body, style="App.TFrame")
        middle.pack(fill="x", pady=(10, 0))
        middle.columnconfigure(0, weight=1)
        middle.columnconfigure(1, weight=1)
        self._build_subject_frame(middle).grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._build_transport_frame(middle).grid(row=0, column=1, sticky="nsew")

        log_frame = ttk.LabelFrame(body, text="Status / Log", style="Card.TLabelframe")
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.log_text = tk.Text(
            log_frame,
            height=14,
            wrap="word",
            bg=PALETTE["log_bg"],
            fg=PALETTE["log_fg"],
            insertbackground=PALETTE["log_fg"],
            relief="flat",
            highlightthickness=0,
            padx=12,
            pady=12,
            font=("SF Mono", 11),
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_text.configure(state="disabled")

        status_bar = ttk.Frame(body, style="App.TFrame")
        status_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(status_bar, textvariable=self.status_var, style="Subtitle.TLabel").pack(anchor="w")
        self._set_status_indicator("Idle")

    def _on_mousewheel(self, event: tk.Event) -> None:
        if not hasattr(self, "scroll_canvas"):
            return
        widget = getattr(event, "widget", None)
        if widget is None or not widget.winfo_exists():
            return
        toplevel = widget.winfo_toplevel()
        if toplevel is not self.root:
            return
        if event.delta == 0:
            return
        step = -1 if event.delta > 0 else 1
        self.scroll_canvas.yview_scroll(step, "units")

    def _build_session_frame(self, master: tk.Misc) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(master, text="Session", style="Card.TLabelframe")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Workspace", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(frame, textvariable=self.workspace_var).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(frame, text="Browse", style="Secondary.TButton", command=self._browse_workspace).grid(row=0, column=2, padx=8, pady=6)

        ttk.Label(frame, text="Model", style="FieldLabel.TLabel").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(frame, textvariable=self.model_var).grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(frame, text="Browse", style="Secondary.TButton", command=self._browse_model).grid(row=1, column=2, padx=8, pady=6)

        ttk.Label(frame, text="MIDI Port", style="FieldLabel.TLabel").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        self.port_box = ttk.Combobox(frame, textvariable=self.port_var, values=[], state="readonly")
        self.port_box.grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(frame, text="Refresh", style="Secondary.TButton", command=self.refresh_ports).grid(row=2, column=2, padx=8, pady=6)

        ttk.Label(frame, text="Key", style="FieldLabel.TLabel").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(frame, textvariable=self.key_var).grid(row=3, column=1, sticky="ew", padx=8, pady=6)

        ttk.Label(frame, text="Mode", style="FieldLabel.TLabel").grid(row=4, column=0, sticky="w", padx=8, pady=6)
        mode_box = ttk.Combobox(frame, textvariable=self.mode_var, values=[v for v in VALID_FORMS if v != "all"], state="readonly")
        mode_box.grid(row=4, column=1, sticky="ew", padx=8, pady=6)
        mode_box.bind("<<ComboboxSelected>>", lambda _e: self._apply_mode_defaults())

        ttk.Label(frame, text="Voices", style="FieldLabel.TLabel").grid(row=5, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(frame, from_=2, to=4, textvariable=self.voices_var, width=6).grid(row=5, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(
            frame,
            text="Choose the workspace and model once; the app keeps them cached for fast restarts.",
            style="Note.TLabel",
            wraplength=360,
            justify="left",
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 2))

        return frame

    def _build_generation_frame(self, master: tk.Misc) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(master, text="Generation", style="Card.TLabelframe")
        for col in range(3):
            frame.columnconfigure(col, weight=1 if col == 1 else 0)

        rows = [
            ("Style", self.style_var, ["bach", "baroque", "renaissance", "classical", "romantic", "modern", "medieval", "other"]),
            ("Length", self.length_var, ["", "short", "medium", "long", "extended"]),
            ("Meter", self.meter_var, ["", "2_4", "3_4", "4_4", "6_8", "3_8", "alla_breve"]),
            ("Texture", self.texture_var, ["", "homophonic", "polyphonic", "mixed"]),
            ("Imitation", self.imitation_var, ["", "none", "low", "high"]),
            ("Harmonic Rhythm", self.harmonic_rhythm_var, ["", "slow", "moderate", "fast"]),
            ("Tension", self.tension_var, ["", "low", "moderate", "high"]),
            ("Chromaticism", self.chromaticism_var, ["", "low", "moderate", "high"]),
            ("Cadence Density", self.cadence_density_var, ["", "low", "medium", "high"]),
        ]
        for row_idx, (label, variable, values) in enumerate(rows):
            ttk.Label(frame, text=label, style="FieldLabel.TLabel").grid(row=row_idx, column=0, sticky="w", padx=8, pady=4)
            ttk.Combobox(frame, textvariable=variable, values=values, state="readonly").grid(
                row=row_idx, column=1, sticky="ew", padx=8, pady=4
            )

        numeric_rows = [
            ("Temperature", self.temperature_var),
            ("Min-p", self.min_p_var),
            ("Tempo BPM", self.tempo_var),
            ("Prebuffer Bars", self.prebuffer_var),
            ("Low-Water Bars", self.low_water_var),
            ("Velocity", self.velocity_var),
        ]
        base_row = len(rows)
        for offset, (label, variable) in enumerate(numeric_rows):
            ttk.Label(frame, text=label, style="FieldLabel.TLabel").grid(row=base_row + offset, column=0, sticky="w", padx=8, pady=4)
            ttk.Entry(frame, textvariable=variable).grid(row=base_row + offset, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(frame, text="Subject Remind Bars", style="FieldLabel.TLabel").grid(row=base_row + len(numeric_rows), column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(frame, textvariable=self.subject_remind_var).grid(row=base_row + len(numeric_rows), column=1, sticky="ew", padx=8, pady=4)

        channel_row = base_row + len(numeric_rows) + 1
        ttk.Checkbutton(frame, text="Single Channel", variable=self.single_channel_var).grid(row=channel_row, column=0, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(frame, text="Auto Channel", variable=self.auto_channel_var, command=self._toggle_channel_entry).grid(row=channel_row, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(frame, text="Channel Base").grid(row=channel_row + 1, column=0, sticky="w", padx=8, pady=4)
        self.channel_entry = ttk.Entry(frame, textvariable=self.channel_base_var)
        self.channel_entry.grid(row=channel_row + 1, column=1, sticky="ew", padx=8, pady=4)

        ttk.Checkbutton(frame, text="MPS bf16", variable=self.mps_bf16_var).grid(row=channel_row + 2, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(
            frame,
            text="The default live decode is tuned for long-form synth playback rather than one-shot batch scoring.",
            style="Note.TLabel",
            wraplength=360,
            justify="left",
        ).grid(row=channel_row + 3, column=0, columnspan=3, sticky="w", padx=8, pady=(6, 2))

        return frame

    def _build_subject_frame(self, master: tk.Misc) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(master, text="Subject", style="Card.TLabelframe")
        mode_row = ttk.Frame(frame, style="Card.TFrame")
        mode_row.pack(fill="x", padx=8, pady=(6, 4))
        for text, value in [("None", "none"), ("String", "string"), ("Builder", "builder"), ("MIDI File", "midi")]:
            ttk.Radiobutton(
                mode_row,
                text=text,
                value=value,
                variable=self.subject_mode_var,
                command=self._update_subject_mode,
            ).pack(side="left", padx=(0, 8))

        quick_row = ttk.Frame(frame, style="Card.TFrame")
        quick_row.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(
            quick_row,
            text="Quick actions",
            style="FieldLabel.TLabel",
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            quick_row,
            text="Load MIDI...",
            style="Secondary.TButton",
            command=self._activate_subject_midi_picker,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            quick_row,
            text="Type Subject",
            style="Secondary.TButton",
            command=lambda: self._set_subject_mode("string"),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            quick_row,
            text="Open Builder",
            style="Secondary.TButton",
            command=lambda: self._set_subject_mode("builder"),
        ).pack(side="left")

        ttk.Label(
            frame,
            text="To upload a subject MIDI, click Load MIDI... or switch to MIDI File mode below.",
            style="Note.TLabel",
            wraplength=420,
            justify="left",
        ).pack(fill="x", padx=8, pady=(0, 8))

        self.subject_string_frame = ttk.Frame(frame, style="Card.TFrame")
        ttk.Label(self.subject_string_frame, text="Subject String", style="FieldLabel.TLabel").pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Entry(self.subject_string_frame, textvariable=self.subject_text_var).pack(fill="x", padx=8, pady=(0, 8))

        self.subject_builder = SubjectBuilderFrame(frame)

        self.subject_midi_frame = ttk.Frame(frame, style="Card.TFrame")
        ttk.Label(self.subject_midi_frame, text="Subject MIDI", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", padx=8, pady=(0, 4))
        self.subject_midi_frame.columnconfigure(0, weight=1)
        ttk.Entry(self.subject_midi_frame, textvariable=self.subject_midi_var).grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(self.subject_midi_frame, text="Browse", style="Secondary.TButton", command=self._browse_subject_midi).grid(row=1, column=1, padx=8, pady=(0, 8))

        self.subject_string_frame.pack(fill="x")
        self.subject_builder.pack(fill="x", padx=8, pady=4)
        self.subject_midi_frame.pack(fill="x")
        self._update_subject_mode()
        return frame

    def _build_transport_frame(self, master: tk.Misc) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(master, text="Transport", style="Card.TLabelframe")
        buttons = ttk.Frame(frame, style="Card.TFrame")
        buttons.pack(fill="x", padx=8, pady=8)
        self.start_button = ttk.Button(buttons, text="Start Live Session", style="Accent.TButton", command=self.start_session)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(buttons, text="Stop", style="Danger.TButton", command=self.stop_session, state="disabled")
        self.stop_button.pack(side="left")

        callout = tk.Frame(
            frame,
            bg=PALETTE["forest_soft"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            bd=0,
            padx=12,
            pady=10,
        )
        callout.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(
            callout,
            text="Default live setup",
            bg=PALETTE["forest_soft"],
            fg=PALETTE["forest"],
            font=("SF Pro Text", 10, "bold"),
        ).pack(anchor="w")
        help_text = (
            "3-voice fugue, alla breve, single channel, pinned prompt memory, and endless playback. "
            "Start there, then shape the synth and subject behavior."
        )
        tk.Label(
            callout,
            text=help_text,
            bg=PALETTE["forest_soft"],
            fg=PALETTE["ink"],
            wraplength=420,
            justify="left",
            font=("SF Pro Text", 10),
        ).pack(anchor="w", pady=(4, 0))
        return frame

    def _browse_workspace(self) -> None:
        selected = filedialog.askdirectory(title="Select workspace root")
        if selected:
            self.workspace_var.set(selected)
            if not self.model_var.get():
                default_model = _default_model_path(Path(selected))
                if default_model:
                    self.model_var.set(str(default_model))

    def _browse_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.model_var.set(selected)

    def _browse_subject_midi(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select subject MIDI",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if selected:
            self.subject_midi_var.set(selected)
            self.subject_mode_var.set("midi")
            self._update_subject_mode()

    def _set_subject_mode(self, mode: str) -> None:
        self.subject_mode_var.set(mode)
        self._update_subject_mode()

    def _activate_subject_midi_picker(self) -> None:
        self._set_subject_mode("midi")
        self._browse_subject_midi()

    def _toggle_channel_entry(self) -> None:
        self.channel_entry.configure(state="disabled" if self.auto_channel_var.get() else "normal")

    def _update_subject_mode(self) -> None:
        mode = self.subject_mode_var.get()
        self.subject_string_frame.pack_forget()
        self.subject_builder.pack_forget()
        self.subject_midi_frame.pack_forget()

        if mode == "string":
            self.subject_string_frame.pack(fill="x")
        elif mode == "builder":
            self.subject_builder.pack(fill="x", padx=8, pady=4)
        elif mode == "midi":
            self.subject_midi_frame.pack(fill="x")
        self.subject_builder.set_enabled(mode == "builder")

    def _apply_mode_defaults(self) -> None:
        if self.mode_var.get() == "fugue":
            self.voices_var.set(3)
            self.meter_var.set("alla_breve")
            self.texture_var.set("polyphonic")
            self.imitation_var.set("high")
            self.harmonic_rhythm_var.set("moderate")
            self.tension_var.set("moderate")
            self.chromaticism_var.set("high")

    def refresh_ports(self) -> None:
        ports = list_midi_output_ports()
        auto_choice = autodetect_midi_output_port(ports) if ports else None
        auto_label = f"Auto ({auto_choice})" if auto_choice else "Auto"
        values = [auto_label] + ports if ports else ["Auto"]
        self.port_box.configure(values=values)
        current = self.port_var.get()
        if current in values and not current.startswith("Auto"):
            return
        self.port_var.set(auto_label)

    def _push_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status_indicator(self, text: str) -> None:
        lowered = text.lower()
        if "error" in lowered:
            bg = PALETTE["danger"]
            fg = PALETTE["panel"]
        elif "play" in lowered:
            bg = PALETTE["forest"]
            fg = PALETTE["panel"]
        elif "buffer" in lowered or "load" in lowered or "start" in lowered or "stop" in lowered:
            bg = PALETTE["gold"]
            fg = PALETTE["panel"]
        else:
            bg = PALETTE["gold_soft"]
            fg = PALETTE["ink"]
        self.status_chip.configure(text=text, bg=bg, fg=fg)

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "log":
                    self._push_log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                    self._set_status_indicator(payload)
                elif kind == "done":
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set(payload)
                    self._set_status_indicator(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _settings_payload(self) -> dict:
        return {
            "workspace": self.workspace_var.get(),
            "model": self.model_var.get(),
            "port": self.port_var.get(),
            "key": self.key_var.get(),
            "mode": self.mode_var.get(),
            "voices": self.voices_var.get(),
            "style": self.style_var.get(),
            "meter": self.meter_var.get(),
            "texture": self.texture_var.get(),
            "imitation": self.imitation_var.get(),
            "harmonic_rhythm": self.harmonic_rhythm_var.get(),
            "tension": self.tension_var.get(),
            "chromaticism": self.chromaticism_var.get(),
            "temperature": self.temperature_var.get(),
            "min_p": self.min_p_var.get(),
            "tempo_bpm": self.tempo_var.get(),
            "prebuffer_bars": self.prebuffer_var.get(),
            "low_water_bars": self.low_water_var.get(),
            "single_channel": self.single_channel_var.get(),
            "auto_channel": self.auto_channel_var.get(),
            "channel_base": self.channel_base_var.get(),
            "velocity": self.velocity_var.get(),
            "cadence_density": self.cadence_density_var.get(),
            "subject_remind_bars": self.subject_remind_var.get(),
            "mps_bf16": self.mps_bf16_var.get(),
            "subject_mode": self.subject_mode_var.get(),
            "subject_text": self.subject_text_var.get(),
            "subject_midi": self.subject_midi_var.get(),
        }

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            self._toggle_channel_entry()
            return
        try:
            payload = json.loads(SETTINGS_PATH.read_text())
        except Exception:
            self._toggle_channel_entry()
            return

        for key, variable in [
            ("workspace", self.workspace_var),
            ("model", self.model_var),
            ("port", self.port_var),
            ("key", self.key_var),
            ("mode", self.mode_var),
            ("voices", self.voices_var),
            ("style", self.style_var),
            ("meter", self.meter_var),
            ("texture", self.texture_var),
            ("imitation", self.imitation_var),
            ("harmonic_rhythm", self.harmonic_rhythm_var),
            ("tension", self.tension_var),
            ("chromaticism", self.chromaticism_var),
            ("temperature", self.temperature_var),
            ("min_p", self.min_p_var),
            ("tempo_bpm", self.tempo_var),
            ("prebuffer_bars", self.prebuffer_var),
            ("low_water_bars", self.low_water_var),
            ("single_channel", self.single_channel_var),
            ("auto_channel", self.auto_channel_var),
            ("channel_base", self.channel_base_var),
            ("velocity", self.velocity_var),
            ("cadence_density", self.cadence_density_var),
            ("subject_remind_bars", self.subject_remind_var),
            ("mps_bf16", self.mps_bf16_var),
            ("subject_mode", self.subject_mode_var),
            ("subject_text", self.subject_text_var),
            ("subject_midi", self.subject_midi_var),
        ]:
            if key in payload:
                variable.set(payload[key])
        self._toggle_channel_entry()
        self._update_subject_mode()

    def _save_settings(self) -> None:
        try:
            SETTINGS_PATH.write_text(json.dumps(self._settings_payload(), indent=2))
        except Exception:
            pass

    def _resolve_subject_inputs(self) -> tuple[str | None, str | None]:
        mode = self.subject_mode_var.get()
        if mode == "none":
            return None, None
        if mode == "string":
            text = self.subject_text_var.get().strip()
            return (text or None), None
        if mode == "builder":
            text = self.subject_builder.get_subject_string().strip()
            return (text or None), None
        midi_path = self.subject_midi_var.get().strip()
        return None, (midi_path or None)

    def _resolve_port_name(self) -> str:
        display = self.port_var.get().strip()
        if display == "Auto" or display.startswith("Auto ("):
            ports = list_midi_output_ports()
            detected = autodetect_midi_output_port(ports)
            if detected is None:
                raise RuntimeError("No MIDI output ports available.")
            return detected
        return display

    def _validate(self) -> dict | None:
        workspace_root = Path(self.workspace_var.get().strip() or ".")
        model_path = Path(self.model_var.get().strip())
        if not model_path.exists():
            messagebox.showerror(APP_TITLE, "Model checkpoint not found.")
            return None
        tokenizer_path = workspace_root / "data" / "tokenizer.json"
        if not tokenizer_path.exists():
            messagebox.showerror(APP_TITLE, f"Tokenizer not found at {tokenizer_path}")
            return None

        subject_text, subject_midi = self._resolve_subject_inputs()
        if self.subject_mode_var.get() != "none" and not subject_text and not subject_midi:
            messagebox.showerror(APP_TITLE, "Subject mode is enabled but no subject content was provided.")
            return None

        cadence_density = self.cadence_density_var.get().strip() or None
        length = self.length_var.get().strip() or None
        remind_text = self.subject_remind_var.get().strip()

        try:
            port_name = self._resolve_port_name()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return None

        channel_base = autodetect_channel_base(port_name) if self.auto_channel_var.get() else int(self.channel_base_var.get())
        return {
            "workspace_root": workspace_root,
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "port_name": port_name,
            "key": self.key_var.get().strip(),
            "mode": self.mode_var.get().strip(),
            "voices": int(self.voices_var.get()),
            "style": self.style_var.get().strip(),
            "length": length,
            "meter": self.meter_var.get().strip() or None,
            "texture": self.texture_var.get().strip() or None,
            "imitation": self.imitation_var.get().strip() or None,
            "harmonic_rhythm": self.harmonic_rhythm_var.get().strip() or None,
            "harmonic_tension": self.tension_var.get().strip() or None,
            "chromaticism": self.chromaticism_var.get().strip() or None,
            "temperature": float(self.temperature_var.get()),
            "min_p": float(self.min_p_var.get()),
            "tempo_bpm": float(self.tempo_var.get()),
            "prebuffer_bars": int(self.prebuffer_var.get()),
            "low_water_bars": int(self.low_water_var.get()),
            "single_channel": bool(self.single_channel_var.get()),
            "channel_base": channel_base,
            "velocity": int(self.velocity_var.get()),
            "cadence_density": cadence_density,
            "subject_remind_bars": int(remind_text) if remind_text else None,
            "mps_bf16": bool(self.mps_bf16_var.get()),
            "subject_str": subject_text,
            "subject_midi": subject_midi,
        }

    def _load_model_and_tokenizer(self, config: dict):
        model_path = str(config["model_path"])
        tokenizer_path = str(config["tokenizer_path"])
        if self.cached_model is not None and self.cached_model_path == model_path and self.cached_tokenizer_path == tokenizer_path:
            return self.cached_model, self.cached_tokenizer

        self.event_queue.put(("status", "Loading model..."))
        model, _ = Trainer.load_checkpoint(model_path)
        model, precision_state = _maybe_cast_generation_model_for_mps(model, mps_bf16=config["mps_bf16"])
        tokenizer = load_tokenizer(tokenizer_path)
        self.cached_model_path = model_path
        self.cached_tokenizer_path = tokenizer_path
        self.cached_model = model
        self.cached_tokenizer = tokenizer
        if precision_state == "enabled":
            self.event_queue.put(("log", "MPS bf16 enabled for live generation."))
        return model, tokenizer

    def start_session(self) -> None:
        if self.session_thread and self.session_thread.is_alive():
            return
        config = self._validate()
        if config is None:
            return

        self._save_settings()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Starting...")
        self._set_status_indicator("Starting...")
        self.event_queue.put(("log", f"Starting live session on port: {config['port_name']}"))
        self.stop_event = threading.Event()

        def _worker() -> None:
            try:
                model, tokenizer = self._load_model_and_tokenizer(config)
                with mido.open_output(config["port_name"]) as midi_out:
                    run_live_generation(
                        model=model,
                        tokenizer=tokenizer,
                        midi_out=midi_out,
                        key_str=config["key"],
                        subject_str=config["subject_str"],
                        subject_midi=config["subject_midi"],
                        temperature=config["temperature"],
                        min_p=config["min_p"],
                        form=config["mode"],
                        num_voices=config["voices"],
                        style=config["style"],
                        length=config["length"],
                        meter=config["meter"],
                        texture=config["texture"],
                        imitation=config["imitation"],
                        harmonic_rhythm=config["harmonic_rhythm"],
                        harmonic_tension=config["harmonic_tension"],
                        chromaticism=config["chromaticism"],
                        cadence_density=config["cadence_density"],
                        subject_remind_bars=config["subject_remind_bars"],
                        tempo_bpm=config["tempo_bpm"],
                        prebuffer_bars=config["prebuffer_bars"],
                        low_water_bars=config["low_water_bars"],
                        channel_base=config["channel_base"],
                        single_channel=config["single_channel"],
                        velocity=config["velocity"],
                        external_stop_event=self.stop_event,
                        status_callback=lambda text: self.event_queue.put(("status", text)),
                    )
                self.event_queue.put(("done", "Stopped"))
            except Exception as exc:
                self.event_queue.put(("log", traceback.format_exc()))
                self.event_queue.put(("done", f"Error: {exc}"))

        self.session_thread = threading.Thread(target=_worker, name="bach-gen-studio-session", daemon=True)
        self.session_thread.start()

    def stop_session(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self.status_var.set("Stopping...")
        self._set_status_indicator("Stopping...")
        self.event_queue.put(("log", "Stop requested."))

    def on_close(self) -> None:
        self._save_settings()
        self.stop_session()
        self.root.after(150, self.root.destroy)


def main() -> None:
    if tk is None:
        raise RuntimeError("tkinter is not available in this Python installation.")
    root = tk.Tk()
    BachGenStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
