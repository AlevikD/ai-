"""
Alevik AI - Personal Assistant App Shell
==========================================
A PySide6 desktop shell matching the Alevik AI mockup:
- Left sidebar navigation (Chat, Apps, Files, Projects, System, Notes, Memory, Settings)
- Top toolbar with Claude / Local AI (LM Studio) mode toggle
- Center chat panel
- Right panel: system stats (CPU/RAM + real NVIDIA GPU usage) and Open-Meteo weather

Run with:
    pip install PySide6 psutil requests spotipy nvidia-ml-py
    # Optional, faster NVIDIA GPU polling:
    pip install nvidia-ml-py
    python alevik_ai.py

Notes:
- GPU % is read from NVIDIA NVML when `nvidia-ml-py` is installed.
  If it is not installed, the app automatically falls back to `nvidia-smi`,
  which is included with the NVIDIA driver.
- LM Studio integration point is in `LocalAIClient` — point it at your
  LM Studio local server (default: http://localhost:1234/v1).
- Claude integration point is in `ClaudeClient` — plug in your API key
  via an environment variable, never hardcode it.
- Weather uses Open-Meteo directly (no API key). Set ALEVIK_WEATHER_LOCATION
  to your city; the fallback location is Helsinki.
"""

import sys
import os
import shutil
import subprocess
import json
import time
from collections import deque
from pathlib import Path

import psutil
import requests

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:
    spotipy = None
    SpotifyOAuth = None

try:
    import keyboard as global_keyboard
except ImportError:
    global_keyboard = None

try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    from ctypes import cast, POINTER
except Exception:
    AudioUtilities = None
    IAudioEndpointVolume = None
    CLSCTX_ALL = None
    cast = POINTER = None
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFrame, QScrollArea, QSizePolicy,
    QComboBox, QProgressBar, QGridLayout, QDialog, QFormLayout, QDialogButtonBox, QMessageBox,
    QStackedWidget, QListWidget, QListWidgetItem, QFileDialog, QCheckBox, QSlider, QGroupBox,
    QTextEdit, QMenu, QSystemTrayIcon, QStyle, QInputDialog, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QPointF, QRectF, QObject, QSize
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QPixmap, QPalette, QBrush, QPolygonF, QAction, QShortcut, QKeySequence, QIcon


# ----------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------

DARK_BG = "#0a1420"
PANEL_BG = "#0f1c2e"
PANEL_BORDER = "#1c3149"
ACCENT = "#18e6ff"
ACCENT_DIM = "#123f55"
TEXT_PRIMARY = "#e8f1f8"
TEXT_SECONDARY = "#7ea3bd"

DEFAULT_WEATHER_LOCATION = os.environ.get("ALEVIK_WEATHER_LOCATION", "Helsinki")
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

SETTINGS_FILE = Path.home() / ".alevik_ai_settings.json"
STATE_FILE = Path.home() / ".alevik_ai_state.json"
CRASH_LOG_FILE = Path.home() / ".alevik_ai_crash.log"
APP_VERSION = "0.3.0"

def load_local_settings():
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_local_settings(data):
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

STYLESHEET = f"""
QMainWindow {{
    background-color: {DARK_BG};
}}
QWidget {{
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', sans-serif;
}}
#sidebar {{
    background-color: {PANEL_BG};
    border-right: 1px solid {PANEL_BORDER};
}}
#navButton {{
    text-align: left;
    padding: 10px 14px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: {TEXT_SECONDARY};
    font-size: 13px;
}}
#navButton:hover {{
    background-color: {ACCENT_DIM};
    color: {TEXT_PRIMARY};
}}
#navButtonActive {{
    text-align: left;
    padding: 10px 14px;
    border-radius: 8px;
    border: none;
    background-color: {ACCENT_DIM};
    color: {ACCENT};
    font-size: 13px;
    font-weight: 600;
}}
#panel {{
    background-color: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
}}
#chatBubbleUser {{
    background-color: {ACCENT_DIM};
    border-radius: 10px;
}}
#chatBubbleAssistant {{
    background-color: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
}}
QLineEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
}}
QPushButton#sendButton {{
    background-color: {ACCENT};
    border-radius: 10px;
    color: {DARK_BG};
    font-weight: 700;
}}
QComboBox {{
    background-color: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    border-radius: 8px;
    padding: 6px 10px;
}}
QLabel#dim {{
    color: {TEXT_SECONDARY};
}}
QLabel#title {{
    font-size: 20px;
    font-weight: 700;
}}
"""


# ----------------------------------------------------------------------
# AI clients
# ----------------------------------------------------------------------



class ClaudeClient:
    """Real Claude API client. Set ANTHROPIC_API_KEY as an environment
    variable — never hardcode the key in source."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        cfg = load_local_settings().get("integrations", {})
        self.api_key = cfg.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")

    def send(self, message: str) -> str:
        if not self.api_key:
            return ("[Claude] No ANTHROPIC_API_KEY environment variable set. "
                    "Set it and restart the app to enable Claude.")
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": message}],
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            return "".join(parts) or "[Claude] (empty response)"
        except requests.exceptions.RequestException as e:
            return f"[Claude] Request failed: {e}"


class LocalAIClient:
    """LM Studio local server client (OpenAI-compatible API).

    LM Studio exposes an OpenAI-style server, default at
    http://localhost:1234/v1, once you start it from the
    'Local Server' tab / `lms server start`.

    Model routing is ported from fresbee.py's AI Assistant page: instead
    of always hitting the same model, it looks at the message and picks
    whichever of your three loaded models best fits the task —
    Flash for quick/simple questions, the 9B (Qwen-class) model for
    recommendations/comparisons/hard reasoning, and the 4B model
    (Gemma-class) for everything in between.
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        cfg = load_local_settings().get("integrations", {})
        self.base_url = (cfg.get("lm_studio_url") or base_url).rstrip("/")

    def list_models(self) -> list[str]:
        """Return the model ids currently loaded/available in LM Studio."""
        resp = requests.get(f"{self.base_url}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    # ------------------------------------------------------------------
    # Smart routing — ported from fresbee.py's choose_ai_models()
    # ------------------------------------------------------------------

    QUICK_TERMS = (
        "what is", "who is", "define", "meaning of", "quick question",
        "short answer", "yes or no", "mikä on", "kuka on",
    )
    COMPLEX_TERMS = (
        "recommend", "suositta", "compare", "vertaa", "vertailu",
        "which is better", "mikä sopii", "sopivin", "paras", "best",
        "analyze", "analysoi", "analyysi", "suggest", "ehdota",
        "should i", "kannattaako", "plan", "suunnittele", "design",
        "debug", "refactor", "architecture", "trade-off", "pros and cons",
    )
    COMPARISON_TERMS = (
        " vs ", " vs. ", "ero", "erot", "parempi kuin", "kumpi", "kumpaa",
        "verrattuna", "versus",
    )

    def choose_model(self, message: str, models: list[str]) -> tuple[str, str]:
        """Pick exactly one loaded model for this message.

        Returns (label, model_id). label is a short human-readable tag
        ('Flash' / '4B' / '9B') for display in the UI.
        """
        text = " ".join(str(message).lower().split())

        def find_model(*names):
            for name in names:
                for m in models:
                    if name.lower() in m.lower():
                        return m
            return None

        flash = find_model("flash")
        mid = find_model("4b", "gemma")
        big = find_model("9b", "qwen")
        fallback = models[0] if models else "local-model"

        quick_score = sum(1 for t in self.QUICK_TERMS if t in text)
        complex_score = sum(1 for t in self.COMPLEX_TERMS if t in text)
        comparison_score = sum(1 for t in self.COMPARISON_TERMS if t in text)

        # Short, simple questions -> Flash (fastest)
        if quick_score > 0 and complex_score == 0 and comparison_score == 0 and len(text) < 80:
            return ("Flash", flash or mid or big or fallback)

        # Recommendations, comparisons, hard reasoning -> 9B
        if complex_score > 0 or comparison_score > 0:
            return ("9B", big or mid or flash or fallback)

        # Everything else -> 4B, a sensible general-purpose middle ground
        return ("4B", mid or flash or big or fallback)

    def call_model(self, model: str, message: str) -> str:
        """Send one message to a specific model id."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "temperature": 0.7,
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def send(self, message: str, model: str | None = None) -> str:
        """Send a message. If `model` is None or 'auto', routes it
        automatically via choose_model(); otherwise calls that exact
        model id directly."""
        try:
            if model and model != "Auto":
                return self.call_model(model, message)

            models = self.list_models()
            if not models:
                return ("[Local AI] No models found. Make sure LM Studio's "
                        "Local Server is running and a model is loaded.")
            label, chosen = self.choose_model(message, models)
            answer = self.call_model(chosen, message)
            return f"[{label} · {chosen}]\n{answer}"
        except requests.exceptions.ConnectionError:
            return ("[Local AI] Couldn't reach LM Studio at "
                    f"{self.base_url}. Make sure the LM Studio local "
                    "server is running (Local Server tab → Start Server).")
        except requests.exceptions.RequestException as e:
            return f"[Local AI] Request failed: {e}"


# ----------------------------------------------------------------------
# Weather service — Open-Meteo (no API key required)
# ----------------------------------------------------------------------


class WeatherService:
    """Small Open-Meteo client used by both chat routing and the sidebar widget.

    Weather requests bypass LM Studio / Claude completely, so asking for the
    weather does not need to spin up the GPU.
    """

    WEATHER_TERMS = (
        "sää", "saa", "weather", "lämpötila", "lampotila", "temperature",
        "sataako", "sadetta", "rain", "tuuli", "wind", "ennuste", "forecast",
    )

    WMO_DESCRIPTIONS = {
        0: "Selkeää", 1: "Enimmäkseen selkeää", 2: "Puolipilvistä", 3: "Pilvistä",
        45: "Sumua", 48: "Jäätävää sumua",
        51: "Heikkoa tihkua", 53: "Tihkua", 55: "Voimakasta tihkua",
        56: "Heikkoa jäätävää tihkua", 57: "Jäätävää tihkua",
        61: "Heikkoa sadetta", 63: "Sadetta", 65: "Voimakasta sadetta",
        66: "Heikkoa jäätävää sadetta", 67: "Jäätävää sadetta",
        71: "Heikkoa lumisadetta", 73: "Lumisadetta", 75: "Voimakasta lumisadetta",
        77: "Lumijyväsiä", 80: "Heikkoja sadekuuroja", 81: "Sadekuuroja",
        82: "Voimakkaita sadekuuroja", 85: "Heikkoja lumikuuroja",
        86: "Voimakkaita lumikuuroja", 95: "Ukkosta",
        96: "Ukkosta ja heikkoa raetta", 99: "Ukkosta ja voimakasta raetta",
    }

    def __init__(self, default_location: str = DEFAULT_WEATHER_LOCATION):
        cfg = load_local_settings().get("weather", {})
        self.default_location = cfg.get("location") or default_location

    @classmethod
    def is_weather_query(cls, message: str) -> bool:
        text = " ".join(str(message).lower().split())
        return any(term in text for term in cls.WEATHER_TERMS)

    def _extract_location(self, message: str) -> str:
        """Pick a location from a short weather command when one is supplied.

        Examples:
            'sää Tampere' -> Tampere
            'weather in Turku' -> Turku
        Otherwise use the configured default location.
        """
        raw = str(message).strip()
        lower = raw.lower()

        for prefix in ("sää ", "saa ", "weather in ", "weather ", "sää kaupungissa "):
            if lower.startswith(prefix):
                location = raw[len(prefix):].strip(" ?!.,")
                # Avoid interpreting common time words as cities.
                if location.lower() not in {
                    "tänään", "tanaan", "nyt", "huomenna", "today", "now", "tomorrow"
                } and location:
                    return location

        return self.default_location

    def geocode(self, location: str) -> dict:
        resp = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params={"name": location, "count": 1, "language": "fi", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            raise ValueError(f"Paikkaa '{location}' ei löytynyt.")
        item = results[0]
        return {
            "name": item.get("name", location),
            "admin1": item.get("admin1", ""),
            "country": item.get("country", ""),
            "latitude": item["latitude"],
            "longitude": item["longitude"],
        }

    def get_weather(self, location: str | None = None) -> dict:
        place = self.geocode(location or self.default_location)
        resp = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "timezone": "auto",
                "current": ",".join([
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "weather_code",
                    "wind_speed_10m",
                ]),
                "daily": ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "weather_code",
                ]),
                "forecast_days": 2,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        daily = data.get("daily", {})

        code = int(current.get("weather_code", -1))
        tomorrow_code = None
        if len(daily.get("weather_code", [])) > 1:
            tomorrow_code = int(daily["weather_code"][1])

        return {
            "location": place["name"],
            "region": place.get("admin1", ""),
            "country": place.get("country", ""),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind": current.get("wind_speed_10m"),
            "weather_code": code,
            "description": self.WMO_DESCRIPTIONS.get(code, "Tuntematon säätila"),
            "tomorrow_min": daily.get("temperature_2m_min", [None, None])[1] if len(daily.get("temperature_2m_min", [])) > 1 else None,
            "tomorrow_max": daily.get("temperature_2m_max", [None, None])[1] if len(daily.get("temperature_2m_max", [])) > 1 else None,
            "tomorrow_rain": daily.get("precipitation_probability_max", [None, None])[1] if len(daily.get("precipitation_probability_max", [])) > 1 else None,
            "tomorrow_description": self.WMO_DESCRIPTIONS.get(tomorrow_code, "") if tomorrow_code is not None else "",
        }

    def answer(self, message: str) -> str:
        location = self._extract_location(message)
        try:
            w = self.get_weather(location)
            tomorrow = ""
            if w["tomorrow_min"] is not None and w["tomorrow_max"] is not None:
                tomorrow = (
                    f"\nHuomenna: {w['tomorrow_description']}, "
                    f"{w['tomorrow_min']:.0f}–{w['tomorrow_max']:.0f} °C"
                )
                if w["tomorrow_rain"] is not None:
                    tomorrow += f", sateen mahdollisuus {w['tomorrow_rain']:.0f} %"

            return (
                f"🌤 {w['location']} — {w['description']}\n"
                f"Lämpötila {w['temperature']:.1f} °C, tuntuu kuin {w['feels_like']:.1f} °C\n"
                f"Kosteus {w['humidity']:.0f} %, tuuli {w['wind']:.1f} km/h"
                f"{tomorrow}\n\n[Open-Meteo]"
            )
        except requests.exceptions.RequestException as e:
            return f"[Weather] Open-Meteo-pyyntö epäonnistui: {e}"
        except (ValueError, KeyError, TypeError) as e:
            return f"[Weather] {e}"


# ----------------------------------------------------------------------
# Background worker — runs network calls off the UI thread
# ----------------------------------------------------------------------

class Worker(QThread):
    """Runs a callable in a background thread and emits its result."""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

class Sidebar(QWidget):
    nav_changed = Signal(str)

    NAV_ITEMS = ["Chat", "Apps", "Files", "Projects", "System", "Notes", "Memory", "Settings"]

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(6)

        logo = QLabel("A  Alevik AI")
        logo.setObjectName("title")
        layout.addWidget(logo)
        sub = QLabel("YOUR PERSONAL ASSISTANT")
        sub.setObjectName("dim")
        sub.setStyleSheet("font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(sub)
        layout.addSpacing(20)

        self.buttons = {}
        for item in self.NAV_ITEMS:
            btn = QPushButton(item)
            btn.setObjectName("navButtonActive" if item == "Chat" else "navButton")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, name=item: self._on_click(name))
            layout.addWidget(btn)
            self.buttons[item] = btn

        layout.addStretch()

        user_row = QHBoxLayout()
        avatar = QLabel("D")
        avatar.setFixedSize(36, 36)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(f"background-color: {ACCENT_DIM}; border-radius: 18px; font-weight: 700;")
        user_row.addWidget(avatar)
        name_col = QVBoxLayout()
        name_col.setSpacing(0)
        name_col.addWidget(QLabel("Daniel"))
        online = QLabel("● Online")
        online.setStyleSheet("color: #3fd97f; font-size: 11px;")
        name_col.addWidget(online)
        user_row.addLayout(name_col)
        user_row.addStretch()
        layout.addLayout(user_row)

    def _on_click(self, name: str):
        for key, btn in self.buttons.items():
            btn.setObjectName("navButtonActive" if key == name else "navButton")
            btn.setStyle(btn.style())
        self.nav_changed.emit(name)


# ----------------------------------------------------------------------
# Top bar: AI mode toggle
# ----------------------------------------------------------------------

class TopBar(QWidget):
    mode_changed = Signal(str)
    local_model_changed = Signal(str)

    # Fallback list shown before LM Studio responds / if it's offline.
    # "Auto" routes each message via LocalAIClient.choose_model() —
    # ported from fresbee.py's smart Flash/4B/9B routing.
    FALLBACK_LOCAL_MODELS = ["Auto", "flash", "4b", "9b"]

    def __init__(self, local_ai: "LocalAIClient"):
        super().__init__()
        self.local_ai = local_ai
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.mode_select = QComboBox()
        self.mode_select.addItems(["Claude — for complex tasks", "Local AI — LM Studio"])
        self.mode_select.currentIndexChanged.connect(self._on_mode_index_changed)
        layout.addWidget(self.mode_select)

        self.model_select = QComboBox()
        self.model_select.addItems(self.FALLBACK_LOCAL_MODELS)
        self.model_select.currentTextChanged.connect(self.local_model_changed.emit)
        self.model_select.setVisible(False)  # only shown in Local AI mode
        layout.addWidget(self.model_select)

        self.refresh_btn = QPushButton("⟳")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip("Refresh model list from LM Studio")
        self.refresh_btn.clicked.connect(self.refresh_models)
        self.refresh_btn.setVisible(False)
        layout.addWidget(self.refresh_btn)

        layout.addStretch()

        clock = QLabel()
        clock.setObjectName("dim")
        layout.addWidget(clock)
        self._clock_label = clock
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(1000)
        self._tick()

        self._models_worker = None

    def _on_mode_index_changed(self, i: int):
        mode = "claude" if i == 0 else "local"
        is_local = mode == "local"
        self.model_select.setVisible(is_local)
        self.refresh_btn.setVisible(is_local)
        self.mode_changed.emit(mode)
        if is_local:
            self.refresh_models()
            self.local_model_changed.emit(self.model_select.currentText())

    def refresh_models(self):
        """Ask LM Studio what models it currently has loaded."""
        self.refresh_btn.setEnabled(False)
        self._models_worker = Worker(self.local_ai.list_models)
        self._models_worker.finished.connect(self._on_models_loaded)
        self._models_worker.error.connect(self._on_models_error)
        self._models_worker.start()

    def _on_models_loaded(self, models: list):
        self.refresh_btn.setEnabled(True)
        if not models:
            return
        current = self.model_select.currentText()
        self.model_select.blockSignals(True)
        self.model_select.clear()
        self.model_select.addItem("Auto")
        self.model_select.addItems(models)
        self.model_select.blockSignals(False)
        if current in models or current == "Auto":
            self.model_select.setCurrentText(current)
        else:
            self.local_model_changed.emit(self.model_select.currentText())

    def _on_models_error(self, message: str):
        self.refresh_btn.setEnabled(True)
        print(f"[Alevik AI] Could not fetch LM Studio models: {message}")

    def _tick(self):
        from datetime import datetime
        self._clock_label.setText(datetime.now().strftime("%H:%M   %a %d.%m.%Y"))


# ----------------------------------------------------------------------
# Chat panel
# ----------------------------------------------------------------------

class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool):
        super().__init__()
        self.setObjectName("chatBubbleUser" if is_user else "chatBubbleAssistant")
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)


class ChatPanel(QWidget):
    def __init__(self, claude: ClaudeClient, local_ai: LocalAIClient, weather: WeatherService):
        super().__init__()
        self.claude = claude
        self.local_ai = local_ai
        self.weather = weather
        self.mode = "claude"
        self.local_model = None
        self._send_worker = None
        self._pending_bubble = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.addStretch()
        self.scroll.setWidget(self.messages_widget)
        layout.addWidget(self.scroll)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Kirjoita viesti tai komento...")
        self.input.returnPressed.connect(self._send)
        input_row.addWidget(self.input)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("sendButton")
        self.send_btn.setFixedWidth(70)
        self.send_btn.clicked.connect(self._send)
        input_row.addWidget(self.send_btn)

        layout.addLayout(input_row)

        self._add_bubble("Avaa sovelluksia, hae tiedostoja tai kysy mitä vain.", is_user=False)

    def set_mode(self, mode: str):
        self.mode = mode

    def set_local_model(self, model_id: str):
        self.local_model = model_id

    def _add_bubble(self, text: str, is_user: bool) -> ChatBubble:
        bubble = ChatBubble(text, is_user)
        # Insert before the trailing stretch
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
        return bubble

    def _send(self):
        text = self.input.text().strip()
        if not text or self._send_worker is not None:
            return  # ignore double-sends while a request is in flight
        self._add_bubble(text, is_user=True)
        self.input.clear()

        self._pending_bubble = self._add_bubble("Thinking…", is_user=False)
        self.send_btn.setEnabled(False)

        # Tool routing happens before any language model. Weather goes straight
        # to Open-Meteo, so LM Studio does not use the GPU for a weather lookup.
        if self.weather.is_weather_query(text):
            self._send_worker = Worker(self.weather.answer, text)
        elif self.mode == "claude":
            self._send_worker = Worker(self.claude.send, text)
        else:
            self._send_worker = Worker(self.local_ai.send, text, model=self.local_model)

        self._send_worker.finished.connect(self._on_response)
        self._send_worker.error.connect(self._on_response_error)
        self._send_worker.start()

    def _on_response(self, response: str):
        self._pending_bubble.findChild(QLabel).setText(response)
        self._finish_send()

    def _on_response_error(self, message: str):
        self._pending_bubble.findChild(QLabel).setText(f"[Error] {message}")
        self._finish_send()

    def _finish_send(self):
        self._pending_bubble = None
        self._send_worker = None
        self.send_btn.setEnabled(True)


# ----------------------------------------------------------------------
# System stats panel
# ----------------------------------------------------------------------


class GPUReader:
    """Read NVIDIA GPU utilization without making the whole app depend on
    an extra package.

    Preferred path: NVIDIA NVML through the optional ``nvidia-ml-py`` package.
    Fallback: ``nvidia-smi`` from the installed NVIDIA graphics driver.
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._nvml_handle = None
        self._nvidia_smi = self._find_nvidia_smi()

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._nvml_handle = None

    @staticmethod
    def _find_nvidia_smi():
        candidates = [
            shutil.which("nvidia-smi"),
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def usage_percent(self):
        """Return GPU core utilization as an integer from 0-100.

        Returns None if no supported NVIDIA GPU reader is available.
        """
        if self._nvml_handle is not None and pynvml is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                return max(0, min(100, int(util.gpu)))
            except Exception:
                # If NVML fails at runtime, still try nvidia-smi below.
                pass

        if self._nvidia_smi:
            try:
                kwargs = {}
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

                result = subprocess.run(
                    [
                        self._nvidia_smi,
                        f"--id={self.gpu_index}",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                    **kwargs,
                )
                if result.returncode == 0:
                    first_line = result.stdout.strip().splitlines()[0]
                    return max(0, min(100, int(float(first_line.strip()))))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass

        return None


    def stats(self):
        """Return utilization, temperature and VRAM information."""
        data = {"usage": self.usage_percent(), "temp": None, "vram_used_mb": None, "vram_total_mb": None}
        if self._nvml_handle is not None and pynvml is not None:
            try:
                data["temp"] = int(pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                data["vram_used_mb"] = mem.used / (1024 ** 2)
                data["vram_total_mb"] = mem.total / (1024 ** 2)
                return data
            except Exception:
                pass
        if self._nvidia_smi:
            try:
                kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                r = subprocess.run([
                    self._nvidia_smi, f"--id={self.gpu_index}",
                    "--query-gpu=temperature.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, timeout=2, check=False, **kwargs)
                if r.returncode == 0:
                    vals = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
                    data["temp"] = int(float(vals[0]))
                    data["vram_used_mb"] = float(vals[1])
                    data["vram_total_mb"] = float(vals[2])
            except Exception:
                pass
        return data


class SpotifyService:
    """Spotify Web API integration. Credentials/config stay only on this PC."""
    def __init__(self):
        self.client = None
        self.error = None
        self.reload()

    def config(self):
        cfg = load_local_settings().get("spotify", {})
        return {
            "client_id": cfg.get("client_id") or os.environ.get("SPOTIPY_CLIENT_ID", ""),
            "client_secret": cfg.get("client_secret") or os.environ.get("SPOTIPY_CLIENT_SECRET", ""),
            "redirect_uri": cfg.get("redirect_uri") or os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
        }

    def save_config(self, client_id, client_secret, redirect_uri):
        data = load_local_settings()
        data["spotify"] = {"client_id": client_id.strip(), "client_secret": client_secret.strip(), "redirect_uri": redirect_uri.strip()}
        save_local_settings(data)
        self.reload()

    def reload(self):
        self.client = None; self.error = None
        cfg = self.config()
        if spotipy is None:
            self.error = "Asenna spotipy: pip install spotipy"; return
        if not cfg["client_id"]:
            self.error = "Spotify Client ID puuttuu"; return
        try:
            auth = SpotifyOAuth(
                client_id=cfg["client_id"], client_secret=cfg["client_secret"] or None,
                redirect_uri=cfg["redirect_uri"],
                scope="user-read-playback-state user-modify-playback-state user-read-currently-playing",
                cache_path=str(Path.home() / ".alevik_ai_spotify_cache"), open_browser=True,
            )
            self.client = spotipy.Spotify(auth_manager=auth, requests_timeout=5)
        except Exception as e:
            self.error = str(e)

    def now_playing(self):
        if not self.client:
            return {"connected": False, "message": self.error or "Spotify ei ole yhdistetty"}
        try:
            state = self.client.current_playback()
            if not state or not state.get("item"):
                return {"connected": True, "playing": False, "title": "Ei toistoa", "artist": "Spotify", "progress": 0, "duration": 0, "image": ""}
            item = state["item"]
            images = (item.get("album") or {}).get("images") or []
            return {
                "connected": True, "playing": bool(state.get("is_playing")),
                "title": item.get("name", "—"),
                "artist": ", ".join(a.get("name", "") for a in item.get("artists", [])),
                "progress": int(state.get("progress_ms") or 0), "duration": int(item.get("duration_ms") or 0),
                "image": images[0].get("url", "") if images else "",
                "shuffle": bool(state.get("shuffle_state", False)),
                "repeat": state.get("repeat_state", "off") or "off",
            }
        except Exception as e:
            return {"connected": False, "message": str(e)}

    def toggle(self):
        if not self.client:
            return
        state = self.client.current_playback()
        if state and state.get("is_playing"):
            self.client.pause_playback()
        else:
            self.client.start_playback()

    def next(self):
        if self.client:
            self.client.next_track()

    def previous(self):
        if self.client:
            self.client.previous_track()

    def toggle_shuffle(self):
        if not self.client:
            return
        state = self.client.current_playback() or {}
        enabled = bool(state.get("shuffle_state", False))
        self.client.shuffle(not enabled)

    def cycle_repeat(self):
        if not self.client:
            return
        state = self.client.current_playback() or {}
        current = state.get("repeat_state", "off") or "off"
        order = ["off", "context", "track"]
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        self.client.repeat(order[(idx + 1) % len(order)])


class RecentAppsTracker:
    FRIENDLY = {
        "code.exe": "VS Code", "spotify.exe": "Spotify", "discord.exe": "Discord",
        "chrome.exe": "Chrome", "msedge.exe": "Edge", "steam.exe": "Steam",
        "explorer.exe": "File Explorer", "pycharm64.exe": "PyCharm",
        "devenv.exe": "Visual Studio", "obs64.exe": "OBS Studio",
        "lm studio.exe": "LM Studio", "lmstudio.exe": "LM Studio",
    }
    def __init__(self): self.recent = deque(maxlen=6)
    def refresh(self):
        found=[]
        for proc in psutil.process_iter(["name"]):
            try:
                raw=(proc.info.get("name") or "").lower()
                if raw in self.FRIENDLY and self.FRIENDLY[raw] not in found: found.append(self.FRIENDLY[raw])
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        for name in found:
            if name in self.recent: self.recent.remove(name)
            self.recent.appendleft(name)
        return list(self.recent)


class SpotifySettingsDialog(QDialog):
    def __init__(self, service, parent=None):
        super().__init__(parent); self.service=service; self.setWindowTitle("Spotify Integration"); self.setMinimumWidth(480)
        form=QFormLayout(self); cfg=service.config()
        self.cid=QLineEdit(cfg["client_id"]); self.secret=QLineEdit(cfg["client_secret"]); self.secret.setEchoMode(QLineEdit.Password)
        self.redirect=QLineEdit(cfg["redirect_uri"])
        form.addRow("Client ID", self.cid); form.addRow("Client Secret", self.secret); form.addRow("Redirect URI", self.redirect)
        note=QLabel("Tiedot tallennetaan vain tälle PC:lle. Redirect URI pitää lisätä täsmälleen samana Spotify Developer Dashboardiin.")
        note.setWordWrap(True); note.setObjectName("dim"); form.addRow(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel); buttons.accepted.connect(self.save); buttons.rejected.connect(self.reject); form.addRow(buttons)
    def save(self):
        if not self.cid.text().strip(): QMessageBox.warning(self,"Spotify","Client ID tarvitaan."); return
        self.service.save_config(self.cid.text(),self.secret.text(),self.redirect.text()); self.accept()


def _fmt_ms(ms):
    sec=max(0,int(ms or 0)//1000); return f"{sec//60}:{sec%60:02d}"

class SpotifyIconButton(QPushButton):
    """High-quality vector Spotify control button drawn with QPainter."""
    SPOTIFY_GREEN = QColor("#1ED760")
    WHITE = QColor("#F5F7FA")
    CYAN = QColor(ACCENT)
    MUTED = QColor("#8AA7BA")

    def __init__(self, icon_name, parent=None, *, primary=False):
        super().__init__(parent)
        self.icon_name = icon_name
        self.primary = primary
        self.active = False
        self.repeat_one = False
        self.playing = False
        self.hovered = False
        self.setText("")
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet("background: transparent; border: none;")
        self.setFixedSize(58, 58) if primary else self.setFixedSize(38, 38)

    def set_active(self, active, *, repeat_one=False):
        self.active = bool(active)
        self.repeat_one = bool(repeat_one)
        self.update()

    def set_playing(self, playing):
        self.playing = bool(playing)
        self.update()

    def enterEvent(self, event):
        self.hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovered = False
        self.update()
        super().leaveEvent(event)

    def _pen(self, color, width=2.2):
        pen = QPen(color, width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect()

        if self.primary:
            # CD / LP inspired center control: subtle discs + cyan outer ring.
            center = QPointF(r.center())
            radius = min(r.width(), r.height()) / 2 - 3
            if self.hovered:
                p.setPen(QPen(QColor(24, 230, 255, 85), 7))
                p.drawEllipse(center, radius - 1, radius - 1)
            p.setBrush(QColor("#10263A"))
            p.setPen(self._pen(self.CYAN if self.hovered else QColor("#315C78"), 2.0))
            p.drawEllipse(center, radius - 2, radius - 2)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("#1D4057"), 1.1))
            p.drawEllipse(center, radius - 9, radius - 9)
            p.drawEllipse(center, 4.2, 4.2)
            icon_color = self.WHITE
        else:
            if self.hovered:
                p.setBrush(QColor(24, 230, 255, 18))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QRectF(2, 2, r.width()-4, r.height()-4))
            icon_color = self.SPOTIFY_GREEN if self.active else self.WHITE

        p.setPen(self._pen(icon_color, 2.25))
        p.setBrush(icon_color)
        cx, cy = r.center().x(), r.center().y()

        if self.icon_name == "playpause":
            if self.playing:
                # Two clean pause pillars.
                w, h, gap = 5.2, 18.0, 4.2
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(QRectF(cx-gap-w, cy-h/2, w, h), 1.5, 1.5)
                p.drawRoundedRect(QRectF(cx+gap, cy-h/2, w, h), 1.5, 1.5)
            else:
                p.setPen(Qt.NoPen)
                tri = QPolygonF([
                    QPointF(cx-5.5, cy-9.5),
                    QPointF(cx-5.5, cy+9.5),
                    QPointF(cx+10.0, cy),
                ])
                p.drawPolygon(tri)

        elif self.icon_name in ("previous", "next"):
            direction = -1 if self.icon_name == "previous" else 1
            p.setPen(Qt.NoPen)
            # Two sharp triangles + stop bar, matching premium media controls.
            for offset in (-3.0, 6.0):
                x = cx + direction * offset
                if direction > 0:
                    tri = QPolygonF([QPointF(x-6, cy-8), QPointF(x-6, cy+8), QPointF(x+3, cy)])
                else:
                    tri = QPolygonF([QPointF(x+6, cy-8), QPointF(x+6, cy+8), QPointF(x-3, cy)])
                p.drawPolygon(tri)
            bar_x = cx + direction * 11.0
            p.drawRoundedRect(QRectF(bar_x-1.3, cy-8, 2.6, 16), 1.2, 1.2)

        elif self.icon_name == "shuffle":
            p.setBrush(Qt.NoBrush)
            p.setPen(self._pen(icon_color, 2.0))
            # Two smooth crossing paths.
            p.drawLine(QPointF(cx-10, cy-7), QPointF(cx-6, cy-7))
            p.drawLine(QPointF(cx-6, cy-7), QPointF(cx+6, cy+7))
            p.drawLine(QPointF(cx+6, cy+7), QPointF(cx+9, cy+7))
            p.drawLine(QPointF(cx-10, cy+7), QPointF(cx-6, cy+7))
            p.drawLine(QPointF(cx-6, cy+7), QPointF(cx-1, cy+2))
            p.drawLine(QPointF(cx+1, cy-2), QPointF(cx+6, cy-7))
            p.drawLine(QPointF(cx+6, cy-7), QPointF(cx+9, cy-7))
            # Arrow heads.
            p.drawLine(QPointF(cx+6, cy+4), QPointF(cx+10, cy+7))
            p.drawLine(QPointF(cx+10, cy+7), QPointF(cx+6, cy+10))
            p.drawLine(QPointF(cx+6, cy-10), QPointF(cx+10, cy-7))
            p.drawLine(QPointF(cx+10, cy-7), QPointF(cx+6, cy-4))

        elif self.icon_name == "repeat":
            p.setBrush(Qt.NoBrush)
            p.setPen(self._pen(icon_color, 2.0))
            # Compact repeat loop.
            p.drawLine(QPointF(cx-8, cy-6), QPointF(cx+6, cy-6))
            p.drawLine(QPointF(cx+6, cy-6), QPointF(cx+9, cy-3))
            p.drawLine(QPointF(cx+9, cy-3), QPointF(cx+6, cy))
            p.drawLine(QPointF(cx+8, cy+6), QPointF(cx-6, cy+6))
            p.drawLine(QPointF(cx-6, cy+6), QPointF(cx-9, cy+3))
            p.drawLine(QPointF(cx-9, cy+3), QPointF(cx-6, cy))
            if self.repeat_one:
                f = p.font()
                f.setPixelSize(9)
                f.setBold(True)
                p.setFont(f)
                p.setPen(icon_color)
                p.drawText(QRectF(cx-4, cy-5, 8, 10), Qt.AlignCenter, "1")

        p.end()


class SpotifyPanel(QFrame):
    def __init__(self, service):
        super().__init__()
        self.setObjectName("panel")
        self.service = service
        self.worker = None
        self.control_worker = None
        self.last_image = ""
        self.last_state = {}

        l = QVBoxLayout(self)
        l.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel("Now Playing")
        title.setStyleSheet("font-size:16px;font-weight:700;")
        head.addWidget(title)
        head.addStretch()
        self.setup = QPushButton("Spotify  ⚙")
        self.setup.setToolTip("Spotify-asetukset")
        self.setup.clicked.connect(self.open_settings)
        self.setup.setStyleSheet("color:#1ED760;background:transparent;border:none;font-weight:600;")
        head.addWidget(self.setup)
        l.addLayout(head)

        info = QHBoxLayout()
        self.cover = QLabel("♫")
        self.cover.setFixedSize(72, 72)
        self.cover.setAlignment(Qt.AlignCenter)
        self.cover.setStyleSheet(
            "background:#0a1420;border:1px solid #1c3149;border-radius:8px;font-size:28px;"
        )
        info.addWidget(self.cover)
        text = QVBoxLayout()
        self.title = QLabel("Yhdistetään…")
        self.title.setStyleSheet("font-weight:700;font-size:14px;")
        self.title.setWordWrap(True)
        text.addWidget(self.title)
        self.artist = QLabel("—")
        self.artist.setObjectName("dim")
        text.addWidget(self.artist)
        info.addLayout(text, 1)
        l.addLayout(info)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(
            "QProgressBar{background:#17334A;border:none;border-radius:3px;}"
            "QProgressBar::chunk{background:#18E6FF;border-radius:3px;}"
        )
        l.addWidget(self.progress)

        times = QHBoxLayout()
        self.elapsed = QLabel("0:00")
        self.elapsed.setObjectName("dim")
        self.total = QLabel("0:00")
        self.total.setObjectName("dim")
        times.addWidget(self.elapsed)
        times.addStretch()
        times.addWidget(self.total)
        l.addLayout(times)

        # Premium Spotify-style media controls.
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch()

        self.shuffle_btn = SpotifyIconButton("shuffle")
        self.shuffle_btn.setToolTip("Shuffle")
        self.shuffle_btn.clicked.connect(lambda: self.control(self.service.toggle_shuffle))
        row.addWidget(self.shuffle_btn)

        self.prev_btn = SpotifyIconButton("previous")
        self.prev_btn.setToolTip("Edellinen")
        self.prev_btn.clicked.connect(lambda: self.control(self.service.previous))
        row.addWidget(self.prev_btn)

        self.play_btn = SpotifyIconButton("playpause", primary=True)
        self.play_btn.setToolTip("Play / Pause")
        self.play_btn.clicked.connect(lambda: self.control(self.service.toggle))
        row.addWidget(self.play_btn)

        self.next_btn = SpotifyIconButton("next")
        self.next_btn.setToolTip("Seuraava")
        self.next_btn.clicked.connect(lambda: self.control(self.service.next))
        row.addWidget(self.next_btn)

        self.repeat_btn = SpotifyIconButton("repeat")
        self.repeat_btn.setToolTip("Repeat")
        self.repeat_btn.clicked.connect(lambda: self.control(self.service.cycle_repeat))
        row.addWidget(self.repeat_btn)

        row.addStretch()
        l.addLayout(row)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(3500)
        self.refresh()

    def open_settings(self):
        if SpotifySettingsDialog(self.service, self).exec():
            self.title.setText("Spotify yhdistetään…")
            self.refresh()

    def control(self, fn):
        # Playback API calls happen off the UI thread so the panel stays smooth.
        if self.control_worker:
            return
        self.control_worker = Worker(fn)
        self.control_worker.finished.connect(self.control_done)
        self.control_worker.error.connect(self.control_failed)
        self.control_worker.start()

    def control_done(self, _result=None):
        self.control_worker = None
        QTimer.singleShot(250, self.refresh)

    def control_failed(self, error):
        self.control_worker = None
        self.artist.setText(error)

    def refresh(self):
        if self.worker:
            return
        self.worker = Worker(self.service.now_playing)
        self.worker.finished.connect(self.done)
        self.worker.error.connect(self.fail)
        self.worker.start()

    def done(self, d):
        self.worker = None
        self.last_state = d
        if not d.get("connected"):
            self.title.setText("Spotify ei yhdistetty")
            self.artist.setText(d.get("message", ""))
            self.progress.setValue(0)
            self.shuffle_btn.set_active(False)
            self.repeat_btn.set_active(False)
            self.play_btn.set_playing(False)
            return

        self.title.setText(d.get("title", "—"))
        self.artist.setText(d.get("artist", "—"))
        dur = d.get("duration", 0)
        pos = d.get("progress", 0)
        self.progress.setValue(int(pos / dur * 1000) if dur else 0)
        self.elapsed.setText(_fmt_ms(pos))
        self.total.setText(_fmt_ms(dur))

        self.play_btn.set_playing(bool(d.get("playing")))
        self.shuffle_btn.set_active(bool(d.get("shuffle")))
        repeat_state = d.get("repeat", "off")
        self.repeat_btn.set_active(repeat_state != "off", repeat_one=(repeat_state == "track"))
        self.repeat_btn.setToolTip({
            "off": "Repeat: pois",
            "context": "Repeat: kaikki",
            "track": "Repeat: yksi kappale",
        }.get(repeat_state, "Repeat"))

        url = d.get("image", "")
        if url and url != self.last_image:
            self.last_image = url
            try:
                r = requests.get(url, timeout=3)
                pix = QPixmap()
                pix.loadFromData(r.content)
                self.cover.setPixmap(
                    pix.scaled(72, 72, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                )
            except Exception:
                pass

    def fail(self, e):
        self.worker = None
        self.artist.setText(e)


class RecentAppsPanel(QFrame):
    def __init__(self, tracker):
        super().__init__(); self.setObjectName("panel"); self.tracker=tracker
        self.layout=QVBoxLayout(self); head=QHBoxLayout(); head.addWidget(QLabel("Recent Apps")); head.addStretch(); self.clear=QPushButton("Clear"); self.clear.clicked.connect(self.clear_recent); head.addWidget(self.clear); self.layout.addLayout(head); self.labels=[]
        for _ in range(6): x=QLabel("—"); x.setObjectName("dim"); self.layout.addWidget(x); self.labels.append(x)
        t=QTimer(self); t.timeout.connect(self.refresh); t.start(5000); self.refresh()
    def clear_recent(self): self.tracker.recent.clear(); self.refresh()
    def refresh(self):
        apps=self.tracker.refresh()
        for i,lbl in enumerate(self.labels): lbl.setText(("• "+apps[i]) if i<len(apps) else "—")


class RingStat(QWidget):
    def __init__(self, label: str):
        super().__init__()
        self.value = 0
        self.label_text = label
        self.setFixedSize(90, 100)

    def set_value(self, value):
        # None means the sensor is unavailable. Keep the ring empty and show N/A.
        if value is None:
            self.value = None
        else:
            self.value = max(0, min(100, int(value)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 5, -10, -25)

        pen_bg = QPen(QColor(PANEL_BORDER))
        pen_bg.setWidth(6)
        painter.setPen(pen_bg)
        painter.drawArc(rect, 0, 360 * 16)

        pen_fg = QPen(QColor(ACCENT))
        pen_fg.setWidth(6)
        pen_fg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_fg)
        numeric_value = self.value if self.value is not None else 0
        span = int(-360 * 16 * (numeric_value / 100))
        painter.drawArc(rect, 90 * 16, span)

        painter.setPen(QColor(TEXT_PRIMARY))
        value_text = f"{self.value}%" if self.value is not None else "N/A"
        painter.drawText(rect, Qt.AlignCenter, value_text)

        painter.setPen(QColor(TEXT_SECONDARY))
        label_rect = self.rect().adjusted(0, rect.height() + 10, 0, 0)
        painter.drawText(label_rect, Qt.AlignHCenter, self.label_text)


class SystemPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        self.gpu_reader = GPUReader()
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("System"))
        header.addStretch()
        layout.addLayout(header)

        rings = QHBoxLayout()
        self.cpu_ring = RingStat("CPU")
        self.ram_ring = RingStat("RAM")
        self.gpu_ring = RingStat("GPU")
        rings.addWidget(self.cpu_ring)
        rings.addWidget(self.ram_ring)
        rings.addWidget(self.gpu_ring)
        layout.addLayout(rings)

        self.gpu_details = QLabel("GPU TEMP —  •  VRAM —")
        self.gpu_details.setObjectName("dim")
        layout.addWidget(self.gpu_details)

        self.disk_label = QLabel("SSD")
        self.disk_label.setObjectName("dim")
        layout.addWidget(self.disk_label)
        self.disk_bar = QProgressBar()
        self.disk_bar.setTextVisible(False)
        self.disk_bar.setFixedHeight(6)
        layout.addWidget(self.disk_bar)

        timer = QTimer(self)
        timer.timeout.connect(self._refresh)
        timer.start(2000)
        self._refresh()

    def _refresh(self):
        self.cpu_ring.set_value(int(psutil.cpu_percent()))
        self.ram_ring.set_value(int(psutil.virtual_memory().percent))
        gpu = self.gpu_reader.stats()
        self.gpu_ring.set_value(gpu["usage"])
        temp = f'{gpu["temp"]} °C' if gpu["temp"] is not None else "N/A"
        if gpu["vram_total_mb"]:
            vram = f'{gpu["vram_used_mb"]/1024:.1f} / {gpu["vram_total_mb"]/1024:.1f} GB'
        else:
            vram = "N/A"
        self.gpu_details.setText(f"GPU TEMP  {temp}   •   VRAM  {vram}")

        disk = psutil.disk_usage("/")
        self.disk_bar.setMaximum(100)
        self.disk_bar.setValue(int(disk.percent))
        used_gb = disk.used / (1024 ** 3)
        total_gb = disk.total / (1024 ** 3)
        self.disk_label.setText(f"SSD  {used_gb:.0f} GB / {total_gb:.0f} GB")


class WeatherPanel(QFrame):
    """Compact right-side weather widget using the same service as chat."""

    def __init__(self, weather: WeatherService):
        super().__init__()
        self.setObjectName("panel")
        self.weather = weather
        self._worker = None

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("Weather"))
        header.addStretch()
        self.refresh_btn = QPushButton("⟳")
        self.refresh_btn.setFixedWidth(28)
        self.refresh_btn.setToolTip("Päivitä sää Open-Meteosta")
        self.refresh_btn.clicked.connect(self.refresh_weather)
        header.addWidget(self.refresh_btn)
        layout.addLayout(header)

        self.location_label = QLabel(self.weather.default_location)
        self.location_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(self.location_label)

        self.temp_label = QLabel("— °C")
        self.temp_label.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {ACCENT};")
        layout.addWidget(self.temp_label)

        self.condition_label = QLabel("Haetaan säätä…")
        self.condition_label.setObjectName("dim")
        self.condition_label.setWordWrap(True)
        layout.addWidget(self.condition_label)

        self.details_label = QLabel("Kosteus —   •   Tuuli —")
        self.details_label.setObjectName("dim")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        self.tomorrow_label = QLabel("Huomenna: —")
        self.tomorrow_label.setObjectName("dim")
        self.tomorrow_label.setWordWrap(True)
        layout.addWidget(self.tomorrow_label)

        # Weather does not need second-by-second polling. Refresh every 15 min.
        timer = QTimer(self)
        timer.timeout.connect(self.refresh_weather)
        timer.start(15 * 60 * 1000)
        self.refresh_weather()

    def refresh_weather(self):
        if self._worker is not None:
            return
        self.refresh_btn.setEnabled(False)
        self._worker = Worker(self.weather.get_weather, self.weather.default_location)
        self._worker.finished.connect(self._on_weather)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_weather(self, w: dict):
        self.refresh_btn.setEnabled(True)
        self._worker = None
        self.location_label.setText(w["location"])
        self.temp_label.setText(f"{w['temperature']:.1f} °C")
        self.condition_label.setText(
            f"{w['description']}  •  tuntuu kuin {w['feels_like']:.1f} °C"
        )
        self.details_label.setText(
            f"Kosteus {w['humidity']:.0f} %   •   Tuuli {w['wind']:.1f} km/h"
        )
        if w["tomorrow_min"] is not None and w["tomorrow_max"] is not None:
            extra = ""
            if w["tomorrow_rain"] is not None:
                extra = f"   •   sade {w['tomorrow_rain']:.0f} %"
            self.tomorrow_label.setText(
                f"Huomenna: {w['tomorrow_description']}, "
                f"{w['tomorrow_min']:.0f}–{w['tomorrow_max']:.0f} °C{extra}"
            )
        else:
            self.tomorrow_label.setText("Huomenna: —")

    def _on_error(self, message: str):
        self.refresh_btn.setEnabled(True)
        self._worker = None
        self.condition_label.setText(f"Säätä ei voitu hakea: {message}")



# ----------------------------------------------------------------------
# Alevik AI v0.3 quality-of-life layer
# ----------------------------------------------------------------------

def load_state():
    try:
        if STATE_FILE.exists():
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
    except Exception:
        pass
    return {}


def save_state(data):
    try:
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log_exception("save_state", exc)


def log_exception(context, exc):
    try:
        from datetime import datetime
        import traceback
        with CRASH_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {context}\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


def human_age(ts):
    try:
        delta = max(0, int(time.time() - float(ts)))
    except Exception:
        return ""
    if delta < 60:
        return "nyt"
    if delta < 3600:
        return f"{delta // 60} min sitten"
    if delta < 86400:
        return f"{delta // 3600} h sitten"
    return f"{delta // 86400} pv sitten"


def build_stylesheet(settings=None):
    settings = settings or load_local_settings()
    theme = settings.get("theme", {})
    intensity = max(30, min(100, int(theme.get("neon_intensity", 85)))) / 100.0
    transparency = max(70, min(100, int(theme.get("panel_transparency", 94)))) / 100.0
    # Cyan stays cyan; intensity changes how bright it appears against the dark UI.
    r = int(10 + (24 - 10) * intensity)
    g = int(95 + (230 - 95) * intensity)
    b = int(120 + (255 - 120) * intensity)
    accent = f"rgb({r},{g},{b})"
    panel_alpha = int(255 * transparency)
    panel = f"rgba(15,28,46,{panel_alpha})"
    accent_dim = f"rgba({r},{g},{b},45)"
    return f"""
    QMainWindow {{ background-color: {DARK_BG}; }}
    QWidget {{ color: {TEXT_PRIMARY}; font-family: 'Segoe UI', sans-serif; }}
    #sidebar {{ background-color: rgba(9,20,33,245); border-right: 1px solid {PANEL_BORDER}; }}
    #navButton {{ text-align:left; padding:10px 14px; border-radius:8px; border:none; background:transparent; color:{TEXT_SECONDARY}; font-size:13px; }}
    #navButton:hover {{ background:{accent_dim}; color:{TEXT_PRIMARY}; }}
    #navButtonActive {{ text-align:left; padding:10px 14px; border-radius:8px; border:none; background:{accent_dim}; color:{accent}; font-size:13px; font-weight:600; }}
    #panel {{ background:{panel}; border:1px solid rgba(39,79,108,210); border-radius:12px; }}
    #chatBubbleUser {{ background:{accent_dim}; border:1px solid rgba({r},{g},{b},70); border-radius:10px; }}
    #chatBubbleAssistant {{ background:{panel}; border:1px solid {PANEL_BORDER}; border-radius:10px; }}
    QLineEdit, QTextEdit, QListWidget, QComboBox {{ background:rgba(8,19,31,225); border:1px solid {PANEL_BORDER}; border-radius:9px; padding:8px 10px; }}
    QPushButton {{ background:rgba(23,51,74,180); border:1px solid rgba(55,98,128,180); border-radius:8px; padding:7px 11px; }}
    QPushButton:hover {{ border-color:{accent}; background:{accent_dim}; }}
    QPushButton#sendButton {{ background:{accent}; border:none; color:{DARK_BG}; font-weight:700; }}
    QLabel#dim {{ color:{TEXT_SECONDARY}; }}
    QLabel#title {{ font-size:20px; font-weight:700; }}
    QProgressBar {{ background:#17334A; border:none; border-radius:3px; }}
    QProgressBar::chunk {{ background:{accent}; border-radius:3px; }}
    QSlider::groove:horizontal {{ height:5px; background:#17334A; border-radius:2px; }}
    QSlider::handle:horizontal {{ width:15px; margin:-5px 0; border-radius:7px; background:{accent}; }}
    QCheckBox {{ spacing:8px; }}
    """


class Worker(QThread):
    """Safe background worker used by network and slow system calls."""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.finished.emit(self.fn(*self.args, **self.kwargs))
        except Exception as exc:
            log_exception(getattr(self.fn, "__name__", "worker"), exc)
            self.error.emit(str(exc))


class NotificationCenter(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        state = load_state()
        self.items = deque(state.get("notifications", []), maxlen=30)
        self._dedupe = {}

    def add(self, text, level="info", dedupe_key=None, cooldown=120):
        now = time.time()
        key = dedupe_key or text
        if key in self._dedupe and now - self._dedupe[key] < cooldown:
            return
        self._dedupe[key] = now
        self.items.appendleft({"text": str(text), "level": level, "time": now})
        self.persist()
        self.changed.emit()

    def clear(self):
        self.items.clear()
        self.persist()
        self.changed.emit()

    def persist(self):
        data = load_state()
        data["notifications"] = list(self.items)
        save_state(data)


class ClipboardManager(QObject):
    changed = Signal()
    SENSITIVE_WORDS = ("password", "passwd", "client_secret", "api_key", "authorization: bearer", "private key")

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.history = deque(load_state().get("clipboard_history", []), maxlen=12)
        app.clipboard().dataChanged.connect(self._capture)

    def enabled(self):
        return bool(load_local_settings().get("privacy", {}).get("clipboard_history", False))

    def _capture(self):
        if not self.enabled():
            return
        text = self.app.clipboard().text().strip()
        if not text or len(text) > 2000:
            return
        low = text.lower()
        # Don't intentionally persist obvious credentials or secret material.
        if any(word in low for word in self.SENSITIVE_WORDS):
            return
        if self.history and self.history[0].get("text") == text:
            return
        self.history.appendleft({"text": text, "time": time.time()})
        self.persist()
        self.changed.emit()

    def clear(self):
        self.history.clear()
        self.persist()
        self.changed.emit()

    def persist(self):
        data = load_state()
        data["clipboard_history"] = list(self.history)
        save_state(data)


class ExtendedGPUReader(GPUReader):
    def top_process(self):
        """Best-effort NVIDIA top GPU process from nvidia-smi pmon."""
        if not self._nvidia_smi:
            return None
        try:
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            r = subprocess.run(
                [self._nvidia_smi, "pmon", "-c", "1", "-s", "u"],
                capture_output=True, text=True, timeout=3, check=False, **kwargs
            )
            best = None
            for line in r.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                try:
                    pid = int(parts[1])
                    sm = int(parts[3]) if parts[3] != "-" else 0
                except Exception:
                    continue
                name = parts[-1]
                try:
                    name = psutil.Process(pid).name()
                except Exception:
                    pass
                if best is None or sm > best[1]:
                    best = (name, sm)
            return best
        except Exception:
            return None


class PersistentRecentAppsTracker:
    FRIENDLY = {
        "code.exe": "VS Code", "spotify.exe": "Spotify", "discord.exe": "Discord",
        "chrome.exe": "Chrome", "msedge.exe": "Edge", "steam.exe": "Steam",
        "explorer.exe": "File Explorer", "pycharm64.exe": "PyCharm",
        "devenv.exe": "Visual Studio", "obs64.exe": "OBS Studio",
        "lm studio.exe": "LM Studio", "lmstudio.exe": "LM Studio",
        "python.exe": "Python", "pythonw.exe": "Python",
    }

    def __init__(self):
        self.entries = load_state().get("recent_apps", {})
        self.recent = deque(maxlen=8)
        self.refresh()

    def _persist(self):
        data = load_state()
        data["recent_apps"] = self.entries
        save_state(data)

    def mark(self, name):
        self.entries[str(name)] = time.time()
        self.refresh_order_only()
        self._persist()

    def refresh_order_only(self):
        ordered = sorted(self.entries.items(), key=lambda kv: kv[1], reverse=True)[:8]
        self.recent = deque((name for name, _ in ordered), maxlen=8)

    def refresh(self):
        changed = False
        now = time.time()
        found = set()
        for proc in psutil.process_iter(["name"]):
            try:
                raw = (proc.info.get("name") or "").lower()
                friendly = self.FRIENDLY.get(raw)
                if friendly:
                    found.add(friendly)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        for name in found:
            old = self.entries.get(name, 0)
            if now - old > 30:
                self.entries[name] = now
                changed = True
        self.refresh_order_only()
        if changed:
            self._persist()
        return list(self.recent)

    def detailed(self):
        self.refresh()
        return [(name, self.entries.get(name, 0)) for name in self.recent]

    def clear(self):
        self.entries.clear()
        self.recent.clear()
        self._persist()


class ProjectManager(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self._ensure_defaults()

    @staticmethod
    def default_projects():
        app_dir = str(Path(__file__).resolve().parent)
        return [
            {"name": "Alevik AI", "path": app_dir, "run": "alevik_ai.py"},
            {"name": "Frisbee Panel", "path": r"C:\Users\daniel\Desktop\other projects\fresbee", "run": "fresbee.py"},
        ]

    def _ensure_defaults(self):
        cfg = load_local_settings()
        projects = cfg.get("projects")

        # Migrate old project entries whenever Alevik starts.
        # Jungle Adventure is retired, and the built-in Alevik/Frisbee entries
        # always use their current canonical paths instead of stale saved paths.
        defaults = {p["name"].lower(): p for p in self.default_projects()}
        cleaned = []
        seen = set()
        if isinstance(projects, list):
            for project in projects:
                if not isinstance(project, dict):
                    continue
                name = str(project.get("name", "")).strip()
                key = name.lower()
                if not name or key == "jungle adventure":
                    continue
                if key in defaults:
                    project = dict(defaults[key])
                if key not in seen:
                    cleaned.append(project)
                    seen.add(key)

        projects = cleaned
        for key, default in defaults.items():
            if key not in seen:
                projects.append(dict(default))
                seen.add(key)

        cfg["projects"] = projects
        save_local_settings(cfg)

    def projects(self):
        return list(load_local_settings().get("projects", []))

    def add(self, name, path):
        cfg = load_local_settings()
        items = cfg.setdefault("projects", [])
        run = ""
        for candidate in ("main.py", "fresbee.py", "alevik_ai.py", "app.py"):
            if (Path(path) / candidate).exists():
                run = candidate
                break
        items.append({"name": name, "path": path, "run": run})
        save_local_settings(cfg)
        self.changed.emit()

    def remove(self, name):
        cfg = load_local_settings()
        cfg["projects"] = [p for p in cfg.get("projects", []) if p.get("name") != name]
        save_local_settings(cfg)
        self.changed.emit()

    def find(self, name):
        for p in self.projects():
            if p.get("name", "").lower() == name.lower():
                return p
        return None

    def open_folder(self, project):
        path = project.get("path", "")
        if not path or not Path(path).exists():
            raise FileNotFoundError(path or "Project path missing")
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def open_vscode(self, project):
        path = project.get("path", "")
        if not Path(path).exists():
            raise FileNotFoundError(path)
        code = shutil.which("code") or shutil.which("code.cmd")
        if not code:
            raise RuntimeError("VS Code 'code' command not found in PATH")
        subprocess.Popen([code, path])

    def run_project(self, project):
        path = Path(project.get("path", ""))
        target = project.get("run") or ""
        if not target:
            for candidate in ("main.py", "fresbee.py", "alevik_ai.py", "app.py"):
                if (path / candidate).exists():
                    target = candidate
                    break
        if not target or not (path / target).exists():
            raise FileNotFoundError("Run file not found")
        subprocess.Popen([sys.executable, str(path / target)], cwd=str(path))


class RecentFilesScanner:
    ALLOWED = {".py", ".json", ".csv", ".md", ".txt", ".toml", ".yaml", ".yml", ".ui"}

    def __init__(self, project_manager):
        self.project_manager = project_manager

    def scan(self, limit=18):
        files = []
        for project in self.project_manager.projects():
            root = Path(project.get("path", ""))
            if not root.exists() or not root.is_dir():
                continue
            root_depth = len(root.parts)
            for cur, dirs, names in os.walk(root):
                cur_path = Path(cur)
                if len(cur_path.parts) - root_depth >= 3:
                    dirs[:] = []
                dirs[:] = [d for d in dirs if d not in {".git", ".venv", "venv", "__pycache__", "node_modules", ".idea"}]
                for name in names:
                    p = cur_path / name
                    if p.suffix.lower() not in self.ALLOWED:
                        continue
                    try:
                        files.append((p.stat().st_mtime, project.get("name", "Project"), str(p)))
                    except OSError:
                        pass
        files.sort(reverse=True)
        return files[:limit]


class AudioService:
    def __init__(self):
        self.endpoint = None
        self.audio_device = None
        self.device_name = "Audio unavailable"
        self.error = None
        self.reload()

    def reload(self):
        self.endpoint = None
        self.audio_device = None
        if os.name != "nt" or AudioUtilities is None:
            self.error = "Asenna pycaw + comtypes Windowsissa"
            return
        try:
            device = AudioUtilities.GetSpeakers()
            self.audio_device = device

            # Newer pycaw exposes EndpointVolume directly. Older versions
            # require activating IAudioEndpointVolume manually.
            endpoint = getattr(device, "EndpointVolume", None)
            if endpoint is None and hasattr(device, "Activate"):
                interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = cast(interface, POINTER(IAudioEndpointVolume))
            if endpoint is None:
                raise RuntimeError("Windows audio endpoint volume interface not found")
            self.endpoint = endpoint

            name = getattr(device, "FriendlyName", None)
            if not name and hasattr(device, "GetId"):
                try:
                    device_id = device.GetId()
                    for candidate in AudioUtilities.GetAllDevices():
                        if getattr(candidate, "id", None) == device_id:
                            name = getattr(candidate, "FriendlyName", None)
                            break
                except Exception:
                    pass
            self.device_name = name or "Default output"
            self.error = None
        except Exception as exc:
            self.error = str(exc)

    def status(self):
        if self.endpoint is None:
            # The default output device may have changed after startup.
            self.reload()
        if self.endpoint is None:
            return {"available": False, "name": self.device_name, "error": self.error}
        try:
            scalar = float(self.endpoint.GetMasterVolumeLevelScalar())
            return {
                "available": True,
                "name": self.device_name,
                "volume": max(0, min(100, int(round(scalar * 100)))),
                "muted": bool(self.endpoint.GetMute()),
            }
        except Exception as exc:
            self.error = str(exc)
            self.endpoint = None
            return {"available": False, "name": self.device_name, "error": self.error}

    def set_volume(self, value):
        if self.endpoint is None:
            self.reload()
        if self.endpoint is None:
            raise RuntimeError(self.error or "Audio unavailable")
        self.endpoint.SetMasterVolumeLevelScalar(max(0, min(100, int(value))) / 100.0, None)

    def toggle_mute(self):
        if self.endpoint is None:
            self.reload()
        if self.endpoint is None:
            raise RuntimeError(self.error or "Audio unavailable")
        self.endpoint.SetMute(not bool(self.endpoint.GetMute()), None)


class BackgroundWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.blur = int(load_local_settings().get("theme", {}).get("background_blur", 2))
        self._pix = QPixmap()
        path = Path(__file__).resolve().parent / "assets" / "alevik_background.png"
        if path.exists():
            self._pix = QPixmap(str(path))

    def set_blur(self, value):
        self.blur = max(0, min(12, int(value)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        if not self._pix.isNull():
            scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            if self.blur > 0:
                factor = max(2, 1 + self.blur)
                tiny = scaled.scaled(max(1, self.width() // factor), max(1, self.height() // factor), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                scaled = tiny.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            p.drawPixmap(0, 0, scaled, x, y, self.width(), self.height())
        else:
            p.fillRect(self.rect(), QColor(DARK_BG))
        p.fillRect(self.rect(), QColor(3, 10, 18, 125))
        super().paintEvent(event)


class AdvancedSystemPanel(QFrame):
    def __init__(self, notifier=None, compact=True):
        super().__init__()
        self.setObjectName("panel")
        self.notifier = notifier
        self.compact = compact
        self.gpu_reader = ExtendedGPUReader()
        self.net_prev = psutil.net_io_counters()
        self.net_time = time.time()
        self.proc_cache = {}
        self._gpu_hot = False

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("System Health")
        title.setStyleSheet("font-weight:700;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        rings = QHBoxLayout()
        self.cpu_ring = RingStat("CPU")
        self.ram_ring = RingStat("RAM")
        self.gpu_ring = RingStat("GPU")
        rings.addWidget(self.cpu_ring); rings.addWidget(self.ram_ring); rings.addWidget(self.gpu_ring)
        layout.addLayout(rings)

        self.gpu_details = QLabel("GPU TEMP —  •  VRAM —")
        self.cpu_temp = QLabel("CPU TEMP —")
        self.disk_label = QLabel("SSD —")
        self.net_label = QLabel("NET ↓ —  ↑ —")
        self.top_cpu = QLabel("Top CPU: —")
        self.top_gpu = QLabel("Top GPU: —")
        for lbl in (self.gpu_details, self.cpu_temp, self.disk_label, self.net_label, self.top_cpu, self.top_gpu):
            lbl.setObjectName("dim")
            lbl.setWordWrap(True)
            layout.addWidget(lbl)
        if compact:
            self.top_cpu.setVisible(False)
            self.top_gpu.setVisible(False)
            self.cpu_temp.setVisible(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2500)
        self.refresh()

    @staticmethod
    def _cpu_temperature():
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False)
            if not temps:
                return None
            vals = []
            for entries in temps.values():
                for entry in entries:
                    if entry.current and 0 < entry.current < 120:
                        vals.append(entry.current)
            return round(max(vals), 1) if vals else None
        except Exception:
            return None

    def _top_cpu_process(self):
        best = ("—", 0.0)
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                val = proc.cpu_percent(None)
                if val > best[1]:
                    best = (proc.info.get("name") or str(proc.info.get("pid")), val)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return best

    def refresh(self):
        cpu = int(psutil.cpu_percent())
        ram = int(psutil.virtual_memory().percent)
        gpu = self.gpu_reader.stats()
        self.cpu_ring.set_value(cpu); self.ram_ring.set_value(ram); self.gpu_ring.set_value(gpu.get("usage"))

        temp = gpu.get("temp")
        vram = "N/A"
        if gpu.get("vram_total_mb"):
            vram = f"{gpu['vram_used_mb']/1024:.1f}/{gpu['vram_total_mb']/1024:.1f} GB"
        self.gpu_details.setText(f"GPU TEMP {temp if temp is not None else 'N/A'} °C  •  VRAM {vram}")
        ctemp = self._cpu_temperature()
        self.cpu_temp.setText(f"CPU TEMP {ctemp:.1f} °C" if ctemp is not None else "CPU TEMP N/A")

        root = Path.home().anchor or "/"
        disk = psutil.disk_usage(root)
        self.disk_label.setText(f"SSD free {disk.free/(1024**3):.0f} GB  •  used {disk.percent:.0f}%")

        now = time.time(); cur = psutil.net_io_counters(); dt = max(.1, now - self.net_time)
        down = max(0, cur.bytes_recv - self.net_prev.bytes_recv) / dt
        up = max(0, cur.bytes_sent - self.net_prev.bytes_sent) / dt
        self.net_prev, self.net_time = cur, now
        self.net_label.setText(f"NET ↓ {down/1024/1024:.2f} MB/s  ↑ {up/1024/1024:.2f} MB/s")

        if not self.compact:
            name, val = self._top_cpu_process()
            self.top_cpu.setText(f"Top CPU: {name}  {val:.0f}%")
            gp = self.gpu_reader.top_process()
            self.top_gpu.setText(f"Top GPU: {gp[0]}  {gp[1]}%" if gp else "Top GPU: N/A")

        if temp is not None and temp >= 83 and not self._gpu_hot:
            self._gpu_hot = True
            if self.notifier:
                self.notifier.add(f"GPU lämpötila on {temp} °C", "warning", "gpu_hot", 300)
        elif temp is not None and temp < 78:
            self._gpu_hot = False


class NotifyingWeatherPanel(WeatherPanel):
    def __init__(self, weather, notifier=None):
        self.notifier = notifier
        super().__init__(weather)

    def _on_weather(self, w):
        super()._on_weather(w)
        code = w.get("weather_code")
        if self.notifier and code in {95, 96, 99}:
            self.notifier.add(f"Säävaroitus: {w.get('description')} ({w.get('location')})", "warning", "weather_storm", 900)
        if self.notifier and (w.get("tomorrow_rain") or 0) >= 80:
            self.notifier.add(f"Huomiselle korkea sateen mahdollisuus: {w.get('tomorrow_rain'):.0f}%", "info", "weather_rain", 900)


class NotifyingSpotifyPanel(SpotifyPanel):
    def __init__(self, service, notifier=None):
        self.notifier = notifier
        self._last_connected = None
        super().__init__(service)

    def done(self, d):
        super().done(d)
        connected = bool(d.get("connected"))
        if self._last_connected is not None and connected != self._last_connected and self.notifier:
            self.notifier.add("Spotify yhdistetty" if connected else "Spotify-yhteys katkesi", "info", "spotify_status", 30)
        self._last_connected = connected


class AudioPanel(QFrame):
    def __init__(self, service):
        super().__init__(); self.setObjectName("panel"); self.service = service; self._updating = False
        l = QVBoxLayout(self)
        h = QHBoxLayout(); h.addWidget(QLabel("Audio")); h.addStretch(); self.device = QLabel("—"); self.device.setObjectName("dim"); h.addWidget(self.device); l.addLayout(h)
        row = QHBoxLayout()
        self.mute = QPushButton("Mute")
        self.mute.clicked.connect(self._toggle)
        row.addWidget(self.mute)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setTracking(True)  # muuta Windows-volumea jo raahatessa
        self.slider.valueChanged.connect(self._preview_volume)
        self.slider.sliderMoved.connect(self._volume)
        self.slider.sliderReleased.connect(lambda: self._volume(self.slider.value()))
        row.addWidget(self.slider, 1)
        self.value = QLabel("—"); self.value.setFixedWidth(38); row.addWidget(self.value)
        l.addLayout(row)
        self.timer = QTimer(self); self.timer.setTimerType(Qt.PreciseTimer); self.timer.timeout.connect(self.refresh); self.timer.start(10); self.refresh()

    def refresh(self):
        d = self.service.status(); self._updating = True
        self.device.setText(d.get("name", "—")[:24])
        if d.get("available"):
            vol = int(d.get("volume", 0))
            self.slider.setEnabled(True)
            self.mute.setEnabled(True)
            # Älä taistele käyttäjän hiiren kanssa raahauksen aikana. Muulloin
            # slider seuraa Windowsin master-volumea 100 kertaa sekunnissa, joten
            # myös Stream Deckistä / näppäimistöstä tehdyt muutokset näkyvät heti.
            if not self.slider.isSliderDown():
                if self.slider.value() != vol:
                    self.slider.setValue(vol)
                self.value.setText(f"{vol}%")
            self.mute.setText("Unmute" if d.get("muted") else "Mute")
        else:
            self.slider.setEnabled(False); self.mute.setEnabled(False); self.value.setText("N/A"); self.device.setToolTip(d.get("error", ""))
        self._updating = False

    def _preview_volume(self, value):
        # Prosentti seuraa kahvaa välittömästi myös raahatessa.
        if not self._updating:
            self.value.setText(f"{int(value)}%")

    def _volume(self, value):
        if self._updating:
            return
        value = max(0, min(100, int(value)))
        self.value.setText(f"{value}%")
        try:
            self.service.set_volume(value)
        except Exception as exc:
            log_exception("audio_set_volume", exc)
            # Yritä hakea Windowsin oletusäänilaite uudelleen, jos se vaihtui.
            try:
                self.service.reload()
                self.service.set_volume(value)
            except Exception as retry_exc:
                log_exception("audio_set_volume_retry", retry_exc)

    def _toggle(self):
        try: self.service.toggle_mute(); self.refresh()
        except Exception as exc: log_exception("audio_toggle_mute", exc)


class NotificationsPanel(QFrame):
    def __init__(self, center):
        super().__init__(); self.setObjectName("panel"); self.center=center
        self.l=QVBoxLayout(self); h=QHBoxLayout(); h.addWidget(QLabel("Notifications")); h.addStretch(); clear=QPushButton("Clear"); clear.clicked.connect(center.clear); h.addWidget(clear); self.l.addLayout(h)
        self.labels=[]
        for _ in range(4):
            lab=QLabel("—"); lab.setObjectName("dim"); lab.setWordWrap(True); self.l.addWidget(lab); self.labels.append(lab)
        center.changed.connect(self.refresh); self.refresh()
    def refresh(self):
        items=list(self.center.items)
        for i, lab in enumerate(self.labels):
            if i < len(items): lab.setText(f"• {items[i]['text']}  ·  {human_age(items[i]['time'])}")
            else: lab.setText("—")


class AdvancedRecentAppsPanel(QFrame):
    def __init__(self, tracker):
        super().__init__(); self.setObjectName("panel"); self.tracker=tracker
        l=QVBoxLayout(self); h=QHBoxLayout(); h.addWidget(QLabel("Recent Apps")); h.addStretch(); clear=QPushButton("Clear"); clear.clicked.connect(self._clear); h.addWidget(clear); l.addLayout(h)
        self.labels=[]
        for _ in range(5):
            lab=QLabel("—"); lab.setObjectName("dim"); l.addWidget(lab); self.labels.append(lab)
        t=QTimer(self); t.timeout.connect(self.refresh); t.start(5000); self.refresh()
    def _clear(self): self.tracker.clear(); self.refresh()
    def refresh(self):
        rows=self.tracker.detailed()
        for i,lab in enumerate(self.labels): lab.setText(f"• {rows[i][0]}  ·  {human_age(rows[i][1])}" if i<len(rows) else "—")


class RightPanel(QWidget):
    def __init__(self, weather, spotify, recent_apps, audio, notifier):
        super().__init__(); self.setFixedWidth(330)
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body=QWidget(); l=QVBoxLayout(body); l.setContentsMargins(0,0,8,0); l.setSpacing(12)
        self.system_panel=AdvancedSystemPanel(notifier, compact=True); l.addWidget(self.system_panel)
        self.weather_panel=NotifyingWeatherPanel(weather, notifier); l.addWidget(self.weather_panel)
        self.spotify_panel=NotifyingSpotifyPanel(spotify, notifier); l.addWidget(self.spotify_panel)
        self.audio_panel=AudioPanel(audio); l.addWidget(self.audio_panel)
        self.notifications_panel=NotificationsPanel(notifier); l.addWidget(self.notifications_panel)
        self.recent_panel=AdvancedRecentAppsPanel(recent_apps); l.addWidget(self.recent_panel)
        l.addStretch(); scroll.setWidget(body); root.addWidget(scroll)


class PageHeader(QWidget):
    def __init__(self, title, subtitle=""):
        super().__init__(); l=QVBoxLayout(self); l.setContentsMargins(0,0,0,8); l.setSpacing(2)
        t=QLabel(title); t.setStyleSheet("font-size:24px;font-weight:700;"); l.addWidget(t)
        if subtitle:
            s=QLabel(subtitle); s.setObjectName("dim"); s.setWordWrap(True); l.addWidget(s)


class AppsPage(QWidget):
    def __init__(self, main_window):
        super().__init__(); self.main=main_window
        l=QVBoxLayout(self); l.addWidget(PageHeader("Apps & Quick Actions", "Yleisimmät sovellukset ja oikea last-used-historia."))
        box=QFrame(); box.setObjectName("panel"); grid=QGridLayout(box)
        actions=[
            ("VS Code", self.main.open_vscode), ("Spotify", self.main.open_spotify),
            ("Frisbee Panel", lambda:self.main.open_project("Frisbee Panel")), ("Downloads", self.main.open_downloads),
            ("Command Palette  Ctrl+K", self.main.open_command_palette), ("Settings", lambda:self.main.navigate("Settings")),
        ]
        for i,(name,fn) in enumerate(actions):
            b=QPushButton(name); b.setMinimumHeight(48); b.clicked.connect(fn); grid.addWidget(b,i//2,i%2)
        l.addWidget(box)
        self.list=QListWidget(); l.addWidget(self.list,1)
        self.timer=QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(5000); self.refresh()
    def refresh(self):
        self.list.clear()
        for name,ts in self.main.recent_apps.detailed(): self.list.addItem(f"{name}    {human_age(ts)}")


class FilesPage(QWidget):
    def __init__(self, scanner, notifier):
        super().__init__(); self.scanner=scanner; self.notifier=notifier
        l=QVBoxLayout(self); h=QHBoxLayout(); h.addWidget(PageHeader("Recent Files", "Viimeksi muokatut tiedostot Alevikin projektikansioista."),1); refresh=QPushButton("Refresh"); refresh.clicked.connect(self.refresh); h.addWidget(refresh); l.addLayout(h)
        self.list=QListWidget(); self.list.itemDoubleClicked.connect(self.open_item); l.addWidget(self.list,1); self.rows=[]; self.refresh()
    def refresh(self):
        try: self.rows=self.scanner.scan(); self.list.clear()
        except Exception as exc: log_exception("recent_files",exc); self.rows=[]; self.list.clear()
        for ts,project,path in self.rows:
            self.list.addItem(f"{Path(path).name}   ·   {project}   ·   {human_age(ts)}\n{path}")
    def open_item(self,item):
        idx=self.list.row(item)
        if not (0<=idx<len(self.rows)): return
        path=self.rows[idx][2]
        try:
            if os.name=="nt": os.startfile(path)
            elif sys.platform=="darwin": subprocess.Popen(["open",path])
            else: subprocess.Popen(["xdg-open",path])
        except Exception as exc: self.notifier.add(f"Tiedostoa ei voitu avata: {exc}","error")


class ProjectCard(QFrame):
    def __init__(self, project, manager, notifier, on_change):
        super().__init__(); self.setObjectName("panel"); self.project=project; self.manager=manager; self.notifier=notifier; self.on_change=on_change
        l=QVBoxLayout(self); name=QLabel(project.get("name","Project")); name.setStyleSheet("font-size:17px;font-weight:700;"); l.addWidget(name)
        path=project.get("path",""); pl=QLabel(path); pl.setObjectName("dim"); pl.setWordWrap(True); l.addWidget(pl)
        status=QLabel("● Found" if Path(path).exists() else "● Path not found"); status.setStyleSheet("color:#1ED760;" if Path(path).exists() else "color:#ffb454;"); l.addWidget(status)
        row=QHBoxLayout()
        for title,fn in (("Open folder",manager.open_folder),("Open VS Code",manager.open_vscode),("Run",manager.run_project)):
            b=QPushButton(title); b.clicked.connect(lambda _=False,f=fn:self._do(f)); row.addWidget(b)
        remove=QPushButton("Remove"); remove.clicked.connect(self._remove); row.addWidget(remove); l.addLayout(row)
    def _do(self,fn):
        try: fn(self.project)
        except Exception as exc: self.notifier.add(f"{self.project.get('name')}: {exc}","error")
    def _remove(self): self.manager.remove(self.project.get("name","")); self.on_change()


class ProjectsPage(QWidget):
    def __init__(self, manager, notifier):
        super().__init__(); self.manager=manager; self.notifier=notifier
        outer=QVBoxLayout(self); h=QHBoxLayout(); h.addWidget(PageHeader("Projects", "Avaa kansio, VS Code tai käynnistä projekti yhdestä paikasta."),1); add=QPushButton("+ Add project"); add.clicked.connect(self.add_project); h.addWidget(add); outer.addLayout(h)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame); self.body=QWidget(); self.cards=QVBoxLayout(self.body); self.cards.addStretch(); scroll.setWidget(self.body); outer.addWidget(scroll,1)
        manager.changed.connect(self.refresh); self.refresh()
    def refresh(self):
        while self.cards.count()>1:
            item=self.cards.takeAt(0); w=item.widget()
            if w: w.deleteLater()
        for p in self.manager.projects(): self.cards.insertWidget(self.cards.count()-1,ProjectCard(p,self.manager,self.notifier,self.refresh))
    def add_project(self):
        path=QFileDialog.getExistingDirectory(self,"Valitse projektikansio")
        if not path:return
        name,ok=QInputDialog.getText(self,"Projektin nimi","Nimi:",text=Path(path).name)
        if ok and name.strip(): self.manager.add(name.strip(),path)


class SystemPage(QWidget):
    def __init__(self, notifier):
        super().__init__(); l=QVBoxLayout(self); l.addWidget(PageHeader("System", "Reaaliaikainen CPU, RAM, GPU, lämpötilat, VRAM, SSD, verkko ja raskaimmat prosessit.")); l.addWidget(AdvancedSystemPanel(notifier, compact=False));
        crash=QFrame(); crash.setObjectName("panel"); cl=QVBoxLayout(crash); cl.addWidget(QLabel("Crash log")); self.crash_label=QLabel(str(CRASH_LOG_FILE)); self.crash_label.setObjectName("dim"); cl.addWidget(self.crash_label); openb=QPushButton("Open crash log"); openb.clicked.connect(self.open_crash); cl.addWidget(openb); l.addWidget(crash); l.addStretch()
    def open_crash(self):
        if not CRASH_LOG_FILE.exists(): CRASH_LOG_FILE.write_text("Alevik AI crash log\n",encoding="utf-8")
        try:
            if os.name=="nt": os.startfile(str(CRASH_LOG_FILE))
        except Exception: pass


class NotesPage(QWidget):
    def __init__(self):
        super().__init__(); l=QVBoxLayout(self); l.addWidget(PageHeader("Notes", "Paikalliset muistiinpanot tallentuvat Alevikin asetuksiin.")); self.edit=QTextEdit(); self.edit.setPlaceholderText("Kirjoita muistiinpanoja…"); self.edit.setPlainText(load_local_settings().get("notes","")); l.addWidget(self.edit,1); save=QPushButton("Save notes"); save.clicked.connect(self.save); l.addWidget(save)
    def save(self):
        cfg=load_local_settings(); cfg["notes"]=self.edit.toPlainText(); save_local_settings(cfg)


class MemoryPage(QWidget):
    def __init__(self, clipboard):
        super().__init__(); self.clipboard=clipboard; l=QVBoxLayout(self); h=QHBoxLayout(); h.addWidget(PageHeader("Memory / Clipboard", "Clipboard history on paikallinen ja jättää ilmeiset salaisuudet tallentamatta."),1); clear=QPushButton("Clear clipboard history"); clear.clicked.connect(clipboard.clear); h.addWidget(clear); l.addLayout(h); self.list=QListWidget(); self.list.itemDoubleClicked.connect(self.copy_item); l.addWidget(self.list,1); clipboard.changed.connect(self.refresh); self.refresh()
    def refresh(self):
        self.list.clear()
        for item in self.clipboard.history:
            text=item.get("text","").replace("\n"," ↵ "); self.list.addItem(f"{text[:180]}    ·   {human_age(item.get('time',0))}")
    def copy_item(self,item):
        idx=self.list.row(item)
        if 0<=idx<len(self.clipboard.history): QApplication.clipboard().setText(list(self.clipboard.history)[idx].get("text",""))


class SettingsPage(QWidget):
    settings_saved = Signal()
    def __init__(self, main):
        super().__init__(); self.main=main
        outer=QVBoxLayout(self); outer.addWidget(PageHeader("Settings", f"Alevik AI v{APP_VERSION} · integraatiot, ulkoasu, startup ja backup."))
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame); body=QWidget(); self.l=QVBoxLayout(body); scroll.setWidget(body); outer.addWidget(scroll,1)
        cfg=load_local_settings(); integ=cfg.get("integrations",{}); sp=cfg.get("spotify",{}); weather=cfg.get("weather",{}); theme=cfg.get("theme",{}); privacy=cfg.get("privacy",{})

        integrations=QGroupBox("Integrations"); f=QFormLayout(integrations)
        self.claude=QLineEdit(integ.get("claude_api_key",os.environ.get("ANTHROPIC_API_KEY",""))); self.claude.setEchoMode(QLineEdit.Password); self.claude.setPlaceholderText("Lisää Claude API key tähän")
        self.lm=QLineEdit(integ.get("lm_studio_url","http://localhost:1234/v1")); self.lm.setPlaceholderText("http://localhost:1234/v1")
        self.weather=QLineEdit(weather.get("location",DEFAULT_WEATHER_LOCATION)); self.weather.setPlaceholderText("Sääkaupunki")
        self.spotify_id=QLineEdit(sp.get("client_id","")); self.spotify_id.setPlaceholderText("Lisää Spotify Client ID tähän")
        self.spotify_secret=QLineEdit(sp.get("client_secret","")); self.spotify_secret.setEchoMode(QLineEdit.Password); self.spotify_secret.setPlaceholderText("Lisää Spotify Client Secret tähän")
        self.spotify_redirect=QLineEdit(sp.get("redirect_uri","http://127.0.0.1:8888/callback"))
        f.addRow("Claude API Key",self.claude); f.addRow("LM Studio URL",self.lm); f.addRow("Open-Meteo location",self.weather); f.addRow("Spotify Client ID",self.spotify_id); f.addRow("Spotify Client Secret",self.spotify_secret); f.addRow("Spotify Redirect URI",self.spotify_redirect)
        self.l.addWidget(integrations)

        appearance=QGroupBox("Appearance"); af=QFormLayout(appearance)
        self.neon=QSlider(Qt.Horizontal); self.neon.setRange(30,100); self.neon.setValue(int(theme.get("neon_intensity",85)))
        self.transparency=QSlider(Qt.Horizontal); self.transparency.setRange(70,100); self.transparency.setValue(int(theme.get("panel_transparency",94)))
        self.blur=QSlider(Qt.Horizontal); self.blur.setRange(0,12); self.blur.setValue(int(theme.get("background_blur",2)))
        self.layout_mode=QComboBox(); self.layout_mode.addItems(["Full","Compact"]); self.layout_mode.setCurrentText(theme.get("layout","Full"))
        af.addRow("Neon cyan intensity",self.neon); af.addRow("Panel opacity",self.transparency); af.addRow("Background blur",self.blur); af.addRow("Layout",self.layout_mode); self.l.addWidget(appearance)

        behavior=QGroupBox("Behavior & Privacy"); bf=QFormLayout(behavior)
        self.startup=QCheckBox("Start Alevik AI with Windows"); self.startup.setChecked(self.main.startup_enabled())
        self.clipboard=QCheckBox("Enable local clipboard history"); self.clipboard.setChecked(bool(privacy.get("clipboard_history",False)))
        bf.addRow(self.startup); bf.addRow(self.clipboard); self.l.addWidget(behavior)

        actions=QHBoxLayout(); save=QPushButton("Save settings"); save.clicked.connect(self.save); backup=QPushButton("Backup"); backup.clicked.connect(self.backup); restore=QPushButton("Restore"); restore.clicked.connect(self.restore); actions.addWidget(save); actions.addWidget(backup); actions.addWidget(restore); actions.addStretch(); self.l.addLayout(actions)

        update=QGroupBox("Update info / Changelog"); ul=QVBoxLayout(update); ul.addWidget(QLabel(f"Alevik AI v{APP_VERSION}")); ch=QLabel("• System tray + Ctrl+Alt+A show/hide\n• Quick Actions + Ctrl+K Command Palette\n• Projects + Recent Files + persistent Recent Apps\n• Optional local Clipboard History\n• System Health + Audio + Weather + Notifications\n• Startup, themes, backup/restore, crash log\n• Spotify premium controls + shuffle/repeat"); ch.setObjectName("dim"); ch.setWordWrap(True); ul.addWidget(ch); self.l.addWidget(update); self.l.addStretch()

    def save(self):
        cfg=load_local_settings(); cfg["integrations"]={"claude_api_key":self.claude.text().strip(),"lm_studio_url":self.lm.text().strip() or "http://localhost:1234/v1"}; cfg["weather"]={"location":self.weather.text().strip() or DEFAULT_WEATHER_LOCATION}; cfg["spotify"]={"client_id":self.spotify_id.text().strip(),"client_secret":self.spotify_secret.text(),"redirect_uri":self.spotify_redirect.text().strip() or "http://127.0.0.1:8888/callback"}; cfg["theme"]={"neon_intensity":self.neon.value(),"panel_transparency":self.transparency.value(),"background_blur":self.blur.value(),"layout":self.layout_mode.currentText()}; cfg["privacy"]={"clipboard_history":self.clipboard.isChecked()}; save_local_settings(cfg)
        try:self.main.set_startup_enabled(self.startup.isChecked())
        except Exception as exc:self.main.notifications.add(f"Startup-asetusta ei voitu vaihtaa: {exc}","error")
        self.main.apply_settings(); self.main.notifications.add("Asetukset tallennettu","success","settings_saved",2); QMessageBox.information(self,"Alevik AI","Asetukset tallennettu.")

    def backup(self):
        path,_=QFileDialog.getSaveFileName(self,"Backup Alevik AI settings",str(Path.home()/"alevik_ai_backup.json"),"JSON (*.json)")
        if not path:return
        payload={"version":APP_VERSION,"settings":load_local_settings(),"state":load_state()}; Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); QMessageBox.information(self,"Backup","Backup created. Huom: integraatioasetukset voivat sisältää salaisuuksia.")

    def restore(self):
        path,_=QFileDialog.getOpenFileName(self,"Restore Alevik AI settings",str(Path.home()),"JSON (*.json)")
        if not path:return
        try:
            payload=json.loads(Path(path).read_text(encoding="utf-8")); settings=payload.get("settings",payload); save_local_settings(settings)
            if isinstance(payload.get("state"),dict): save_state(payload["state"])
            QMessageBox.information(self,"Restore","Settings restored. Käynnistä Alevik AI uudelleen, jotta kaikki osat latautuvat uudestaan.")
        except Exception as exc: QMessageBox.critical(self,"Restore",str(exc))


class CommandPalette(QDialog):
    def __init__(self, parent, actions):
        super().__init__(parent); self.setWindowTitle("Command Palette"); self.resize(620,420); self.actions=actions
        l=QVBoxLayout(self); self.search=QLineEdit(); self.search.setPlaceholderText("Search apps, projects, settings or commands…"); l.addWidget(self.search); self.list=QListWidget(); l.addWidget(self.list,1); self.search.textChanged.connect(self.refresh); self.search.returnPressed.connect(self.run_current); self.list.itemDoubleClicked.connect(lambda _:self.run_current()); self.refresh(); self.search.setFocus()
    def refresh(self):
        q=self.search.text().lower().strip(); self.list.clear(); self.filtered=[]
        for name,fn in self.actions:
            if not q or q in name.lower(): self.filtered.append((name,fn)); self.list.addItem(name)
        if self.list.count():self.list.setCurrentRow(0)
    def run_current(self):
        row=self.list.currentRow()
        if 0<=row<len(self.filtered):
            _,fn=self.filtered[row]; self.accept(); QTimer.singleShot(0,fn)


class HotkeyBridge(QObject):
    toggle_requested = Signal()


class AlevikMainWindow(QMainWindow):
    PAGE_ORDER=["Chat","Apps","Files","Projects","System","Notes","Memory","Settings"]
    def __init__(self, app):
        super().__init__(); self.app=app; self._force_quit=False; self._hotkey_id=None
        self.setWindowTitle(f"Alevik AI v{APP_VERSION}"); self.resize(1480,920); self.setMinimumSize(1080,700)

        self.notifications=NotificationCenter(); self.project_manager=ProjectManager(); self.recent_apps=PersistentRecentAppsTracker(); self.file_scanner=RecentFilesScanner(self.project_manager); self.audio_service=AudioService(); self.clipboard=ClipboardManager(app)
        self.claude_client=ClaudeClient(); self.local_ai_client=LocalAIClient(); self.weather_service=WeatherService(); self.spotify_service=SpotifyService()

        central=BackgroundWidget(); self.background=central; self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        self.sidebar=Sidebar(); root.addWidget(self.sidebar)
        middle=QWidget(); ml=QVBoxLayout(middle); ml.setContentsMargins(20,16,20,16); ml.setSpacing(10)
        self.stack=QStackedWidget(); ml.addWidget(self.stack,1); root.addWidget(middle,1)
        self.right_panel=RightPanel(self.weather_service,self.spotify_service,self.recent_apps,self.audio_service,self.notifications); root.addWidget(self.right_panel)

        # Chat page keeps the familiar top AI selector.
        chat=QWidget(); cl=QVBoxLayout(chat); cl.setContentsMargins(0,0,0,0); self.top_bar=TopBar(self.local_ai_client); cl.addWidget(self.top_bar); self.chat_panel=ChatPanel(self.claude_client,self.local_ai_client,self.weather_service); cl.addWidget(self.chat_panel,1); self.stack.addWidget(chat)
        self.apps_page=AppsPage(self); self.stack.addWidget(self.apps_page)
        self.files_page=FilesPage(self.file_scanner,self.notifications); self.stack.addWidget(self.files_page)
        self.projects_page=ProjectsPage(self.project_manager,self.notifications); self.stack.addWidget(self.projects_page)
        self.system_page=SystemPage(self.notifications); self.stack.addWidget(self.system_page)
        self.notes_page=NotesPage(); self.stack.addWidget(self.notes_page)
        self.memory_page=MemoryPage(self.clipboard); self.stack.addWidget(self.memory_page)
        self.settings_page=SettingsPage(self); self.stack.addWidget(self.settings_page)

        self.top_bar.mode_changed.connect(self.chat_panel.set_mode); self.top_bar.local_model_changed.connect(self.chat_panel.set_local_model)
        self.sidebar.nav_changed.connect(self.navigate)

        self._setup_tray(); self._setup_hotkeys(); self.apply_settings()
        self.palette_shortcut=QShortcut(QKeySequence("Ctrl+K"),self); self.palette_shortcut.activated.connect(self.open_command_palette)
        self.notifications.add(f"Alevik AI v{APP_VERSION} käynnissä","success","app_started",10)

    # ------------------------- navigation/actions -------------------------
    def navigate(self,name):
        if name in self.PAGE_ORDER:
            self.stack.setCurrentIndex(self.PAGE_ORDER.index(name))
            for key, btn in self.sidebar.buttons.items():
                btn.setObjectName("navButtonActive" if key == name else "navButton")
                btn.setStyle(btn.style())
            if name=="Files": self.files_page.refresh()
            if name=="Apps": self.apps_page.refresh()
            if name=="Memory": self.memory_page.refresh()

    def open_vscode(self):
        code=shutil.which("code") or shutil.which("code.cmd")
        try:
            if code: subprocess.Popen([code]); self.recent_apps.mark("VS Code")
            else: raise RuntimeError("VS Code 'code' command not found")
        except Exception as exc:self.notifications.add(str(exc),"error")
    def open_spotify(self):
        try:
            if os.name=="nt": os.startfile("spotify:")
            else: subprocess.Popen(["spotify"])
            self.recent_apps.mark("Spotify")
        except Exception as exc:self.notifications.add(f"Spotify: {exc}","error")
    def open_downloads(self):
        path=Path.home()/"Downloads"
        try:
            if os.name=="nt": os.startfile(str(path))
            elif sys.platform=="darwin":subprocess.Popen(["open",str(path)])
            else:subprocess.Popen(["xdg-open",str(path)])
            self.recent_apps.mark("Downloads")
        except Exception as exc:self.notifications.add(f"Downloads: {exc}","error")
    def open_project(self,name):
        p=self.project_manager.find(name)
        if not p:self.notifications.add(f"Projektia {name} ei löydy","warning");return
        try:self.project_manager.open_folder(p);self.recent_apps.mark(name)
        except Exception as exc:self.notifications.add(f"{name}: {exc}","error")

    def palette_actions(self):
        actions=[("Go to Chat",lambda:self.navigate("Chat")),("Go to Apps",lambda:self.navigate("Apps")),("Go to Files",lambda:self.navigate("Files")),("Go to Projects",lambda:self.navigate("Projects")),("Go to System",lambda:self.navigate("System")),("Go to Notes",lambda:self.navigate("Notes")),("Go to Memory",lambda:self.navigate("Memory")),("Go to Settings",lambda:self.navigate("Settings")),("Open VS Code",self.open_vscode),("Open Spotify",self.open_spotify),("Open Downloads",self.open_downloads),("Show / Hide Alevik AI",self.toggle_visibility)]
        for p in self.project_manager.projects():
            actions.append((f"Project: {p.get('name')} — open folder",lambda p=p:self._safe_project(self.project_manager.open_folder,p)))
            actions.append((f"Project: {p.get('name')} — VS Code",lambda p=p:self._safe_project(self.project_manager.open_vscode,p)))
            actions.append((f"Project: {p.get('name')} — run",lambda p=p:self._safe_project(self.project_manager.run_project,p)))
        return actions
    def _safe_project(self,fn,p):
        try:fn(p);self.recent_apps.mark(p.get("name","Project"))
        except Exception as exc:self.notifications.add(str(exc),"error")
    def open_command_palette(self):CommandPalette(self,self.palette_actions()).exec()

    # ------------------------- settings/theme ----------------------------
    def apply_settings(self):
        cfg=load_local_settings(); QApplication.instance().setStyleSheet(build_stylesheet(cfg)); theme=cfg.get("theme",{}); self.background.set_blur(theme.get("background_blur",2)); mode=theme.get("layout","Full"); self.right_panel.setVisible(mode!="Compact"); self.sidebar.setFixedWidth(180 if mode=="Compact" else 220)
        integ=cfg.get("integrations",{}); self.claude_client.api_key=integ.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY",""); self.local_ai_client.base_url=(integ.get("lm_studio_url") or "http://localhost:1234/v1").rstrip("/"); self.weather_service.default_location=cfg.get("weather",{}).get("location") or DEFAULT_WEATHER_LOCATION; self.spotify_service.reload(); self.right_panel.weather_panel.weather.default_location=self.weather_service.default_location; self.right_panel.weather_panel.refresh_weather(); self.right_panel.spotify_panel.refresh()

    # ------------------------- tray/hotkeys ------------------------------
    def _setup_tray(self):
        icon=self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon); self.setWindowIcon(icon); self.tray=QSystemTrayIcon(icon,self); menu=QMenu(); self.tray_toggle=QAction("Show / Hide Alevik AI",self); self.tray_toggle.triggered.connect(self.toggle_visibility); menu.addAction(self.tray_toggle); menu.addAction("Command Palette",self.open_command_palette); menu.addSeparator(); quit_action=QAction("Quit Alevik AI",self); quit_action.triggered.connect(self.quit_app); menu.addAction(quit_action); self.tray.setContextMenu(menu); self.tray.activated.connect(self._tray_activated); self.tray.setToolTip(f"Alevik AI v{APP_VERSION}"); self.tray.show()
    def _tray_activated(self,reason):
        if reason==QSystemTrayIcon.ActivationReason.Trigger:self.toggle_visibility()
    def _setup_hotkeys(self):
        self.hotkey_bridge=HotkeyBridge(); self.hotkey_bridge.toggle_requested.connect(self.toggle_visibility)
        if global_keyboard is not None:
            try:self._hotkey_id=global_keyboard.add_hotkey("ctrl+alt+a",lambda:self.hotkey_bridge.toggle_requested.emit());self.notifications.add("Ctrl+Alt+A global hotkey active","success","hotkey",10)
            except Exception as exc:self.notifications.add(f"Global hotkey unavailable: {exc}","warning","hotkey_error",60)
        else:self.notifications.add("Global hotkey needs: pip install keyboard","info","hotkey_missing",300)
    def toggle_visibility(self):
        if self.isVisible() and not self.isMinimized():self.hide()
        else:self.showNormal();self.raise_();self.activateWindow()
    def quit_app(self):
        self._force_quit=True
        if self._hotkey_id is not None and global_keyboard is not None:
            try:global_keyboard.remove_hotkey(self._hotkey_id)
            except Exception:pass
        self.tray.hide(); self.app.quit()
    def closeEvent(self,event):
        if self._force_quit:return super().closeEvent(event)
        event.ignore();self.hide();self.tray.showMessage("Alevik AI","Alevik jäi taustalle. Ctrl+Alt+A näyttää sen uudelleen.",QSystemTrayIcon.MessageIcon.Information,2500)

    # ------------------------- Windows startup ---------------------------
    @staticmethod
    def _startup_command():
        if getattr(sys,"frozen",False):return f'"{sys.executable}"'
        exe=Path(sys.executable); pythonw=exe.with_name("pythonw.exe") if os.name=="nt" else exe
        if os.name=="nt" and pythonw.exists():exe=pythonw
        return f'"{exe}" "{Path(__file__).resolve()}"'
    def startup_enabled(self):
        if os.name!="nt":return False
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r"Software\Microsoft\Windows\CurrentVersion\Run",0,winreg.KEY_READ) as key:winreg.QueryValueEx(key,"AlevikAI");return True
        except Exception:return False
    def set_startup_enabled(self,enabled):
        if os.name!="nt":return
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r"Software\Microsoft\Windows\CurrentVersion\Run",0,winreg.KEY_SET_VALUE) as key:
            if enabled:winreg.SetValueEx(key,"AlevikAI",0,winreg.REG_SZ,self._startup_command())
            else:
                try:winreg.DeleteValue(key,"AlevikAI")
                except FileNotFoundError:pass


def install_exception_hook(window_getter=None):
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return sys.__excepthook__(exc_type, exc, tb)
        log_exception("unhandled", exc)
        try:
            win=window_getter() if window_getter else None
            if win and getattr(win,"notifications",None):win.notifications.add(f"Virhe tallennettu crash logiin: {exc}","error","crash",5)
        except Exception:pass
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook=hook


def main():
    app=QApplication(sys.argv); app.setQuitOnLastWindowClosed(False); app.setStyleSheet(build_stylesheet())
    server_name="AlevikAI_SingleInstance_v2"; probe=QLocalSocket(); probe.connectToServer(server_name)
    if probe.waitForConnected(250):
        probe.write(b"SHOW");probe.flush();probe.waitForBytesWritten(250);probe.disconnectFromServer();return
    QLocalServer.removeServer(server_name);server=QLocalServer(app);server.listen(server_name)
    holder={}; install_exception_hook(lambda:holder.get("window")); window=AlevikMainWindow(app);holder["window"]=window;window.show()
    def activate_existing():
        while server.hasPendingConnections():
            sock=server.nextPendingConnection();sock.waitForReadyRead(100);window.showNormal();window.raise_();window.activateWindow();sock.disconnectFromServer()
    server.newConnection.connect(activate_existing)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
