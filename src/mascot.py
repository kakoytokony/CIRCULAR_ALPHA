"""Alpha-chan chat and the floating mascot used by the Streamlit app."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import tomllib
from pathlib import Path

import requests
import streamlit as st


_GEMINI_MODEL = "gemini-3.5-flash-lite"
_REQUEST_TIMEOUT = (3.0, 8.0)
_MAX_HISTORY_ITEMS = 6
_MAX_CONTEXT_CHARS = 2400
_FRIENDLY_API_ERROR = "I’m having trouble reaching Alpha-chan right now. Please try again in a moment."
_LOGGER = logging.getLogger(__name__)


def _clean_key(value: object) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _api_key(user_api_key: str = "") -> str:
    if user_api_key.strip():
        return _clean_key(user_api_key)
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = ""
    if not key:
        try:
            with (Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml").open("rb") as handle:
                key = tomllib.load(handle).get("GEMINI_API_KEY", "")
        except (OSError, tomllib.TOMLDecodeError):
            key = ""
    return _clean_key(key or os.environ.get("GEMINI_API_KEY"))


def _error_message(response: requests.Response) -> str:
    try:
        message = response.json().get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = ""
    return " ".join(str(message or response.text or "request failed").split())[:240]


def _error_category(response: requests.Response) -> str:
    try:
        category = response.json().get("error", {}).get("status", "")
    except (ValueError, AttributeError):
        category = ""
    return str(category or f"HTTP_{response.status_code}")[:64]


def _is_russian(text: str) -> bool:
    return any("\u0400" <= char <= "\u04ff" for char in text)


def _trim_context(context_summary: dict) -> str:
    return json.dumps(context_summary or {}, ensure_ascii=False, default=str)[:_MAX_CONTEXT_CHARS]


def _gemini_contents(history: list[dict], user_message: str) -> list[dict]:
    contents: list[dict] = []
    for item in [*history[-_MAX_HISTORY_ITEMS:], {"role": "user", "content": user_message}]:
        text = str(item.get("content", "")).strip()[:1200]
        if not text:
            continue
        role = "model" if item.get("role") in {"assistant", "model"} else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n" + text
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
    if not contents or contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": user_message.strip() or "Hello"}]})
    return contents


def _parse_gemini_answer(response: requests.Response) -> str:
    parts = response.json()["candidates"][0]["content"]["parts"]
    answer = "\n".join(
        str(part.get("text", "")).strip()
        for part in parts
        if part.get("thought") is not True and part.get("text")
    ).strip()
    if not answer:
        raise ValueError("Gemini returned no visible answer")
    return answer


def _gemini_request(key: str, system: str, contents: list[dict]) -> str:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent"
    generation_config = {
        "maxOutputTokens": 220,
        "temperature": 0.4,
    }
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": generation_config,
    }
    started = time.perf_counter()
    try:
        response = requests.post(endpoint, params={"key": key}, json=payload, timeout=_REQUEST_TIMEOUT)
    except requests.RequestException as error:
        _LOGGER.warning(
            "Gemini request failed: status=transport_error category=%s model=%s duration_ms=%d",
            type(error).__name__, _GEMINI_MODEL, round((time.perf_counter() - started) * 1000),
        )
        raise RuntimeError("Gemini request failed") from error
    if response.status_code != 200:
        _LOGGER.warning(
            "Gemini request failed: status=%d category=%s model=%s duration_ms=%d message=%s",
            response.status_code,
            _error_category(response),
            _GEMINI_MODEL,
            round((time.perf_counter() - started) * 1000),
            _error_message(response),
        )
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {_error_message(response)}")
    _LOGGER.info(
        "Gemini request succeeded: status=%d model=%s duration_ms=%d",
        response.status_code, _GEMINI_MODEL, round((time.perf_counter() - started) * 1000),
    )
    try:
        return _parse_gemini_answer(response)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        _LOGGER.warning(
            "Gemini response parsing failed: status=%d category=INVALID_RESPONSE model=%s duration_ms=%d message=%s",
            response.status_code,
            _GEMINI_MODEL,
            round((time.perf_counter() - started) * 1000),
            str(error)[:240],
        )
        raise RuntimeError("Gemini returned an invalid response") from error


def ask_alpha_chan(user_message: str, history: list[dict], lang: str, context_summary: dict, user_api_key: str = "") -> tuple[str, str]:
    """Return a short answer without exposing transport errors in the conversation."""
    key = _api_key(user_api_key)
    if not key:
        return "Add a Gemini API key above to chat with Alpha-chan.", "AI unavailable"
    language = "Russian" if _is_russian(user_message) else "English"
    system = (
        "You are Alpha-chan, a warm, sharp assistant inside Circular Alpha. "
        f"Always reply in the exact language the user wrote in ({language}; Russian if Russian, English if English). "
        "Be warm, concise (2–4 sentences), and never output internal reasoning or meta bullets. "
        "Use only the live dashboard context for numbers, distinguish stress tests from forecasts, "
        "and do not give personalised investment advice.\n\n"
        "LIVE DASHBOARD CONTEXT:\n" + _trim_context(context_summary)
    )
    try:
        return _gemini_request(key, system, _gemini_contents(history, user_message)), "Gemini"
    except RuntimeError:
        return _FRIENDLY_API_ERROR, "AI unavailable"


def _chat_css() -> None:
    st.markdown(
        """<style>
        div[data-testid="stPopover"] div[data-testid="stChatMessage"] { padding: .25rem .45rem; margin: 0 0 .2rem; }
        div[data-testid="stPopover"] div[data-testid="stChatMessageContent"] { padding: .35rem .55rem; border-radius: 12px; }
        </style>""",
        unsafe_allow_html=True,
    )


@st.fragment
def render_mascot(lang: str, context_summary: dict) -> None:
    """Render one interactive mascot and its compact chat panel."""
    st.session_state.setdefault("mascot_messages", [])
    st.session_state.setdefault("alpha_api_key_value", "")
    st.session_state.setdefault("alpha_chat_open", False)
    asset = _asset_data_uri()
    asset_background = (
        f'background:#f7f3fc url("{asset}") center / contain no-repeat;'
        if asset
        else "background:#fff4e5;"
    )
    mascot_label = "Chat with Alpha-chan" if asset else "Mascot image unavailable"
    st.markdown(
        f'''<style>
        .st-key-alpha-mascot-widget {{ position:fixed; right:28px; bottom:28px; z-index:2147483647; width:96px; }}
        .st-key-alpha-mascot-widget button {{ width:96px; height:96px; padding:0; border:2px solid #8d6ac5; border-radius:24px; {asset_background} image-rendering:pixelated; cursor:pointer; box-shadow:0 8px 24px #311c6030; transition:transform .2s ease,box-shadow .2s ease; }}
        .st-key-alpha-mascot-widget button:hover {{ transform:scale(1.08); box-shadow:0 0 0 4px #cdb6f855,0 10px 28px #311c6040; }}
        .st-key-alpha-mascot-widget button:focus-visible {{ outline:3px solid #9b6cff; outline-offset:3px; }}
        .st-key-alpha-mascot-widget button p {{ display:{'none' if asset else 'block'}; color:#7a4b00; font-size:11px; line-height:1.2; }}
        .st-key-alpha-chat-panel {{ position:fixed; right:28px; bottom:140px; z-index:2147483646; width:min(370px,calc(100vw - 32px)); max-height:calc(100vh - 164px); overflow-y:auto; padding:14px; background:rgba(255,255,255,.98); border:1px solid #debeef; border-radius:18px; box-shadow:0 14px 40px #311c6030; }}
        .st-key-alpha-chat-panel [data-testid="stChatMessage"] {{ padding:.25rem .45rem; margin:0 0 .2rem; }}
        .st-key-alpha-chat-panel [data-testid="stChatMessageContent"] {{ padding:.35rem .55rem; border-radius:12px; }}
        @media (max-width:700px) {{ .st-key-alpha-mascot-widget {{ right:16px; bottom:18px; width:72px; }} .st-key-alpha-mascot-widget button {{ width:72px; height:72px; border-radius:18px; }} .st-key-alpha-chat-panel {{ right:16px; bottom:104px; width:calc(100vw - 32px); max-height:calc(100vh - 122px); }} }}
        </style>''',
        unsafe_allow_html=True,
    )
    with st.container(key="alpha-mascot-widget"):
        if st.button(mascot_label, key="alpha_mascot_toggle", help=mascot_label, width="stretch"):
            st.session_state["alpha_chat_open"] = not st.session_state["alpha_chat_open"]
            st.rerun(scope="fragment")
    if not st.session_state["alpha_chat_open"]:
        return
    with st.container(key="alpha-chat-panel"):
        header_cols = st.columns([1, 0.18])
        header_cols[0].markdown("#### Alpha-chan")
        if header_cols[1].button("x", key="alpha_chat_close", help="Close chat"):
            st.session_state["alpha_chat_open"] = False
            st.rerun(scope="fragment")
        if not _api_key():
            with st.expander("Connect Live AI Key", expanded=False):
                st.text_input("API key (session only)", type="password", key="alpha_api_key_top", placeholder="Gemini API key")
                st.session_state["alpha_api_key_value"] = st.session_state.get("alpha_api_key_top", "")
                st.markdown("[Get a Gemini key](https://aistudio.google.com/apikey)")
        with st.container(height=260, border=False):
            for item in st.session_state["mascot_messages"]:
                with st.chat_message(item["role"]):
                    st.markdown(item["content"])
        quick = ["What is circular financing?", "What happens if the selected company collapses?"]
        quick_cols = st.columns(2)
        for index, question in enumerate(quick):
            if quick_cols[index].button(question, key=f"alpha_quick_{index}", width="stretch"):
                st.session_state["mascot_pending"] = question
        prompt = st.chat_input("Ask Alpha-chan anything...", key="alpha_input_top")
        if not prompt:
            prompt = st.session_state.pop("mascot_pending", "")
        if prompt:
            history = list(st.session_state["mascot_messages"])
            st.session_state["mascot_messages"].append({"role": "user", "content": prompt})
            answer, _provider = ask_alpha_chan(prompt, history, lang, context_summary, st.session_state["alpha_api_key_value"])
            st.session_state["mascot_messages"].append({"role": "assistant", "content": answer})
            st.rerun(scope="fragment")
        if st.button("Clear chat", key="alpha_clear_top", width="stretch"):
            st.session_state["mascot_messages"] = []
            st.rerun(scope="fragment")


@st.fragment
def render_ai_chat(lang: str, context_summary: dict, key_suffix: str = "top") -> None:
    """Render one compact chat surface; the sidebar call is intentionally a no-op."""
    if key_suffix != "top":
        return
    _chat_css()
    st.session_state.setdefault("mascot_messages", [])
    st.session_state.setdefault("alpha_api_key_value", "")
    with st.popover("💬 Chat with Alpha-chan", width="stretch"):
        if not _api_key():
            with st.expander("🔑 Connect Live AI Key", expanded=False):
                st.text_input("API key (session only)", type="password", key="alpha_api_key_top", placeholder="Gemini API key")
                st.session_state["alpha_api_key_value"] = st.session_state.get("alpha_api_key_top", "")
                st.markdown("[Get a Gemini key](https://aistudio.google.com/apikey)")
        with st.container(height=280, border=False):
            for item in st.session_state["mascot_messages"]:
                with st.chat_message(item["role"]):
                    st.markdown(item["content"])
        quick = ["What is circular financing?", "What happens if the selected company collapses?"]
        quick_cols = st.columns(2)
        for index, question in enumerate(quick):
            if quick_cols[index].button(question, key=f"alpha_quick_{index}", width="stretch"):
                st.session_state["mascot_pending"] = question
        prompt = st.chat_input("Ask Alpha-chan anything...", key="alpha_input_top")
        if not prompt:
            prompt = st.session_state.pop("mascot_pending", "")
        if prompt:
            history = list(st.session_state["mascot_messages"])
            st.session_state["mascot_messages"].append({"role": "user", "content": prompt})
            answer, _provider = ask_alpha_chan(prompt, history, lang, context_summary, st.session_state["alpha_api_key_value"])
            st.session_state["mascot_messages"].append({"role": "assistant", "content": answer})
            st.rerun(scope="fragment")
        if st.button("Clear chat", key="alpha_clear_top", width="stretch"):
            st.session_state["mascot_messages"] = []
            st.rerun(scope="fragment")


def _asset_data_uri() -> str:
    asset = Path(__file__).resolve().parents[1] / "assets" / "mascot.png"
    try:
        if not asset.is_file():
            raise FileNotFoundError(f"mascot asset does not exist: {asset}")
        payload = asset.read_bytes()
        if not payload:
            raise OSError(f"mascot asset is empty: {asset}")
    except OSError as error:
        _LOGGER.error("Mascot asset load failed: path=%s error=%s", asset, error, exc_info=True)
        return ""
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _legacy_render_mascot(lang: str, context_summary: dict) -> None:
    """Legacy decorative renderer retained for backwards compatibility."""
    asset = _asset_data_uri()
    image = f'<img src="{asset}" alt="Alpha-chan" />' if asset else (
        '<svg viewBox="0 0 96 96" role="img" aria-label="Alpha-chan">'
        '<circle cx="48" cy="48" r="44" fill="#f0e7ff" stroke="#6d3fc2" stroke-width="3"/>'
        '<circle cx="34" cy="43" r="5" fill="#311c60"/><circle cx="62" cy="43" r="5" fill="#311c60"/>'
        '<path d="M31 59 Q48 72 65 59" fill="none" stroke="#311c60" stroke-width="3" stroke-linecap="round"/>'
        '</svg>'
    )
    greeting = "Привет! Я Alpha-chan — спроси меня о чём угодно!" if _is_russian(str(lang)) else "Hi! I'm Alpha-chan — ask me anything!"
    st.markdown(
        f'''<style>
        @keyframes alpha-float {{ 0%,100% {{ transform:translateY(0) rotate(-1deg); }} 50% {{ transform:translateY(-6px) rotate(1deg); }} }}
        #circular-alpha-mascot {{ position:fixed; right:28px; bottom:88px; z-index:2147483647; width:min(220px,calc(100vw - 40px)); pointer-events:none; font-family:'Plus Jakarta Sans',Arial,sans-serif; color:#311c60; animation:alpha-float 4s ease-in-out infinite; }}
        #circular-alpha-mascot .alpha-bubble {{ background:#fff; border:1px solid #debeef; border-radius:14px; padding:9px 11px; margin-bottom:8px; font-size:12px; line-height:1.35; box-shadow:0 6px 18px #311c6030; }}
        #circular-alpha-mascot .alpha-avatar {{ width:88px; height:88px; border:2px solid #4b279b; border-radius:18px; background:#f7f3fc; display:grid; place-items:center; overflow:hidden; transition:transform .25s ease,box-shadow .25s ease; margin-left:auto; pointer-events:auto; }}
        #circular-alpha-mascot img,#circular-alpha-mascot svg {{ width:100%; height:100%; object-fit:cover; }}
        #circular-alpha-mascot:hover .alpha-avatar {{ transform:rotate(-7deg) scale(1.06); box-shadow:0 0 22px 6px #9b6cffaa; }}
        @media (max-width:700px) {{ #circular-alpha-mascot {{ right:12px; bottom:76px; width:180px; }} #circular-alpha-mascot .alpha-bubble {{ font-size:11px; }} #circular-alpha-mascot .alpha-avatar {{ width:68px; height:68px; }} }}
        </style><div id="circular-alpha-mascot"><div class="alpha-bubble">{greeting}</div><div class="alpha-avatar">{image}</div></div>''',
        unsafe_allow_html=True,
    )
