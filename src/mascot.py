"""Alpha-chan AI chat and draggable mascot widget."""
from __future__ import annotations

import base64
import json
import os
import tomllib
import re
from functools import lru_cache
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components


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
    return " ".join(str(message or response.text or "request failed").split())[:300]


@lru_cache(maxsize=8)
def _available_gemini_models(key: str) -> tuple[str, ...]:
    response = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key}, timeout=25)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {_error_message(response)}")
    models = []
    for model in response.json().get("models", []):
        name = str(model.get("name", "")).removeprefix("models/")
        if name and "generateContent" in model.get("supportedGenerationMethods", []):
            models.append(name)
    def model_rank(name: str) -> tuple[int, int, str]:
        versions = [int(part) for part in re.findall(r"(?:^|[-.])([0-9]+)(?=[-.]|$)", name)]
        version = versions[0] * 100 + (versions[1] if len(versions) > 1 else 0) if versions else 0
        special = int("-tts" in name or "-image" in name)
        return (version, -special, name)
    return tuple(sorted(models, key=model_rank, reverse=True))


def _gemini_contents(history: list[dict], user_message: str) -> list[dict]:
    contents: list[dict] = []
    for item in [*history[-12:], {"role": "user", "content": user_message}]:
        text = str(item.get("content", "")).strip()
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


def _gemini_request(key: str, model: str, system: str, contents: list[dict]) -> str:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    generation_config = {
        "thinkingConfig": {"thinkingBudget": 0},
        "maxOutputTokens": 450,
        "temperature": 0.4,
    }
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": generation_config,
    }
    try:
        response = requests.post(endpoint, params={"key": key}, json=payload, timeout=60)
    except requests.RequestException as error:
        raise RuntimeError(f"Gemini transport error: {error}") from error
    if response.status_code == 400:
        retry_payload = {**payload, "generationConfig": {key: value for key, value in generation_config.items() if key != "thinkingConfig"}}
        try:
            response = requests.post(endpoint, params={"key": key}, json=retry_payload, timeout=60)
        except requests.RequestException as error:
            raise RuntimeError(f"Gemini transport error: {error}") from error
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {_error_message(response)}")
    try:
        parts = response.json()["candidates"][0]["content"]["parts"]
        answer = "\n".join(
            str(part.get("text", "")).strip()
            for part in parts
            if part.get("thought") is not True and part.get("text")
        ).strip()
        if not answer:
            raise ValueError("no visible answer parts")
        return answer
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid Gemini response: {response.text[:300]}") from error


def ask_alpha_chan(user_message: str, history: list[dict], lang: str, context_summary: dict, user_api_key: str = "") -> tuple[str, str]:
    """Ask Gemini and return a live answer; API failures are reported, never hidden by fallback text."""
    key = _api_key(user_api_key)
    if not key:
        return "Gemini API key is not configured.", "Gemini unavailable"
    language = "Russian" if re.search(r"[А-Яа-яЁё]", user_message) else "English"
    language = "Russian" if lang in {"Русский", "Р СѓСЃСЃРєРёР№"} else "English"
    language = "Russian" if any("\u0400" <= char <= "\u04ff" for char in user_message) else "English"
    system = ("You are Alpha-chan, a warm, sharp assistant inside Circular Alpha. Answer naturally in "
              f"{language}; explain finance simply for non-quants. Never invent numbers: use only the live dashboard context, "
              "distinguish stress tests from forecasts, and do not give personalised investment advice. "
              "Always reply in the exact language the user wrote in (Russian if Russian, English if English). "
              "Be warm, concise (2–4 sentences), and never output internal reasoning or meta bullets.\n\n"
              "LIVE DASHBOARD CONTEXT:\n" + json.dumps(context_summary, ensure_ascii=False, default=str))
    contents = _gemini_contents(history, user_message)
    try:
        return _gemini_request(key, "gemini-3.8-flash", system, contents), "Gemini"
    except RuntimeError as error:
        if "transport error" in str(error).lower():
            return f"Gemini error: {error}", "Gemini error"
        initial_error = error
    for model in _available_gemini_models(key):
        if model == "gemini-3.8-flash":
            continue
        try:
            return _gemini_request(key, model, system, contents), "Gemini"
        except RuntimeError as error:
            if "transport error" in str(error).lower():
                return f"Gemini error: {error}", "Gemini error"
    return f"Gemini error: {initial_error}", "Gemini error"


@st.fragment
def render_ai_chat(lang: str, context_summary: dict, key_suffix: str = "top") -> None:
    prefix = str(key_suffix)
    st.session_state.setdefault("mascot_messages", [])
    st.session_state.setdefault("alpha_api_key_value", "")
    configured_key = bool(_api_key())
    with st.popover("💬 Chat with Alpha-chan (AI)", use_container_width=True):
        if not configured_key:
            with st.expander("🔑 Connect Live AI Key (optional)", expanded=False):
                widget = f"alpha_api_key_{prefix}"
                st.text_input("API key (session only)", type="password", key=widget, placeholder="Gemini API key")
                st.session_state["alpha_api_key_value"] = st.session_state.get(widget, "")
                st.markdown("[Get a Gemini key](https://aistudio.google.com/apikey)")
        quick = ["What is circular financing?", "What happens if the selected company collapses?", "How should I read Cluster Lens?", "Did the SMA rule beat buy-and-hold?"]
        cols = st.columns(2)
        for i, question in enumerate(quick):
            if cols[i % 2].button(question, key=f"alpha_quick_{prefix}_{i}"):
                st.session_state["mascot_pending"] = question
        for item in st.session_state["mascot_messages"]:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
        prompt = st.chat_input("Ask Alpha-chan anything...", key=f"alpha_input_{prefix}") or st.session_state.pop("mascot_pending", "")
        if prompt:
            st.session_state["mascot_messages"].append({"role": "user", "content": prompt})
            answer, provider = ask_alpha_chan(prompt, st.session_state["mascot_messages"][:-1], lang, context_summary, st.session_state["alpha_api_key_value"])
            st.session_state["mascot_messages"].append({"role": "assistant", "content": answer})
            st.caption(f"Answer source: {provider}")
        if st.button("Clear chat", key=f"alpha_clear_{prefix}"):
            st.session_state["mascot_messages"] = []


def _asset_data_uri() -> str:
    asset = Path(__file__).resolve().parents[1] / "assets" / "mascot.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(asset.read_bytes()).decode("ascii") if asset.exists() else ""


def render_mascot(lang: str, context_summary: dict) -> None:
    asset = _asset_data_uri()
    image = f'<img src="{asset}" alt="Alpha-chan" />' if asset else "<div>α</div>"
    greeting = "Привет! Я Alpha-chan — спроси меня о чём угодно!" if lang in {"Русский", "Р СѓСЃСЃРєРёР№"} else "Hi! I'm Alpha-chan — ask me anything!"
    html = f'''<div style="position:fixed;right:24px;bottom:24px;z-index:99999;font-family:Arial;color:#311c60"><div style="background:white;border:1px solid #debeef;border-radius:12px;padding:9px;width:200px;font-size:12px;box-shadow:0 6px 18px #311c6030">{greeting}</div><div style="width:88px;height:88px;border:2px solid #4b279b;border-radius:16px;background:#f7f3fc;display:grid;place-items:center">{image}</div></div>'''
    components.html(html, height=0, width=0)


def render_mascot(lang: str, context_summary: dict) -> None:
    """Render the mascot in the parent Streamlit page, with iframe drag support when available."""
    asset = _asset_data_uri()
    image = f'<img src="{asset}" alt="Alpha-chan" />' if asset else (
        '<svg viewBox="0 0 96 96" role="img" aria-label="Alpha-chan">'
        '<circle cx="48" cy="48" r="44" fill="#f0e7ff" stroke="#6d3fc2" stroke-width="3"/>'
        '<circle cx="34" cy="43" r="5" fill="#311c60"/><circle cx="62" cy="43" r="5" fill="#311c60"/>'
        '<path d="M31 59 Q48 72 65 59" fill="none" stroke="#311c60" stroke-width="3" stroke-linecap="round"/>'
        '</svg>'
    )
    russian = bool(re.search(r"рус", str(lang), re.IGNORECASE))
    greeting = "Привет! Я Alpha-chan — спроси меня о чём угодно!" if russian else "Hi! I'm Alpha-chan — ask me anything!"
    html = f'''<style>
@keyframes alpha-float {{ 0%, 100% {{ transform: translateY(0) rotate(-1deg); }} 50% {{ transform: translateY(-7px) rotate(1deg); }} }}
#circular-alpha-mascot {{ position:fixed; bottom:88px; right:28px; z-index:2147483647; font-family:Arial,sans-serif; color:#311c60; animation:alpha-float 4s ease-in-out infinite; }}
#circular-alpha-mascot .alpha-bubble {{ background:#fff; border:1px solid #debeef; border-radius:14px; padding:9px 11px; width:200px; margin-bottom:8px; font-size:12px; line-height:1.35; box-shadow:0 6px 18px #311c6030; }}
#circular-alpha-mascot .alpha-avatar {{ width:88px; height:88px; border:2px solid #4b279b; border-radius:18px; background:#f7f3fc; display:grid; place-items:center; overflow:hidden; transition:transform .25s ease, box-shadow .25s ease; }}
#circular-alpha-mascot img, #circular-alpha-mascot svg {{ width:100%; height:100%; object-fit:cover; }}
#circular-alpha-mascot:hover .alpha-avatar {{ transform:rotate(-7deg) scale(1.08); box-shadow:0 0 22px 6px #9b6cffaa; }}
</style><div id="circular-alpha-mascot" title="Drag Alpha-chan"><div class="alpha-bubble">{greeting}</div><div class="alpha-avatar">{image}</div></div>'''
    st.markdown(html, unsafe_allow_html=True)
    components.html('''<script>
try {
  const mascot = window.parent.document.getElementById("circular-alpha-mascot");
  if (mascot) {
    mascot.style.cursor = "grab";
    let dragging = false, offsetX = 0, offsetY = 0;
    mascot.addEventListener("pointerdown", (event) => { dragging = true; mascot.setPointerCapture(event.pointerId); const rect = mascot.getBoundingClientRect(); offsetX = event.clientX - rect.left; offsetY = event.clientY - rect.top; mascot.style.cursor = "grabbing"; });
    mascot.addEventListener("pointermove", (event) => { if (!dragging) return; mascot.style.left = `${event.clientX - offsetX}px`; mascot.style.top = `${event.clientY - offsetY}px`; mascot.style.right = "auto"; mascot.style.bottom = "auto"; });
    mascot.addEventListener("pointerup", () => { dragging = false; mascot.style.cursor = "grab"; });
  }
} catch (e) {}
</script>''', height=0, width=0)
