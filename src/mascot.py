"""Alpha-chan AI chat, local fallback, and draggable mascot widget."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components


def _api_key(user_api_key: str = "") -> str:
    if user_api_key.strip():
        return str(user_api_key).strip().strip('"').strip("'")
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("OPENAI_API_KEY") or ""
        return str(key).strip().strip('"').strip("'")
    except Exception:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        return str(key).strip().strip('"').strip("'")


def _local_reply(message: str, lang: str, context: dict) -> str:
    company, shock = context.get("selected_company", "the selected company"), context.get("shock_pct", "the selected")
    if lang == "Русский":
        return f"Я помогу разобрать Circular Alpha. Сейчас {company} имеет сценарий шока {shock}%. Подключите Gemini или OpenAI key для свободного диалога; без ключа я использую только факты панели и не выдумываю числа."
    return f"I can help explain Circular Alpha. Right now {company} has a {shock}% shock scenario. Connect a Gemini or OpenAI key for a full conversation; without one I use only dashboard facts and never invent numbers."


def _error_message(response: requests.Response) -> str:
    try:
        message = response.json().get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = ""
    return " ".join(str(message or response.text or "request failed").split())[:180]


def ask_alpha_chan(user_message: str, history: list[dict], lang: str, context_summary: dict, user_api_key: str = "") -> tuple[str, str]:
    """Ask OpenAI or Gemini, returning (answer, provider); always fails soft."""
    key = _api_key(user_api_key)
    if not key:
        return _local_reply(user_message, lang, context_summary), "local"
    system = ("You are Alpha-chan, a warm, sharp assistant inside Circular Alpha. Answer naturally in "
              f"{'Russian' if lang == 'Русский' else 'English'}; explain finance simply for non-quants. Never invent numbers: "
              "use only the live dashboard context, say when unavailable, distinguish stress tests from forecasts, and do not give personalised investment advice.\n\n"
              "LIVE DASHBOARD CONTEXT:\n" + json.dumps(context_summary, ensure_ascii=False, default=str))
    try:
        messages = [{"role": "system", "content": system}]
        messages.extend({"role": m.get("role", "user"), "content": m.get("content", "")} for m in history[-12:])
        messages.append({"role": "user", "content": user_message})
        if key.startswith("sk-"):
            response = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0.3}, timeout=25)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip(), "OpenAI"
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages[1:]
        ]
        last_error = ""
        for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                json={"system_instruction": {"parts": [{"text": system}]}, "contents": contents},
                timeout=25,
            )
            if response.status_code == 404:
                last_error = f"HTTP {response.status_code}: {_error_message(response)}"
                continue
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {_error_message(response)}")
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip(), "Gemini"
        raise RuntimeError(last_error or "HTTP 404: model not found")
    except Exception as error:
        diagnostic = str(error)[:240]
        company = context_summary.get("selected_company", "the selected company")
        shock = context_summary.get("shock_pct", "the selected")
        fallback = f"I can help explain Circular Alpha. Right now {company} has a {shock}% shock scenario. I’m using dashboard facts and never invent numbers."
        return fallback, f"local ({diagnostic})"


def render_ai_chat(lang: str, context_summary: dict, key_suffix: str = "top") -> None:
    prefix = str(key_suffix)
    if "mascot_messages" not in st.session_state:
        st.session_state.mascot_messages = []
    if "alpha_api_key_value" not in st.session_state:
        st.session_state["alpha_api_key_value"] = ""
    label = "💬 Chat with Alpha-chan (AI)" if lang != "Русский" else "💬 Чат с Alpha-chan (ИИ)"
    configured_key = any(str(os.environ.get(name) or "").strip() for name in ("GEMINI_API_KEY", "OPENAI_API_KEY"))
    if not configured_key:
        try:
            configured_key = any(str(st.secrets.get(name) or "").strip() for name in ("GEMINI_API_KEY", "OPENAI_API_KEY"))
        except Exception:
            configured_key = False
    with st.popover(label, use_container_width=True):
        with st.expander("🔑 Connect Live AI Key (optional)", expanded=False):
            api_key_widget = f"alpha_api_key_{prefix}"
            st.text_input("API key (session only)", type="password", key=api_key_widget, placeholder="Gemini or sk-...", help="Never written to disk")
            st.session_state["alpha_api_key_value"] = st.session_state.get(api_key_widget, "")
            st.markdown("[Get a Gemini key](https://aistudio.google.com/apikey)")
        quick = ["What is circular financing?", "What happens if the selected company collapses?", "How should I read Cluster Lens?", "Did the SMA rule beat buy-and-hold?"]
        cols = st.columns(2)
        for i, question in enumerate(quick):
            if cols[i % 2].button(question, key=f"alpha_quick_{prefix}_{i}"):
                st.session_state.mascot_pending = question
        for item in st.session_state.mascot_messages:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
        prompt = st.chat_input("Ask Alpha-chan anything...", key=f"alpha_input_{prefix}")
        prompt = prompt or st.session_state.pop("mascot_pending", "")
        if prompt:
            st.session_state.mascot_messages.append({"role": "user", "content": prompt})
            answer, provider = ask_alpha_chan(prompt, st.session_state.mascot_messages[:-1], lang, context_summary, st.session_state["alpha_api_key_value"])
            st.session_state.mascot_messages.append({"role": "assistant", "content": answer})
            st.caption(f"Answer source: {provider}")
            st.rerun()
        if st.button("Clear chat", key=f"alpha_clear_{prefix}"):
            st.session_state.mascot_messages = []
            st.rerun()


def _asset_data_uri() -> str:
    asset = Path(__file__).resolve().parents[1] / "assets" / "mascot.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(asset.read_bytes()).decode("ascii") if asset.exists() else ""


def render_mascot(lang: str, context_summary: dict) -> None:
    greeting = "Привет! Я Alpha-chan — спроси меня что угодно!" if lang == "Русский" else "Hi! I'm Alpha-chan — ask me anything!"
    asset = _asset_data_uri()
    image = f'<img src="{asset}" alt="Alpha-chan" />' if asset else '<div>α</div>'
    html = f'''<script>(()=>{{const d=window.parent.document,old=d.getElementById('alpha-float');if(old)old.remove();const r=d.createElement('div');r.id='alpha-float';r.innerHTML=`<style>#alpha-float{{position:fixed;right:24px;bottom:24px;z-index:99999;font-family:Arial;color:#311c60}}#alpha-float .bubble{{position:absolute;right:0;bottom:108px;background:white;border:1px solid #debeef;border-radius:12px;padding:9px;width:200px;font-size:12px;box-shadow:0 6px 18px #311c6030}}#alpha-float .frame{{width:88px;height:88px;border:2px solid #4b279b;border-radius:16px;background:#f7f3fc;display:grid;place-items:center;cursor:grab;animation:bob 3s ease-in-out infinite;transition:filter .2s}}#alpha-float .frame:hover{{filter:drop-shadow(0 0 10px #d9a441);transform:rotate(-4deg)}}#alpha-float .frame.dragging{{animation:wobble .35s infinite;cursor:grabbing}}#alpha-float img{{width:80px;height:80px;object-fit:cover;border-radius:12px;image-rendering:pixelated}}.spark{{position:absolute;width:6px;height:6px;background:#d9a441;box-shadow:14px 2px #4b279b,-12px 8px #ab8fbb;animation:burst .65s forwards;pointer-events:none}}@keyframes bob{{50%{{transform:translateY(-6px)}}}}@keyframes wobble{{50%{{transform:rotate(4deg)}}}}@keyframes burst{{to{{transform:translate(var(--x),var(--y)) scale(1.5);opacity:0}}}}</style><div class="bubble">{greeting}</div><div class="frame">{image}</div>`;d.body.appendChild(r);const f=r.querySelector('.frame');function spark(){{for(let i=0;i<6;i++){{const s=d.createElement('i');s.className='spark';s.style.setProperty('--x',(Math.random()*60-30)+'px');s.style.setProperty('--y',(Math.random()*-50-10)+'px');r.appendChild(s);setTimeout(()=>s.remove(),700)}}}}f.onclick=()=>{{if(!r.dataset.dragged)spark();r.dataset.dragged=''}};let drag=false,sx,sy,ox,oy;f.onpointerdown=e=>{{drag=true;r.dataset.dragged='';f.classList.add('dragging');sx=e.clientX;sy=e.clientY;const q=r.getBoundingClientRect();ox=q.left;oy=q.top;f.setPointerCapture(e.pointerId)}};f.onpointermove=e=>{{if(drag){{if(Math.abs(e.clientX-sx)+Math.abs(e.clientY-sy)>5)r.dataset.dragged='1';r.style.left=ox+e.clientX-sx+'px';r.style.top=oy+e.clientY-sy+'px';r.style.right='auto';r.style.bottom='auto'}}}};f.onpointerup=()=>{{drag=false;f.classList.remove('dragging')}}}})();</script>'''
    components.html(html, height=0, width=0)


def render_ai_chat(lang: str, context_summary: dict, key_suffix: str = "top") -> None:
    prefix = str(key_suffix)
    if "mascot_messages" not in st.session_state:
        st.session_state.mascot_messages = []
    if "alpha_api_key_value" not in st.session_state:
        st.session_state["alpha_api_key_value"] = ""
    configured_key = any(str(os.environ.get(name) or "").strip() for name in ("GEMINI_API_KEY", "OPENAI_API_KEY"))
    if not configured_key:
        try:
            configured_key = any(str(st.secrets.get(name) or "").strip() for name in ("GEMINI_API_KEY", "OPENAI_API_KEY"))
        except Exception:
            configured_key = False
    with st.popover("💬 Chat with Alpha-chan (AI)", use_container_width=True):
        if not configured_key:
            with st.expander("🔑 Connect Live AI Key (optional)", expanded=False):
                api_key_widget = f"alpha_api_key_{prefix}"
                st.text_input("API key (session only)", type="password", key=api_key_widget, placeholder="Gemini or sk-...", help="Never written to disk")
                st.session_state["alpha_api_key_value"] = st.session_state.get(api_key_widget, "")
                st.markdown("[Get a Gemini key](https://aistudio.google.com/apikey)")
        quick = ["What is circular financing?", "What happens if the selected company collapses?", "How should I read Cluster Lens?", "Did the SMA rule beat buy-and-hold?"]
        cols = st.columns(2)
        for i, question in enumerate(quick):
            if cols[i % 2].button(question, key=f"alpha_quick_{prefix}_{i}"):
                st.session_state.mascot_pending = question
        for item in st.session_state.mascot_messages:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
        prompt = st.chat_input("Ask Alpha-chan anything...", key=f"alpha_input_{prefix}")
        prompt = prompt or st.session_state.pop("mascot_pending", "")
        if prompt:
            st.session_state.mascot_messages.append({"role": "user", "content": prompt})
            answer, provider = ask_alpha_chan(prompt, st.session_state.mascot_messages[:-1], lang, context_summary, st.session_state["alpha_api_key_value"])
            st.session_state.mascot_messages.append({"role": "assistant", "content": answer})
            st.caption(f"Answer source: {provider}")
            st.rerun()
        if st.button("Clear chat", key=f"alpha_clear_{prefix}"):
            st.session_state.mascot_messages = []
            st.rerun()
