"""
Gradio chat interface with distinct personality and conversation memory.
Run with: gradio gradio_app.py (or python gradio_app.py)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Backend URL (same process or set GRADIO_BACKEND_URL)
BACKEND_URL = os.getenv("GRADIO_BACKEND_URL", "http://127.0.0.1:8000")

# Personality: warm, slightly witty makeup & lifestyle advisor
PERSONALITY_NAME = "Glow"
PERSONALITY_GREETING = (
    "Hey! I'm **Glow** — your makeup and lifestyle buddy. "
    "I can help you discover products from our catalog, search by vibe (like 'red lipstick under $10'), "
    "or look things up on the web. What’s on your mind?"
)


def _content_str(content) -> str:
    """Normalize message content to string (Gradio may send list for multimodal)."""
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(str(p) for p in content).strip()
    return str(content).strip()


def _messages_for_api(messages: list[dict]) -> list[dict]:
    """Convert Gradio message list to API history (user/assistant only)."""
    out = []
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            continue
        content = _content_str(m.get("content"))
        if not content:
            continue
        out.append({"role": m["role"], "content": content})
    return out


def chat_with_backend(user_message: str, messages: list[dict]) -> tuple[str, list[dict]]:
    """Send user message and history to the FastAPI backend; return new messages (Gradio 6 format)."""
    history_list = _messages_for_api(messages)
    try:
        r = requests.post(
            f"{BACKEND_URL.rstrip('/')}/chat",
            json={"message": user_message, "history": history_list},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        reply = data.get("response", "Something went wrong.")
    except requests.RequestException as e:
        reply = f"Couldn’t reach the assistant: {e}. Is the server running at {BACKEND_URL}?"
    except Exception as e:
        reply = f"Oops: {e}"

    new_messages = messages + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
    return "", new_messages


def create_ui():
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("Install Gradio: pip install gradio")

    # Gradio 6: Chatbot uses messages format (role/content); theme moved to launch()
    initial_messages = [{"role": "assistant", "content": PERSONALITY_GREETING}]
    with gr.Blocks(title="Glow – Makeup & Search Chat") as demo:
        gr.Markdown(f"# {PERSONALITY_NAME}\n*Your makeup and lifestyle assistant*")
        chatbot = gr.Chatbot(
            label="Chat",
            value=initial_messages,
            height=400,
        )
        msg = gr.Textbox(
            label="Message",
            placeholder="Ask about products, search by style, or anything else...",
            show_label=False,
            container=False,
        )
        submit = gr.Button("Send")

        def respond(message, messages):
            if not (message or message.strip()):
                return "", messages
            _, new_messages = chat_with_backend(message.strip(), messages)
            return "", new_messages

        submit.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

    return demo


if __name__ == "__main__":
    import gradio as gr
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
