#!/usr/bin/env python3
"""
Gradio app for EdgeFlow VLM chat with image upload and live metrics.

Features:
- Image upload (persistent across turns until changed)
- Chat UI with optional streaming
- Model picker fetched from /v1/models
- Metrics display: latency, tokens, tokens/sec

Usage:
  python scripts/gradio_vlm_chat.py --api http://localhost:8000 --share
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import requests
import re


DEFAULT_API: str = "http://localhost:8000"


@dataclass
class Metrics:
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    note: str = ""


def list_models(api_base: str) -> List[str]:
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/v1/models", timeout=15)
        data = resp.json()
        ids = [m["id"] for m in data.get("data", [])]
        return ids or ["qwen3-vl-4b"]
    except Exception:
        return ["qwen3-vl-4b"]


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


_ASSISTANT_HEADER_RE = re.compile(r"(?im)^\s*assistant\s*:?\s*$")
_ROLE_HEADER_FULL_LINE_RE = re.compile(r"(?im)^\s*(system|user|assistant)\s*:?\s*$")


def sanitize_assistant_text(text: str) -> str:
    """
    Remove any echoed chat role headers so only the assistant's response remains.
    Handles common formats like:
      system\n...\nuser\n...\nassistant\n<answer>
      Assistant: <answer>
    """
    if not text:
        return text

    # If transcript-like content exists, keep everything after the last 'assistant' header line
    matches = list(_ASSISTANT_HEADER_RE.finditer(text))
    if matches:
        return text[matches[-1].end():].strip()

    # Otherwise, drop any standalone role header lines and a leading "assistant:" prefix
    lines: List[str] = []
    for line in text.splitlines():
        if _ROLE_HEADER_FULL_LINE_RE.fullmatch(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"(?i)^\s*assistant\s*:\s*", "", cleaned).strip()
    return cleaned


def build_messages_from_chat(history: List[Tuple[str, str]], system_prompt: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for user_text, assistant_text in history:
        if user_text is not None:
            messages.append({"role": "user", "content": user_text})
        if assistant_text is not None and assistant_text != "":
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


def build_payload(
    model: str,
    messages: List[Dict[str, Any]],
    user_input: str,
    image_base64: Optional[str],
    image_mime: str,
    stream: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
) -> Dict[str, Any]:
    payload_messages = list(messages)
    user_msg: Dict[str, Any] = {"role": "user", "content": user_input}
    if image_base64:
        user_msg["attachments"] = [
            {"kind": "image", "data": image_base64, "mime_type": image_mime}
        ]
    payload_messages.append(user_msg)

    return {
        "model": model,
        "messages": payload_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }


def non_stream_chat(api_base: str, payload: Dict[str, Any]) -> Tuple[str, Metrics]:
    start = time.perf_counter()
    resp = requests.post(
        f"{api_base.rstrip('/')}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=600,
    )
    duration = time.perf_counter() - start

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    latency_ms = duration * 1000.0
    tokens_per_second = (completion_tokens / duration) if duration > 0 else 0.0

    return content, Metrics(
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tokens_per_second=tokens_per_second,
        note="from API usage",
    )


def stream_chat(api_base: str, payload: Dict[str, Any]) -> Generator[str, None, Metrics]:
    start = time.perf_counter()
    resp = requests.post(
        f"{api_base.rstrip('/')}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=600,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    accumulated: List[str] = []
    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data: "):
            continue
        data = text[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content_piece = delta.get("content")
        if content_piece:
            accumulated.append(content_piece)
            yield content_piece

    duration = time.perf_counter() - start
    full_text = "".join(accumulated)
    # Approximate token count by whitespace splitting when streaming (server does not send usage)
    approx_tokens = len(full_text.split())
    tps = (approx_tokens / duration) if duration > 0 else 0.0
    return Metrics(
        latency_ms=duration * 1000.0,
        prompt_tokens=0,
        completion_tokens=approx_tokens,
        total_tokens=approx_tokens,
        tokens_per_second=tps,
        note="approx (streaming)",
    )


def app_logic(
    api_base: str,
    model: str,
    system_prompt: str,
    user_input: str,
    chat_history: List[Tuple[str, str]],
    image_file: Optional[str],
    image_mime: str,
    stream: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
):
    messages = build_messages_from_chat(chat_history, system_prompt)
    image_b64: Optional[str] = None
    if image_file:
        try:
            from pathlib import Path

            img_bytes = Path(image_file).read_bytes()
            image_b64 = encode_image_to_base64(img_bytes)
        except Exception:
            image_b64 = None
    payload = build_payload(
        model=model,
        messages=messages,
        user_input=user_input,
        image_base64=image_b64,
        image_mime=image_mime,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    if not stream:
        reply, metrics = non_stream_chat(api_base, payload)
        reply = sanitize_assistant_text(reply)
        chat_history = chat_history + [(user_input, reply)]
        metrics_box = (
            f"Latency: {metrics.latency_ms:.1f} ms\n"
            f"Prompt tokens: {metrics.prompt_tokens}\n"
            f"Completion tokens: {metrics.completion_tokens}\n"
            f"Total tokens: {metrics.total_tokens}\n"
            f"Tokens/sec: {metrics.tokens_per_second:.2f} ({metrics.note})"
        )
        return chat_history, gr.update(value=""), metrics_box

    # Streaming path: yield incremental updates to Chatbot
    def stream_generator():
        partial = ""
        try:
            gen = stream_chat(api_base, payload)
            for piece in gen:
                partial += piece
                display = sanitize_assistant_text(partial)
                yield chat_history + [(user_input, display)], ""
            metrics: Metrics = gen.send(None)  # type: ignore[func-returns-value]
        except StopIteration as stop:
            metrics = stop.value  # type: ignore[assignment]
        except Exception as e:
            # Fallback to non-stream path if server doesn't support SSE
            reply, metrics = non_stream_chat(api_base, payload)
            reply = sanitize_assistant_text(reply)
            metrics.note = f"fallback: {metrics.note or ''}".strip()
            yield chat_history + [(user_input, reply)], ""

        metrics_box = (
            f"Latency: {metrics.latency_ms:.1f} ms\n"
            f"Prompt tokens: {metrics.prompt_tokens}\n"
            f"Completion tokens: {metrics.completion_tokens}\n"
            f"Total tokens: {metrics.total_tokens}\n"
            f"Tokens/sec: {metrics.tokens_per_second:.2f} ({metrics.note})"
        )
        yield chat_history + [(user_input, sanitize_assistant_text(partial))], "", metrics_box

    return stream_generator()


def build_ui(default_api: str, launch_share: bool):
    with gr.Blocks(title="EdgeFlow VLM Chat") as demo:
        gr.Markdown("""
        **EdgeFlow VLM Chat** — Upload an image and chat with the model. Metrics show latency and throughput.
        """)

        with gr.Row():
            api_base = gr.Textbox(label="API Base URL", value=default_api)
            refresh_models = gr.Button("Refresh Models", variant="secondary")
        _initial_models = list_models(default_api)
        _initial_value = _initial_models[0] if _initial_models else None
        model = gr.Dropdown(label="Model", choices=_initial_models, value=_initial_value)

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                top_k = gr.Slider(1, 100, value=40, step=1, label="Top-k")
                max_tokens = gr.Slider(16, 4096, value=512, step=16, label="Max new tokens")
            stream = gr.Checkbox(value=False, label="Stream output")
            fast_mode = gr.Checkbox(value=False, label="Fast mode (64 tokens, temp 0.2)")
            max_image_side = gr.Slider(256, 2048, value=640, step=64, label="Max image side (px)")
            system_prompt = gr.Textbox(label="System Prompt", value="You are a helpful vision assistant.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                with gr.Row():
                    user_input = gr.Textbox(placeholder="Ask about the image...", scale=4)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat")
            with gr.Column(scale=2):
                image = gr.Image(type="filepath", sources=["upload", "clipboard"], label="Image")
                image_mime = gr.Radio(["image/jpeg", "image/png"], value="image/jpeg", label="MIME type")
                metrics_box = gr.Textbox(label="Metrics", value="", lines=8)

        def do_refresh_models(api_text: str):
            ids = list_models(api_text)
            return gr.update(choices=ids, value=(ids[0] if ids else None))

        refresh_models.click(do_refresh_models, inputs=[api_base], outputs=[model])

        def on_clear():
            return [], "", ""

        clear_btn.click(on_clear, outputs=[chatbot, user_input, metrics_box])

        def apply_fast_mode(stream_val, fast_val, temp, max_new, img_side):
            if fast_val:
                return True if stream_val else stream_val, 0.2, 64, img_side
            return stream_val, temp, max_new, img_side

        send_btn.click(
            fn=apply_fast_mode,
            inputs=[stream, fast_mode, temperature, max_tokens, max_image_side],
            outputs=[stream, temperature, max_tokens, max_image_side],
        ).then(
            fn=app_logic,
            inputs=[
                api_base,
                model,
                system_prompt,
                user_input,
                chatbot,
                image,
                image_mime,
                stream,
                temperature,
                max_tokens,
                top_p,
                top_k,
            ],
            outputs=[chatbot, user_input, metrics_box],
        )

    demo.queue(api_open=False).launch(share=launch_share)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio VLM chat app")
    parser.add_argument("--api", default=DEFAULT_API, help="API base URL")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    build_ui(args.api, args.share)


if __name__ == "__main__":
    main()


