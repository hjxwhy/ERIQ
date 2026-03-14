"""
ERIQ Benchmark Viewer
A Gradio web app to browse all 15 ERIQ subtasks entry-by-entry,
showing all images and the full conversation for each entry.

Usage:
    python eval_code/viewer.py
    python eval_code/viewer.py --data_root /home/unitree/remote_jensen2/ERIQ/ --port 7860
"""

import argparse
import base64
import json
import math
import os
import re
from pathlib import Path

import gradio as gr

# ──────────────────────────────────────────────────────────────────────────── #
# Constants
# ──────────────────────────────────────────────────────────────────────────── #

ALL_TASKS = [
    "QA_ACTION_UNDERSTANDING",
    "QA_HUMAN_INTENTION",
    "QA_HUMAN_INTERACTION",
    "QA_MISTAKE_RECOVERY",
    "QA_MISTAKE_CLASSIFY",
    "QA_MISTAKE_EXISTENCE",
    "QA_SUBTASK_PLANNING",
    "QA_SUCCESS_DETECTION",
    "QA_SCENE_UNDERSTANDING",
    "QA_TASK_PROGRESS",
    "QA_TRAJ_UNDERSTANDING",
    "QA_DUALVIEW_MATCHING",
    "QA_TASK_GROUNDING",
    "QA_RELATIVE_POS_GROUNDING",
    "QA_FINE_GRAINED_PLAN",
]

# Global state
_cache: dict[str, list] = {}
DATA_ROOT: str = ""


# ──────────────────────────────────────────────────────────────────────────── #
# Data helpers
# ──────────────────────────────────────────────────────────────────────────── #

def load_task(task_name: str) -> list:
    if task_name in _cache:
        return _cache[task_name]
    json_path = os.path.join(DATA_ROOT, f"{task_name}.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _cache[task_name] = data
    return data


def get_image_paths(entry: dict) -> list[str]:
    raw = entry.get("image", [])
    paths = [raw] if isinstance(raw, str) else list(raw)
    return [
        os.path.join(DATA_ROOT, "images", p)
        for p in paths
        if os.path.exists(os.path.join(DATA_ROOT, "images", p))
    ]


def image_to_b64(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def images_to_html(image_paths: list[str]) -> str:
    """Render images as a scrollable CSS-grid HTML block."""
    if not image_paths:
        return "<p style='color:#aaa;padding:16px'>No images for this entry.</p>"

    n = len(image_paths)
    cols = 1 if n == 1 else (2 if n <= 4 else (3 if n <= 6 else 4))

    # Per-image max-height: single image stays in viewport, multi-image is compact
    if n == 1:
        img_max_h = "560px"
    elif n <= 4:
        img_max_h = "380px"
    else:
        img_max_h = "280px"

    cells = ""
    for path in image_paths:
        try:
            src = image_to_b64(path)
        except Exception:
            src = ""
        fname = Path(path).name
        cells += (
            f'<div style="background:#111;border-radius:6px;overflow:hidden;'
            f'display:flex;align-items:center;justify-content:center">'
            f'<img src="{src}" title="{fname}" '
            f'style="max-width:100%;max-height:{img_max_h};'
            f'height:auto;display:block;object-fit:contain"/>'
            f"</div>"
        )

    return (
        f'<div style="'
        f"display:grid;"
        f"grid-template-columns:repeat({cols},1fr);"
        f"gap:8px;"
        f"max-height:640px;"
        f"overflow-y:auto;"
        f"padding:8px;"
        f"background:#1a1a2e;"
        f'border-radius:8px">'
        f"{cells}"
        f"</div>"
    )


def format_conversation(entry: dict) -> str:
    lines = []
    for turn in entry.get("conversations", []):
        speaker = turn.get("from", "")
        text = re.sub(r"<image>\n?", "", turn.get("value", "")).strip()
        if speaker == "human":
            lines.append("### 🧑 Question\n\n" + text)
        elif speaker == "gpt":
            lines.append(f"\n---\n### ✅ Ground Truth Answer\n\n**{text}**")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────── #
# Event handlers
# ──────────────────────────────────────────────────────────────────────────── #

def _entry_outputs(data: list, idx: int, n: int):
    entry = data[idx]
    images = get_image_paths(entry)
    html = images_to_html(images)
    conv_md = format_conversation(entry)
    entry_id = entry.get("id", "—")
    info = f"**ID:** `{entry_id}`   |   **Images:** {len(images)}"
    slider_label = gr.update(label=f"Entry ({idx + 1} / {n})")
    return html, conv_md, info, slider_label


def on_task_change(task_name: str):
    data = load_task(task_name)
    n = len(data)
    empty_html = "<p style='color:#aaa;padding:16px'>No data found.</p>"
    if n == 0:
        return (
            gr.update(value=0, minimum=0, maximum=0, step=1, label="Entry (0 / 0)"),
            empty_html, "", "",
        )
    slider_upd = gr.update(value=0, minimum=0, maximum=n - 1, step=1, label=f"Entry (1 / {n})")
    html, conv_md, info, _ = _entry_outputs(data, 0, n)
    return slider_upd, html, conv_md, info


def on_index_change(task_name: str, idx: int):
    data = load_task(task_name)
    n = len(data)
    if n == 0 or idx < 0 or idx >= n:
        return "<p style='color:#aaa'>No entry.</p>", "", "", gr.update()
    return _entry_outputs(data, int(idx), n)


def on_prev(task_name: str, idx: int):
    data = load_task(task_name)
    new_idx = max(0, int(idx) - 1)
    return (new_idx,) + on_index_change(task_name, new_idx)


def on_next(task_name: str, idx: int):
    data = load_task(task_name)
    new_idx = min(len(data) - 1, int(idx) + 1)
    return (new_idx,) + on_index_change(task_name, new_idx)


def on_jump(task_name: str, jump_str: str):
    data = load_task(task_name)
    empty = "<p style='color:#aaa'>Not found.</p>"
    if not data:
        return 0, empty, "", "", gr.update()
    jump_str = jump_str.strip()
    for i, entry in enumerate(data):
        if entry.get("id", "") == jump_str:
            return (i,) + on_index_change(task_name, i)
    try:
        num = int(jump_str)
        idx = max(0, min(len(data) - 1, num - 1))
        return (idx,) + on_index_change(task_name, idx)
    except ValueError:
        pass
    return 0, empty, f"*Entry `{jump_str}` not found.*", "", gr.update()


# ──────────────────────────────────────────────────────────────────────────── #
# UI
# ──────────────────────────────────────────────────────────────────────────── #

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ERIQ Benchmark Viewer", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# 🤖 ERIQ Benchmark Viewer\n"
            "Browse all 15 subtasks of the ERIQ embodied reasoning benchmark."
        )

        with gr.Row():
            task_dd = gr.Dropdown(choices=ALL_TASKS, value=ALL_TASKS[0], label="Task", scale=3)
            jump_box = gr.Textbox(
                placeholder="Jump to ID or 1-based index  (e.g. QA_ACTION_UNDERSTANDING:42 or 42)",
                label="Jump to entry", scale=3,
            )
            jump_btn = gr.Button("Go", scale=1, variant="secondary")

        with gr.Row():
            prev_btn = gr.Button("◀  Prev", scale=1)
            idx_slider = gr.Slider(
                minimum=0, maximum=0, step=1, value=0,
                label="Entry (– / –)", scale=8, interactive=True,
            )
            next_btn = gr.Button("Next  ▶", scale=1)

        entry_info = gr.Markdown("")

        # Images rendered as scrollable HTML grid
        img_html = gr.HTML(label="Images")

        conv_panel = gr.Markdown(label="Conversation", value="")

        # ── Event wiring ─────────────────────────────────────────────────── #
        OUTPUTS = [img_html, conv_panel, entry_info, idx_slider]

        task_dd.change(
            fn=on_task_change,
            inputs=[task_dd],
            outputs=[idx_slider, img_html, conv_panel, entry_info],
        )
        idx_slider.release(
            fn=on_index_change,
            inputs=[task_dd, idx_slider],
            outputs=OUTPUTS,
        )
        prev_btn.click(
            fn=on_prev,
            inputs=[task_dd, idx_slider],
            outputs=[idx_slider] + OUTPUTS,
        )
        next_btn.click(
            fn=on_next,
            inputs=[task_dd, idx_slider],
            outputs=[idx_slider] + OUTPUTS,
        )
        jump_btn.click(
            fn=on_jump,
            inputs=[task_dd, jump_box],
            outputs=[idx_slider] + OUTPUTS,
        )
        jump_box.submit(
            fn=on_jump,
            inputs=[task_dd, jump_box],
            outputs=[idx_slider] + OUTPUTS,
        )
        demo.load(
            fn=on_task_change,
            inputs=[task_dd],
            outputs=[idx_slider, img_html, conv_panel, entry_info],
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────── #
# Entry point
# ──────────────────────────────────────────────────────────────────────────── #

def parse_args():
    parser = argparse.ArgumentParser(description="ERIQ Benchmark Viewer")
    parser.add_argument("--data_root", type=str,
                        default="/home/unitree/remote_jensen2/ERIQ/",
                        help="Root directory of the ERIQ dataset")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_ROOT = args.data_root

    if not os.path.isdir(DATA_ROOT):
        print(f"[ERROR] Data root not found: {DATA_ROOT}")
        raise SystemExit(1)

    print(f"Data root : {DATA_ROOT}")
    print(f"Port      : {args.port}")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
