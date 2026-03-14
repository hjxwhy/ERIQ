"""
ERIQ Benchmark Viewer
A Gradio web app to browse all 15 ERIQ subtasks entry-by-entry,
showing all images and the full conversation for each entry.

Usage:
    python eval_code/viewer.py
    python eval_code/viewer.py --data_root /home/unitree/remote_jensen2/ERIQ/ --port 7860
"""

import argparse
import json
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

# Global state: cache loaded task data to avoid re-reading on every navigation
_cache: dict[str, list] = {}
DATA_ROOT: str = ""


# ──────────────────────────────────────────────────────────────────────────── #
# Data helpers
# ──────────────────────────────────────────────────────────────────────────── #

def load_task(task_name: str) -> list:
    """Load and cache a task JSON. Returns list of entries."""
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
    """Return list of absolute image file paths for an entry."""
    raw = entry.get("image", [])
    paths = [raw] if isinstance(raw, str) else list(raw)
    abs_paths = []
    for p in paths:
        full = os.path.join(DATA_ROOT, "images", p)
        if os.path.exists(full):
            abs_paths.append(full)
    return abs_paths


def format_conversation(entry: dict) -> str:
    """Render the conversation as clean Markdown."""
    lines = []
    convs = entry.get("conversations", [])
    for turn in convs:
        speaker = turn.get("from", "")
        text = turn.get("value", "")
        # Strip <image> tokens
        text = re.sub(r"<image>\n?", "", text).strip()
        if speaker == "human":
            lines.append("### 🧑 Question\n")
            lines.append(text)
        elif speaker == "gpt":
            lines.append("\n---\n### ✅ Ground Truth Answer\n")
            lines.append(f"**{text}**")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────── #
# Gradio event handlers
# ──────────────────────────────────────────────────────────────────────────── #

def on_task_change(task_name: str):
    """Called when the user selects a new task. Resets to entry 0."""
    data = load_task(task_name)
    n = len(data)
    if n == 0:
        return (
            gr.update(value=0, minimum=0, maximum=0, step=1, label="Entry (0 / 0)"),
            [],
            "*No data found for this task.*",
            "",
        )
    entry = data[0]
    images = get_image_paths(entry)
    conv_md = format_conversation(entry)
    entry_id = entry.get("id", "—")
    return (
        gr.update(value=0, minimum=0, maximum=n - 1, step=1, label=f"Entry (1 / {n})"),
        images,
        conv_md,
        f"**ID:** `{entry_id}`   |   **Images:** {len(images)}",
    )


def on_index_change(task_name: str, idx: int):
    """Called when the slider value changes."""
    data = load_task(task_name)
    n = len(data)
    if n == 0 or idx < 0 or idx >= n:
        return [], "*No entry.*", "", gr.update(label="Entry")
    entry = data[idx]
    images = get_image_paths(entry)
    conv_md = format_conversation(entry)
    entry_id = entry.get("id", "—")
    return (
        images,
        conv_md,
        f"**ID:** `{entry_id}`   |   **Images:** {len(images)}",
        gr.update(label=f"Entry ({idx + 1} / {n})"),
    )


def on_prev(task_name: str, idx: int):
    data = load_task(task_name)
    new_idx = max(0, idx - 1)
    return (new_idx,) + on_index_change(task_name, new_idx)


def on_next(task_name: str, idx: int):
    data = load_task(task_name)
    new_idx = min(len(data) - 1, idx + 1)
    return (new_idx,) + on_index_change(task_name, new_idx)


def on_jump(task_name: str, jump_str: str):
    """Jump to a specific entry ID or 1-based index."""
    data = load_task(task_name)
    if not data:
        return 0, [], "*No data.*", "", gr.update()

    # Try matching by ID string first
    jump_str = jump_str.strip()
    for i, entry in enumerate(data):
        if entry.get("id", "") == jump_str:
            return (i,) + on_index_change(task_name, i)

    # Try 1-based numeric index
    try:
        num = int(jump_str)
        idx = max(0, min(len(data) - 1, num - 1))
        return (idx,) + on_index_change(task_name, idx)
    except ValueError:
        pass

    return 0, [], f"*Entry `{jump_str}` not found.*", "", gr.update()


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
            task_dd = gr.Dropdown(
                choices=ALL_TASKS,
                value=ALL_TASKS[0],
                label="Task",
                scale=3,
            )
            jump_box = gr.Textbox(
                placeholder="Jump to ID or index (e.g. QA_ACTION_UNDERSTANDING:42 or 42)",
                label="Jump to entry",
                scale=3,
            )
            jump_btn = gr.Button("Go", scale=1, variant="secondary")

        with gr.Row():
            prev_btn = gr.Button("◀  Prev", scale=1)
            idx_slider = gr.Slider(
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                label="Entry (– / –)",
                scale=8,
                interactive=True,
            )
            next_btn = gr.Button("Next  ▶", scale=1)

        entry_info = gr.Markdown("", elem_id="entry-info")

        with gr.Row(equal_height=True):
            gallery = gr.Gallery(
                label="Images",
                columns=3,
                height=520,
                object_fit="contain",
                allow_preview=True,
                type="filepath",
                scale=3,
            )
            conv_panel = gr.Markdown(
                label="Conversation",
                value="",
                elem_id="conv-panel",
            )

        # ── Event wiring ─────────────────────────────────────────────────── #

        # Task change → reset to entry 0
        task_dd.change(
            fn=on_task_change,
            inputs=[task_dd],
            outputs=[idx_slider, gallery, conv_panel, entry_info],
        )

        # Slider drag / type
        idx_slider.release(
            fn=on_index_change,
            inputs=[task_dd, idx_slider],
            outputs=[gallery, conv_panel, entry_info, idx_slider],
        )

        # Prev / Next buttons
        prev_btn.click(
            fn=on_prev,
            inputs=[task_dd, idx_slider],
            outputs=[idx_slider, gallery, conv_panel, entry_info, idx_slider],
        )
        next_btn.click(
            fn=on_next,
            inputs=[task_dd, idx_slider],
            outputs=[idx_slider, gallery, conv_panel, entry_info, idx_slider],
        )

        # Jump
        jump_btn.click(
            fn=on_jump,
            inputs=[task_dd, jump_box],
            outputs=[idx_slider, gallery, conv_panel, entry_info, idx_slider],
        )
        jump_box.submit(
            fn=on_jump,
            inputs=[task_dd, jump_box],
            outputs=[idx_slider, gallery, conv_panel, entry_info, idx_slider],
        )

        # Load initial task on startup
        demo.load(
            fn=on_task_change,
            inputs=[task_dd],
            outputs=[idx_slider, gallery, conv_panel, entry_info],
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────── #
# Entry point
# ──────────────────────────────────────────────────────────────────────────── #

def parse_args():
    parser = argparse.ArgumentParser(description="ERIQ Benchmark Viewer")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/unitree/remote_jensen2/ERIQ/",
        help="Root directory of the ERIQ dataset",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve the Gradio app on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_ROOT = args.data_root

    # Verify data root exists
    if not os.path.isdir(DATA_ROOT):
        print(f"[ERROR] Data root not found: {DATA_ROOT}")
        raise SystemExit(1)

    print(f"Data root : {DATA_ROOT}")
    print(f"Port      : {args.port}")
    print(f"Tasks     : {len(ALL_TASKS)}")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
