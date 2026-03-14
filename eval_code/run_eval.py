"""
ERIQ Benchmark Evaluation Script
Runs inference with Qwen2.5-VL or Qwen3-VL on all 15 ERIQ subtasks and saves
results in the required format (with 'prediction' and 'gt_answer' fields).

Supported model families (auto-detected from config.json):
  - Qwen2.5-VL  (model_type: qwen2_5_vl)
  - Qwen3-VL    (model_type: qwen3_vl)
  - Qwen3-VL-MoE (model_type: qwen3_vl_moe)

Usage:
    python eval_code/run_eval.py \
        --model_path /home/unitree/remote_jensen/huangjianxin/transfer2ali/Unifolm-VLM-Base/ \
        --data_root /home/unitree/remote_jensen2/ERIQ/ \
        --output_dir results/

    # Then score:
    python eval_code/eval_hf.py results/all_results.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Map model_type string -> model class
_MODEL_TYPE_MAP = {
    "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3_vl": Qwen3VLForConditionalGeneration,
    "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration,
}


def get_model_class(model_path: str):
    """
    Detect the correct model class by reading config.json from the model directory.
    Falls back to Qwen2_5_VLForConditionalGeneration if the type is unrecognised.
    """
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        if model_type in _MODEL_TYPE_MAP:
            cls = _MODEL_TYPE_MAP[model_type]
            print(f"Detected model_type='{model_type}' → using {cls.__name__}")
            return cls
        print(f"[WARNING] Unknown model_type='{model_type}', falling back to Qwen2_5_VLForConditionalGeneration")
    else:
        print("[WARNING] config.json not found, falling back to Qwen2_5_VLForConditionalGeneration")
    return Qwen2_5_VLForConditionalGeneration

# All 15 ERIQ subtask JSON filenames
ALL_TASKS = [
    "QA_ACTION_UNDERSTANDING.json",
    "QA_HUMAN_INTENTION.json",
    "QA_HUMAN_INTERACTION.json",
    "QA_MISTAKE_RECOVERY.json",
    "QA_MISTAKE_CLASSIFY.json",
    "QA_MISTAKE_EXISTENCE.json",
    "QA_SUBTASK_PLANNING.json",
    "QA_SUCCESS_DETECTION.json",
    "QA_SCENE_UNDERSTANDING.json",
    "QA_TASK_PROGRESS.json",
    "QA_TRAJ_UNDERSTANDING.json",
    "QA_DUALVIEW_MATCHING.json",
    "QA_TASK_GROUNDING.json",
    "QA_RELATIVE_POS_GROUNDING.json",
    "QA_FINE_GRAINED_PLAN.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="ERIQ Benchmark Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/unitree/remote_jensen/huangjianxin/transfer2ali/Unifolm-VLM-Base/",
        help="Path to the Qwen2.5-VL model directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/unitree/remote_jensen2/ERIQ/",
        help="Root directory of the ERIQ dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save prediction JSON files",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific task JSON filenames to evaluate (default: all 15 tasks). "
             "Example: --tasks QA_ACTION_UNDERSTANDING.json QA_MISTAKE_EXISTENCE.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum pixels for image processing",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum pixels for image processing",
    )
    return parser.parse_args()


def build_messages(entry: dict, image_root: str) -> list:
    """
    Build Qwen2.5-VL chat messages from a single ERIQ entry.

    The conversations[0]['value'] contains <image> tokens interleaved with text.
    We replace each <image> token with an image content block pointing to the
    corresponding image file, then append the remaining text as a text block.
    """
    human_turn = entry["conversations"][0]["value"]
    raw_image = entry.get("image", [])
    # 'image' can be a bare string (single-image tasks) or a list (multi-image/video tasks)
    image_paths = [raw_image] if isinstance(raw_image, str) else raw_image

    # Split the human turn on <image> tokens to interleave images and text
    parts = human_turn.split("<image>")

    content = []
    image_idx = 0

    for i, part in enumerate(parts):
        # Each split part before index i was preceded by an <image> token
        # (except the first part which has no preceding <image>)
        if i > 0 and image_idx < len(image_paths):
            abs_image_path = os.path.join(image_root, "images", image_paths[image_idx])
            content.append({"type": "image", "image": abs_image_path})
            image_idx += 1

        # Add text part (strip leading/trailing newlines but keep inner content)
        stripped = part.strip("\n") + "Please answer directly with only the letter of the correct option and nothing else."
        if stripped:
            content.append({"type": "text", "text": stripped})

    # If there are more images than <image> tokens (shouldn't happen but be safe)
    while image_idx < len(image_paths):
        abs_image_path = os.path.join(image_root, "images", image_paths[image_idx])
        content.append({"type": "image", "image": abs_image_path})
        image_idx += 1

    messages = [{"role": "user", "content": content}]
    return messages


def run_inference_batch(
    model,
    processor,
    entries: list,
    image_root: str,
    max_new_tokens: int,
) -> list:
    """
    Run inference on a batch of entries. Returns list of decoded prediction strings.
    """
    all_messages = [build_messages(e, image_root) for e in entries]

    # Apply chat template to get text inputs
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    # Process vision info (images) for each message
    image_inputs_list = []
    video_inputs_list = []
    for msgs in all_messages:
        image_inputs, video_inputs = process_vision_info(msgs)
        image_inputs_list.append(image_inputs)
        video_inputs_list.append(video_inputs)

    # For batch_size > 1 we need to flatten image lists; for batch_size=1 it's straightforward
    # Flatten for the processor call
    flat_images = []
    for imgs in image_inputs_list:
        if imgs:
            flat_images.extend(imgs)

    flat_videos = []
    for vids in video_inputs_list:
        if vids:
            flat_videos.extend(vids)

    inputs = processor(
        text=texts,
        images=flat_images if flat_images else None,
        videos=flat_videos if flat_videos else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
        )

    # Trim prompt tokens from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    predictions = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return predictions


def evaluate_task(
    task_json_path: str,
    image_root: str,
    model,
    processor,
    batch_size: int,
    max_new_tokens: int,
) -> list:
    """
    Load a task JSON, run inference on all entries, and return results list.
    """
    with open(task_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_name = Path(task_json_path).stem
    print(f"\n{'='*60}")
    print(f"Task: {task_name}  ({len(data)} samples)")
    print(f"{'='*60}")

    results = []

    for batch_start in tqdm(range(0, len(data), batch_size), desc=task_name):
        batch = data[batch_start : batch_start + batch_size]

        try:
            predictions = run_inference_batch(
                model, processor, batch, image_root, max_new_tokens
            )
        except Exception as e:
            print(f"\n[WARNING] Inference failed for batch starting at {batch_start}: {e}")
            predictions = [""] * len(batch)

        for entry, pred in zip(batch, predictions):
            # Extract gt_answer from conversations (second turn, from gpt)
            gt_answer = ""
            if len(entry.get("conversations", [])) >= 2:
                gt_answer = entry["conversations"][1]["value"].strip()

            result_entry = dict(entry)  # copy all original fields
            result_entry["prediction"] = pred.strip()
            result_entry["gt_answer"] = gt_answer
            results.append(result_entry)

    return results


def main():
    args = parse_args()

    # Resolve output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which tasks to run
    task_filenames = args.tasks if args.tasks else ALL_TASKS
    # Normalize: allow task names without .json extension
    task_filenames = [
        t if t.endswith(".json") else t + ".json" for t in task_filenames
    ]

    print("=" * 60)
    print("ERIQ Benchmark Evaluation")
    print("=" * 60)
    print(f"Model path : {args.model_path}")
    print(f"Data root  : {args.data_root}")
    print(f"Output dir : {output_dir}")
    print(f"Tasks      : {len(task_filenames)} task(s)")
    print(f"Batch size : {args.batch_size}")
    print(f"Max tokens : {args.max_new_tokens}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Check if all tasks are already done (skip model loading entirely)
    # ------------------------------------------------------------------ #
    def _is_complete(task_output_path):
        """Return True only if the file exists AND every entry has a non-empty prediction."""
        if not task_output_path.exists():
            return False
        with open(task_output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) > 0 and all(x.get("prediction", "").strip() for x in data)

    pending = [
        t for t in task_filenames
        if not _is_complete(output_dir / f"{Path(t).stem}.json")
        and os.path.exists(os.path.join(args.data_root, t))
    ]
    if not pending:
        print("\nAll tasks already completed. Skipping model loading.")
        # Still need to merge & score existing results
        all_results = []
        for task_filename in task_filenames:
            task_output_path = output_dir / f"{Path(task_filename).stem}.json"
            if task_output_path.exists():
                with open(task_output_path, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
        all_results_path = output_dir / "all_results.json"
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"Merged {len(all_results)} results → {all_results_path}")
        eval_script = Path(__file__).parent / "eval_hf.py"
        if eval_script.exists():
            import subprocess
            subprocess.run([sys.executable, str(eval_script), str(all_results_path)], check=False)
        return

    # ------------------------------------------------------------------ #
    # Load model & processor
    # ------------------------------------------------------------------ #
    print("\nLoading model and processor...")
    ModelClass = get_model_class(args.model_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = ModelClass.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully.\n")

    # ------------------------------------------------------------------ #
    # Run evaluation per task
    # ------------------------------------------------------------------ #
    all_results = []

    for task_filename in task_filenames:
        task_name = Path(task_filename).stem
        task_output_path = output_dir / f"{task_name}.json"

            # ── Skip already-completed tasks ──────────────────────────────── #
        if task_output_path.exists():
            with open(task_output_path, "r", encoding="utf-8") as f:
                task_results = json.load(f)
            filled = sum(1 for x in task_results if x.get("prediction", "").strip())
            if filled == len(task_results) and len(task_results) > 0:
                print(f"\n[SKIP] {task_name} — {filled}/{len(task_results)} predictions complete.")
                all_results.extend(task_results)
                continue
            else:
                print(f"\n[REDO] {task_name} — only {filled}/{len(task_results)} predictions filled, re-running.")

        task_json_path = os.path.join(args.data_root, task_filename)
        if not os.path.exists(task_json_path):
            print(f"[WARNING] Task file not found, skipping: {task_json_path}")
            continue

        task_results = evaluate_task(
            task_json_path=task_json_path,
            image_root=args.data_root,
            model=model,
            processor=processor,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

        # Save per-task results
        with open(task_output_path, "w", encoding="utf-8") as f:
            json.dump(task_results, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(task_results)} results to {task_output_path}")

        all_results.extend(task_results)

    # ------------------------------------------------------------------ #
    # Save merged results
    # ------------------------------------------------------------------ #
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\nAll results saved to {all_results_path} ({len(all_results)} total entries)")

    # ------------------------------------------------------------------ #
    # Quick scoring summary
    # ------------------------------------------------------------------ #
    print("\nRunning evaluation...")
    eval_script = Path(__file__).parent / "eval_hf.py"
    if eval_script.exists():
        import subprocess
        subprocess.run(
            [sys.executable, str(eval_script), str(all_results_path)],
            check=False,
        )
    else:
        print(f"Evaluation script not found at {eval_script}. Run manually:")
        print(f"  python eval_code/eval_hf.py {all_results_path}")


if __name__ == "__main__":
    main()
