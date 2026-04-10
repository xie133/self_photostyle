#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Score generated images against the PhotoStyle aesthetic and rank them.
"""

from __future__ import annotations

import argparse
import importlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "generated_images"
DATASET_FILE = BASE_DIR / "photo_style_dataset.jsonl"
LOG_DIR = BASE_DIR / "logs"
PHOTO_DIR = BASE_DIR / "my_photos"
MODEL_NAME = "gemini-2.5-flash"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_TOP_K_STYLE = 5


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[setup] Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pip_name])


def ensure_dependencies() -> None:
    ensure_package("google.genai", "google-genai")


ensure_dependencies()

from google import genai
from google.genai import types


def load_style_outputs(path: Path) -> List[str]:
    outputs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            output = str(row.get("output", "")).strip()
            if output:
                outputs.append(output)
    if not outputs:
        raise ValueError("No valid style descriptions found in dataset.")
    return outputs


def iter_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder.resolve()}")
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def latest_generation_log() -> Path | None:
    candidates = sorted(LOG_DIR.glob("generation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_images_from_generation_log(log_path: Path) -> List[Path]:
    data = json.loads(log_path.read_text(encoding="utf-8"))
    image_paths: List[Path] = []
    for row in data:
        image_path = Path(str(row.get("image_path", "")))
        if image_path.exists():
            image_paths.append(image_path)
    return image_paths


def read_image_bytes(image_path: Path) -> bytes:
    return image_path.read_bytes()


def build_score_prompt(style_texts: List[str]) -> str:
    style_block = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(style_texts))
    return (
        "\u4f60\u73b0\u5728\u662f\u6211\u7684\u4e2a\u4eba\u5ba1\u7f8e\u8bc4\u5ba1\u5458\u3002\u8bf7\u6839\u636e\u4ee5\u4e0b\u98ce\u683c\u5e93\uff0c"
        "\u5224\u65ad\u8fd9\u5f20\u56fe\u7247\u662f\u5426\u7b26\u5408\u6211\u7684 PhotoStyle \u5ba1\u7f8e\u3002\n\n"
        "\u8bc4\u5ba1\u91cd\u70b9\uff1a\n"
        "- \u662f\u5426\u6709\u4e71\u4e2d\u6709\u5e8f\u7684\u8857\u5934\u590d\u6742\u611f\n"
        "- \u4e3b\u4f53\u662f\u5426\u6e05\u6670\u7a81\u51fa\n"
        "- \u662f\u5426\u5177\u6709\u771f\u5b9e\u6444\u5f71\u611f\u800c\u4e0d\u662f\u63d2\u753b\u611f\u6216\u6d77\u62a5\u611f\n"
        "- \u5149\u5f71\u3001\u6750\u8d28\u3001\u7a7a\u95f4\u548c\u52a8\u4f5c\u662f\u5426\u53ef\u4fe1\n"
        "- \u662f\u5426\u6709\u751f\u6d3b\u6c14\u606f\u3001\u73b0\u573a\u611f\u548c\u547c\u5438\u611f\n\n"
        "\u53c2\u8003\u98ce\u683c\u5e93\uff1a\n"
        f"{style_block}\n\n"
        "\u8bf7\u53ea\u8fd4\u56de JSON\uff0c\u683c\u5f0f\u5982\u4e0b\uff1a\n"
        "{\"score\": 0-100, \"verdict\": \"fit|partial|miss\", \"reason\": \"\u4e00\u53e5\u4e2d\u6587\u77ed\u8bc4\"}"
    )


def score_image(client: genai.Client, image_path: Path, style_texts: List[str]) -> dict:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            build_score_prompt(style_texts),
            types.Part.from_bytes(data=read_image_bytes(image_path), mime_type=mime_type or "image/png"),
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    text = (response.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid scorer JSON for {image_path.name}: {text}") from exc

    return {
        "image_path": str(image_path.resolve()),
        "filename": image_path.name,
        "score": int(payload.get("score", 0)),
        "verdict": str(payload.get("verdict", "unknown")),
        "reason": str(payload.get("reason", "")).strip(),
    }


def print_ranked_results(results: List[dict]) -> None:
    print("\nRanked results:")
    for index, result in enumerate(results, start=1):
        print(
            f"[{index}] {result['score']:>3} | {result['verdict']:<7} | "
            f"{result['filename']} | {result['reason']}"
        )


def parse_selection(raw: str, max_index: int) -> List[int]:
    if not raw.strip():
        return []

    normalized = raw.strip().lower()
    if normalized in {"none", "no", "n", "bad"}:
        return []
    if normalized == "both" and max_index >= 2:
        return [1, 2]

    picked: List[int] = []
    seen = set()
    for chunk in raw.replace(" ", "").split(","):
        if not chunk:
            continue
        value = int(chunk)
        if value < 1 or value > max_index:
            raise ValueError(f"Selection {value} is out of range 1-{max_index}.")
        if value not in seen:
            seen.add(value)
            picked.append(value)
    return picked


def copy_selected_images(results: List[dict], selected_indices: List[int], destination: Path) -> List[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied_paths: List[Path] = []

    for index in selected_indices:
        source = Path(results[index - 1]["image_path"])
        stem = source.stem
        suffix = source.suffix
        candidate = destination / source.name
        counter = 1

        while candidate.exists():
            candidate = destination / f"{stem}_selected_{counter}{suffix}"
            counter += 1

        shutil.copy2(source, candidate)
        copied_paths.append(candidate)

    return copied_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated images against your distilled photo style.")
    parser.add_argument(
        "--dir",
        default=str(IMAGE_DIR),
        help="Directory containing generated images to score.",
    )
    parser.add_argument(
        "--top-k-style",
        type=int,
        default=DEFAULT_TOP_K_STYLE,
        help="How many style descriptions from the dataset to use for scoring guidance.",
    )
    parser.add_argument(
        "--latest-batch",
        action="store_true",
        help="Score only the most recent generated batch from logs/ instead of the whole image directory.",
    )
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY. Please set it before running.")

    style_texts = load_style_outputs(DATASET_FILE)[: max(1, args.top_k_style)]
    if args.latest_batch:
        log_path = latest_generation_log()
        if log_path is None:
            raise FileNotFoundError("No generation logs found in logs/.")
        image_paths = load_images_from_generation_log(log_path)
        if not image_paths:
            raise FileNotFoundError(f"No existing images referenced by {log_path.resolve()}.")
        print(f"Scoring latest batch from: {log_path.resolve()}")
    else:
        image_paths = iter_images(Path(args.dir))

    client = genai.Client(api_key=api_key)
    results = []
    for image_path in image_paths:
        try:
            result = score_image(client, image_path, style_texts)
            results.append(result)
            print(f"{result['score']:>3} | {result['verdict']:<7} | {image_path.name} | {result['reason']}")
        except Exception as exc:
            print(f"[warn] Failed to score {image_path.name}: {exc}")

    results.sort(key=lambda item: item["score"], reverse=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / f"score_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved score report: {output_path.resolve()}")

    if not results:
        return

    print_ranked_results(results)
    print("\nChoose images to add to my_photos/ for retraining.")
    if len(results) >= 2:
        print("Type: 1, 2, both, none, or comma-separated indices such as 1,3")
    else:
        print("Type: 1 or none")

    raw = input("Selection: ")
    try:
        selected_indices = parse_selection(raw, len(results))
    except ValueError as exc:
        raise ValueError(f"Invalid selection: {exc}") from exc

    if not selected_indices:
        print("No images selected.")
        return

    copied_paths = copy_selected_images(results, selected_indices, PHOTO_DIR)
    print("\nCopied to my_photos/:")
    for path in copied_paths:
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] {exc}")
        sys.exit(1)
