#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Score generated images against the PhotoStyle aesthetic, collect structured
feedback, and feed liked images back into my_photos/.
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "generated_images"
DATASET_FILE = BASE_DIR / "photo_style_dataset.jsonl"
LOG_DIR = BASE_DIR / "logs"
PHOTO_DIR = BASE_DIR / "my_photos"
FEEDBACK_LOG_FILE = LOG_DIR / "feedback_history.jsonl"
MODEL_NAME = "gemini-2.5-flash"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_TOP_K_STYLE = 5
FEEDBACK_LEVELS = ("like", "okay", "dislike")


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
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


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


def load_generation_metadata() -> Dict[str, dict]:
    metadata: Dict[str, dict] = {}
    for log_path in sorted(LOG_DIR.glob("generation_*.json")):
        try:
            rows = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for row in rows:
            image_path = str(row.get("image_path", "")).strip()
            if image_path:
                metadata[image_path] = row
    return metadata


def read_image_bytes(image_path: Path) -> bytes:
    return image_path.read_bytes()


def build_score_prompt(style_texts: List[str]) -> str:
    style_block = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(style_texts))
    return (
        "You are my personal visual style reviewer. Evaluate whether this image matches my PhotoStyle library.\n\n"
        "Focus on:\n"
        "- subject clarity and visual hierarchy\n"
        "- composition and space\n"
        "- photographic realism rather than illustration\n"
        "- believable light, texture, atmosphere, and motion\n"
        "- whether the image feels emotionally aligned with the style library\n\n"
        "Reference style library:\n"
        f"{style_block}\n\n"
        "Return JSON only in this format:\n"
        '{"score": 0-100, "verdict": "fit|partial|miss", "reason": "one short sentence"}'
    )


def normalize_model_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return text


def score_image(client: genai.Client, image_path: Path, style_texts: List[str]) -> dict:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            build_score_prompt(style_texts),
            types.Part.from_bytes(data=read_image_bytes(image_path), mime_type=mime_type or "image/png"),
        ],
        config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),
    )

    text = normalize_model_json((response.text or "").strip())
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
        print(f"[{index}] {result['score']:>3} | {result['verdict']:<7} | {result['filename']} | {result['reason']}")


def parse_selection(raw: str, max_index: int) -> List[int]:
    if not raw.strip():
        return []

    normalized = raw.strip().lower()
    if normalized in {"none", "no", "n"}:
        return []
    if normalized == "both" and max_index >= 2:
        return [1, 2]
    if normalized == "all":
        return list(range(1, max_index + 1))

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


def prompt_feedback_bucket(label: str, max_index: int) -> List[int]:
    if max_index >= 2:
        print(f"Choose {label} images: 1, 2, both, none, all, or comma-separated indices.")
    else:
        print(f"Choose {label} images: 1 or none.")
    raw = input(f"{label.capitalize()} selection: ")
    return parse_selection(raw, max_index)


def validate_feedback_buckets(buckets: Dict[str, List[int]]) -> None:
    assigned_to: Dict[int, str] = {}
    for label, indices in buckets.items():
        for index in indices:
            if index in assigned_to:
                raise ValueError(f"Image {index} was assigned to both '{assigned_to[index]}' and '{label}'.")
            assigned_to[index] = label


def copy_liked_images(results: List[dict], liked_indices: List[int], destination: Path) -> List[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied_paths: List[Path] = []
    for index in liked_indices:
        source = Path(results[index - 1]["image_path"])
        candidate = destination / source.name
        stem = source.stem
        suffix = source.suffix
        counter = 1
        while candidate.exists():
            candidate = destination / f"{stem}_liked_{counter}{suffix}"
            counter += 1
        shutil.copy2(source, candidate)
        copied_paths.append(candidate)
    return copied_paths


def append_feedback_log(records: List[dict]) -> None:
    FEEDBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG_FILE.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_feedback_history() -> None:
    if not FEEDBACK_LOG_FILE.exists():
        print("\nNo feedback history yet.")
        return

    composition_stats = defaultdict(lambda: {"total": 0, "like": 0, "okay": 0, "dislike": 0})
    style_stats = defaultdict(lambda: {"total": 0, "like": 0, "okay": 0, "dislike": 0})
    combo_stats = defaultdict(lambda: {"total": 0, "like": 0, "okay": 0, "dislike": 0})

    with FEEDBACK_LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            feedback = row.get("feedback")
            if feedback not in FEEDBACK_LEVELS:
                continue
            composition = row.get("effective_composition") or row.get("requested_composition") or "unknown"
            style_mode = row.get("style_mode") or "unknown"
            combo = f"{style_mode} + {composition}"
            for bucket in (composition_stats[composition], style_stats[style_mode], combo_stats[combo]):
                bucket["total"] += 1
                bucket[feedback] += 1

    def top_entries(stats: Dict[str, dict]) -> List[tuple]:
        ranked = []
        for key, value in stats.items():
            like_rate = value["like"] / value["total"] if value["total"] else 0.0
            ranked.append((key, like_rate, value))
        ranked.sort(key=lambda item: (item[1], item[2]["total"]), reverse=True)
        return ranked[:3]

    print("\nFeedback insights from history:")
    print("Top compositions by like-rate:")
    for key, like_rate, value in top_entries(composition_stats):
        print(f"- {key}: like {value['like']}/{value['total']} ({like_rate:.0%}), okay {value['okay']}, dislike {value['dislike']}")

    print("Top style modes by like-rate:")
    for key, like_rate, value in top_entries(style_stats):
        print(f"- {key}: like {value['like']}/{value['total']} ({like_rate:.0%}), okay {value['okay']}, dislike {value['dislike']}")

    print("Top style+composition combinations by like-rate:")
    for key, like_rate, value in top_entries(combo_stats):
        print(f"- {key}: like {value['like']}/{value['total']} ({like_rate:.0%}), okay {value['okay']}, dislike {value['dislike']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated images against your distilled photo style.")
    parser.add_argument("--dir", default=str(IMAGE_DIR), help="Directory containing generated images to score.")
    parser.add_argument("--top-k-style", type=int, default=DEFAULT_TOP_K_STYLE, help="How many style descriptions from the dataset to use for scoring guidance.")
    parser.add_argument("--latest-batch", action="store_true", help="Score only the most recent generated batch from logs/ instead of the whole image directory.")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY. Please set it before running.")

    style_texts = load_style_outputs(DATASET_FILE)[: max(1, args.top_k_style)]
    source_log_path: Path | None = None
    if args.latest_batch:
        source_log_path = latest_generation_log()
        if source_log_path is None:
            raise FileNotFoundError("No generation logs found in logs/.")
        image_paths = load_images_from_generation_log(source_log_path)
        if not image_paths:
            raise FileNotFoundError(f"No existing images referenced by {source_log_path.resolve()}.")
        print(f"Scoring latest batch from: {source_log_path.resolve()}")
    else:
        image_paths = iter_images(Path(args.dir))

    metadata_by_image = load_generation_metadata()
    client = genai.Client(api_key=api_key)
    results = []
    for image_path in image_paths:
        try:
            result = score_image(client, image_path, style_texts)
            meta = metadata_by_image.get(result["image_path"], {})
            result.update(
                {
                    "subject": meta.get("subject", ""),
                    "style_mode": meta.get("style_mode", "unknown"),
                    "requested_composition": meta.get("requested_composition") or meta.get("composition", "unknown"),
                    "effective_composition": meta.get("effective_composition") or meta.get("composition", "unknown"),
                    "recommended_compositions": meta.get("recommended_compositions", []),
                    "recommended_style_modes": meta.get("recommended_style_modes", []),
                    "detected_topics": meta.get("detected_topics", []),
                }
            )
            results.append(result)
            print(f"{result['score']:>3} | {result['verdict']:<7} | {image_path.name} | {result['reason']}")
        except Exception as exc:
            print(f"[warn] Failed to score {image_path.name}: {exc}")

    results.sort(key=lambda item: item["score"], reverse=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOG_DIR / f"score_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved score report: {report_path.resolve()}")
    if not results:
        return

    print_ranked_results(results)
    print("\nGive each image one feedback bucket: like / okay / dislike.")
    like_indices = prompt_feedback_bucket("like", len(results))
    okay_indices = prompt_feedback_bucket("okay", len(results))
    dislike_indices = prompt_feedback_bucket("dislike", len(results))
    feedback_buckets = {"like": like_indices, "okay": okay_indices, "dislike": dislike_indices}
    validate_feedback_buckets(feedback_buckets)

    feedback_records = []
    for label, indices in feedback_buckets.items():
        for index in indices:
            result = results[index - 1]
            feedback_records.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "image_path": result["image_path"],
                    "filename": result["filename"],
                    "score": result["score"],
                    "verdict": result["verdict"],
                    "reason": result["reason"],
                    "feedback": label,
                    "subject": result.get("subject", ""),
                    "style_mode": result.get("style_mode", "unknown"),
                    "requested_composition": result.get("requested_composition", "unknown"),
                    "effective_composition": result.get("effective_composition", "unknown"),
                    "recommended_compositions": result.get("recommended_compositions", []),
                    "recommended_style_modes": result.get("recommended_style_modes", []),
                    "detected_topics": result.get("detected_topics", []),
                    "source_generation_log": str(source_log_path.resolve()) if source_log_path else "",
                    "score_report": str(report_path.resolve()),
                }
            )

    append_feedback_log(feedback_records)
    print(f"\nAppended feedback to: {FEEDBACK_LOG_FILE.resolve()}")

    if like_indices:
        copied_paths = copy_liked_images(results, like_indices, PHOTO_DIR)
        print("\nCopied liked images to my_photos/:")
        for path in copied_paths:
            print(f"- {path.resolve()}")
    else:
        print("\nNo liked images selected, so nothing was copied to my_photos/.")

    summarize_feedback_history()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] {exc}")
        sys.exit(1)
