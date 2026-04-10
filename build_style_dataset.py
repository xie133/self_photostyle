#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch analyze local street photography images with Gemini 2.5 Flash
and export the distilled aesthetic descriptions to JSONL for fine-tuning.

Official SDK note:
Google currently recommends the `google-genai` package for Gemini API access.
This script will auto-install required dependencies if they are missing.
"""

from __future__ import annotations

import importlib
import json
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List


INSTRUCTION_TEXT = "\u751f\u6210\u4e00\u5f20\u6709\u751f\u547d\u529b\u7684\u8857\u62cd"
BASE_DIR = Path(__file__).resolve().parent
PHOTO_DIR = BASE_DIR / "my_photos"
OUTPUT_FILE = BASE_DIR / "photo_style_dataset.jsonl"
PROMPT_FILE = BASE_DIR / "prompts" / "aesthetic_analysis_prompt.txt"
MODEL_NAME = "gemini-2.5-flash"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2
REQUEST_INTERVAL_SECONDS = 0.5


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[setup] Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pip_name])


def ensure_dependencies() -> None:
    ensure_package("google.genai", "google-genai")
    ensure_package("tqdm", "tqdm")


ensure_dependencies()

from google import genai
from google.genai import types
from tqdm import tqdm


def load_prompt() -> str:
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8").strip()

    return (
        "\u4f60\u73b0\u5728\u662f\u4e00\u540d\u9876\u7ea7\u8857\u5934\u6444\u5f71\u5e08\u3002\u8bf7\u5206\u6790\u8fd9\u5f20\u7167\u7247\u4e2d\u201c\u4e71\u7cdf\u7cdf\u4f46\u6709\u751f\u547d\u529b\u201d\u7684\u7279\u5f81\u3002"
        "\u91cd\u70b9\u63d0\u53d6\uff1a\u6784\u56fe\u4e2d\u7684\u89c6\u89c9\u5e72\u6270\u9879\u3001\u771f\u5b9e\u8857\u9053\u7684\u7eb9\u7406\u611f\u3001\u52a8\u6001\u6a21\u7cca\u7684\u8fd0\u7528\u3001\u4ee5\u53ca\u4e0d\u523b\u610f\u7684\u5149\u5f71\u3002"
        "\u6700\u540e\uff0c\u8bf7\u5c06\u8fd9\u4e9b\u7279\u5f81\u8f6c\u5316\u6210\u4e00\u6bb5\u9002\u5408 Gemini 3 Flash Image (banananano) \u751f\u6210\u56fe\u7247\u7684\u81ea\u7136\u8bed\u8a00 Prompt\u3002"
        "\n\n"
        "\u8f93\u51fa\u8981\u6c42\uff1a\u53ea\u8fd4\u56de\u6700\u7ec8\u751f\u6210\u7528\u7684\u81ea\u7136\u8bed\u8a00 Prompt\uff0c\u4e0d\u8981\u89e3\u91ca\uff0c\u4e0d\u8981\u5206\u70b9\uff0c\u4e0d\u8981\u52a0\u6807\u9898\u3002"
    )


def iter_image_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder.resolve()}")

    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def read_image_bytes(image_path: Path) -> bytes:
    with image_path.open("rb") as f:
        return f.read()


def get_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/jpeg"


def build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing GEMINI_API_KEY. Please set your Gemini API key as an environment variable."
        )
    return genai.Client(api_key=api_key)


def analyze_image(client: genai.Client, image_path: Path) -> str:
    image_bytes = read_image_bytes(image_path)
    mime_type = get_mime_type(image_path)
    prompt = load_prompt()
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )

            text = (response.text or "").strip()
            if not text:
                raise ValueError("Empty response text")
            return text
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                sleep_seconds = RETRY_BASE_DELAY ** (attempt - 1)
                print(
                    f"[warn] API failed for {image_path.name} (attempt {attempt}/{MAX_RETRIES}): {exc}"
                )
                time.sleep(sleep_seconds)
            else:
                break

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {image_path.name}") from last_error


def append_jsonl(output_path: Path, rows: List[dict]) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    client = build_client()
    image_files = list(iter_image_files(PHOTO_DIR))

    if not image_files:
        print(f"[info] No images found in {PHOTO_DIR.resolve()}")
        return

    output_path = OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    success_count = 0
    failed_files: List[str] = []

    progress = tqdm(image_files, desc="Analyzing photos", unit="image")
    for image_path in progress:
        progress.set_postfix(file=image_path.name)
        try:
            aesthetic_prompt = analyze_image(client, image_path)
            row = {
                "instruction": INSTRUCTION_TEXT,
                "input": "",
                "output": aesthetic_prompt,
            }
            append_jsonl(output_path, [row])
            success_count += 1
        except OSError as exc:
            failed_files.append(f"{image_path.name} | file error: {exc}")
            print(f"[warn] Skipping unreadable file: {image_path} | {exc}")
        except Exception as exc:
            failed_files.append(f"{image_path.name} | api error: {exc}")
            print(f"[warn] Skipping failed API result: {image_path} | {exc}")
        finally:
            time.sleep(REQUEST_INTERVAL_SECONDS)

    print("\nDone.")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_files)}")
    print(f"Output: {output_path.resolve()}")

    if failed_files:
        print("\nFailed items:")
        for item in failed_files:
            print(f"- {item}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] {exc}")
        sys.exit(1)
