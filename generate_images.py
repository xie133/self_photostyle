#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate images in your distilled personal visual style by reusing
descriptions from photo_style_dataset.jsonl as a style library.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "photo_style_dataset.jsonl"
OUTPUT_DIR = BASE_DIR / "generated_images"
LOG_DIR = BASE_DIR / "logs"
MODEL_NAME = "gemini-2.5-flash-image"
DEFAULT_STYLE_SAMPLES = 5
DEFAULT_DRAFT_STYLE_SAMPLES = 3
DEFAULT_SUBJECT = (
    "\u4e00\u4e2a\u96e8\u540e\u508d\u665a\u9a91\u8f66\u7a7f\u8fc7\u8def\u53e3\u7684\u5e74\u8f7b\u4eba\uff0c"
    "\u57ce\u5e02\u8857\u9053\u6f6e\u6e7f\u53d1\u4eae\uff0c\u884c\u4eba\u548c\u8f66\u8f86\u4ea4\u9519\uff0c"
    "\u50cf\u6444\u5f71\u5e08\u5728\u771f\u5b9e\u751f\u6d3b\u91cc\u77ac\u95f4\u6293\u62cd\u5230\u7684\u7247\u523b"
)

STYLE_PRESETS = {
    "balanced": (
        "- \u753b\u9762\u4fdd\u6301\u6742\u4e71\uff0c\u4f46\u9700\u8981\u6709\u6e05\u6670\u7684\u79e9\u5e8f\u611f\n"
        "- \u4e3b\u4f53\u5e94\u8be5\u5728\u590d\u6742\u73af\u5883\u4e2d\u88ab\u4e00\u773c\u8bc6\u522b\n"
        "- \u771f\u5b9e\u611f\u4f18\u5148\uff0c\u907f\u514d\u68a6\u5e7b\u5316\u6216\u8fc7\u5ea6\u620f\u5267\u5316\u5904\u7406"
    ),
    "messy_ordered_real": (
        "- \u753b\u9762\u8981\u66f4\u2018\u4e71\u4e2d\u6709\u5e8f\u2019\uff1a\u80cc\u666f\u53ef\u4ee5\u590d\u6742\uff0c"
        "\u4f46\u89c6\u7ebf\u5fc5\u987b\u88ab\u5f15\u5bfc\u5230\u4e3b\u4f53\n"
        "- \u4e3b\u4f53\u5fc5\u987b\u7a81\u51fa\uff0c\u53ef\u901a\u8fc7\u6784\u56fe\u3001\u660e\u6697\u3001"
        "\u989c\u8272\u5bf9\u6bd4\u6216\u8f7b\u5fae\u666f\u6df1\u5206\u79bb\u5b9e\u73b0\uff0c\u4f46\u4e0d\u8981\u50cf\u6446\u62cd\n"
        "- \u73b0\u5b9e\u611f\u8981\u66f4\u5f3a\uff1a\u6750\u8d28\u3001\u6c14\u5019\u3001\u5149\u7ebf\u3001"
        "\u900f\u89c6\u3001\u4eba\u7269\u59ff\u6001\u90fd\u8981\u50cf\u771f\u5b9e\u6444\u5f71\n"
        "- \u5141\u8bb8\u73af\u5883\u4e2d\u6709\u5e72\u6270\u5143\u7d20\uff0c\u4f46\u4e0d\u80fd\u62a2\u8d70"
        "\u4e3b\u4f53\u7684\u5b58\u5728\u611f\n"
        "- \u907f\u514d\u63d2\u753b\u611f\u3001CG \u611f\u3001\u7535\u5f71\u6d77\u62a5\u611f"
        "\u548c\u8fc7\u5ea6\u6f54\u51c0\u7684\u753b\u9762"
    ),
    "single_subject_poetic": (
        "- \u753b\u9762\u53ef\u4ee5\u66f4\u7b80\u6d01\uff0c\u56f4\u7ed5\u5355\u4e00\u4e3b\u4f53\u5c55\u5f00\uff0c\u4e0d\u9700\u8981\u8857\u5934\u590d\u6742\u611f\n"
        "- \u4e3b\u4f53\u5fc5\u987b\u660e\u786e\u3001\u96c6\u4e2d\uff0c\u62e5\u6709\u8f83\u5f3a\u7684\u5b58\u5728\u611f\u548c\u60c5\u7eea\u91cd\u5fc3\n"
        "- \u5149\u5f71\u3001\u989c\u8272\u548c\u7a7a\u95f4\u5e94\u8be5\u5e26\u6709\u8bd7\u610f\uff0c\u4f46\u4ecd\u4fdd\u7559\u771f\u5b9e\u6444\u5f71\u8d28\u611f\n"
        "- \u5141\u8bb8\u8f7b\u5fae\u68a6\u5e7b\u611f\uff0c\u4f46\u907f\u514d\u5ec9\u4ef7\u7279\u6548\u611f\u548c\u8fc7\u5ea6 AI \u63d2\u753b\u611f\n"
        "- \u66f4\u504f\u5411\u5b89\u9759\u3001\u8bd7\u6027\u3001\u51dd\u89c6\u611f\u5f3a\u7684\u5355\u5f20\u4f5c\u54c1"
    ),
    "experimental_floral_motion": (
        "- \u4e3b\u4f53\u53ef\u4ee5\u662f\u591a\u6735\u82b1\u4e0e\u82b1\u679d\uff0c\u91cd\u70b9\u662f\u6574\u4f53\u89c6\u89c9\u97f5\u5f8b\uff0c\u4e0d\u662f\u5355\u6735\u82b1\u7684\u5199\u771f\u7279\u5199\n"
        "- \u5f3a\u5316\u73af\u5f62\u65cb\u8f6c\u62d6\u5f71\u3001\u6da1\u6d41\u611f\u3001\u6c34\u7eb9\u611f\u548c\u5149\u5b66\u53cd\u5c04\uff0c\u8ba9\u753b\u9762\u50cf\u5b9e\u9a8c\u6444\u5f71\n"
        "- \u82b1\u6735\u4ecd\u7136\u8981\u80fd\u88ab\u8bc6\u522b\uff0c\u4f46\u8981\u88ab\u5377\u5165\u6d41\u52a8\u611f\u91cc\uff0c\u5f62\u6210\u5c42\u53e0\u4e0e\u8ff0\u60c5\n"
        "- \u8272\u5f69\u53ef\u4ee5\u66f4\u6d53\u90c1\uff0c\u4f46\u8d28\u611f\u4ecd\u7136\u8981\u50cf\u771f\u5b9e\u955c\u5934\u4e0b\u7684\u5f71\u50cf\uff0c\u4e0d\u8981\u63d2\u753b\u5316\n"
        "- \u5141\u8bb8\u8ff7\u79bb\u611f\u3001\u68a6\u5e7b\u611f\u548c\u5149\u5f71\u62bd\u8c61\uff0c\u4f46\u8981\u50cf\u9ad8\u7ea7\u5b9e\u9a8c\u6444\u5f71\u4f5c\u54c1"
    ),
    "graphic_documentary_silhouette": (
        "- \u753b\u9762\u5f3a\u8c03\u56fe\u5f62\u611f\u3001\u7ed3\u6784\u611f\u548c\u4eba\u7269\u7ad9\u4f4d\u8282\u594f\uff0c\u800c\u4e0d\u662f\u8868\u60c5\u7ec6\u8282\n"
        "- \u4eba\u7269\u5e94\u8be5\u5728\u5f3a\u9006\u5149\u4e0b\u5f62\u6210\u7eaf\u9ed1\u526a\u5f71\uff0c\u8f6e\u5ed3\u5e72\u51c0\uff0c\u52a8\u4f5c\u5404\u4e0d\u76f8\u540c\n"
        "- \u53ef\u4ee5\u4f7f\u7528\u9ed1\u767d\u6216\u8fd1\u4e4e\u9ed1\u767d\u7684\u9ad8\u53cd\u5dee\u8868\u8fbe\uff0c\u5149\u7ebf\u786c\u6717\uff0c\u6709\u7eaa\u5b9e\u6444\u5f71\u611f\n"
        "- \u80cc\u666f\u5c3d\u91cf\u7b80\u5316\uff0c\u7ed9\u51e0\u4f55\u7ed3\u6784\u3001\u6a2a\u5411\u7ebf\u6761\u548c\u7559\u767d\u5929\u7a7a\u8ba9\u4f4d\n"
        "- \u6574\u4f53\u6c14\u8d28\u8981\u514b\u5236\u3001\u7d27\u5f20\u3001\u50cf\u7eaa\u5b9e\u6444\u5f71\u91cc\u88ab\u6293\u5230\u7684\u4e00\u4e2a\u56fe\u5f62\u77ac\u95f4"
    ),
}

BUDGET_MODES = {
    "draft": {
        "default_count": 2,
        "default_samples": DEFAULT_DRAFT_STYLE_SAMPLES,
    },
    "final": {
        "default_count": 4,
        "default_samples": DEFAULT_STYLE_SAMPLES,
    },
}

ASPECT_RATIOS = {
    "1:1": "1:1",
    "3:4": "3:4",
    "4:3": "4:3",
    "9:16": "9:16",
    "16:9": "16:9",
}

COMPOSITION_PRESETS = {
    "auto": "- 构图可以根据主题自由选择，但必须让主体、空间和节奏关系清晰成立",
    "central": "- 优先使用中央构图，让主体明确稳定地落在画面中心，整体重心清楚",
    "rule_of_thirds": "- 优先参考三分法构图，让主体或视觉重心落在三分线或三分交点附近",
    "symmetry": "- 优先使用对称构图，强调中轴、镜像关系和几何秩序感",
    "leading_lines": "- 优先使用引导线构图，让道路、边缘、建筑线条或光影把视线带向主体",
    "negative_space": "- 优先使用留白构图，让主体周围保留大面积呼吸空间，强化情绪与孤独感",
    "layered_depth": "- 优先构建前景、中景、背景的层次，让画面更有空间纵深和视觉递进",
    "repetition": "- 优先使用重复与节奏构图，强化图形秩序、重复元素和视觉韵律",
    "frame_within_frame": "- 优先使用框景构图，让门、窗、结构边界或环境元素包裹主体",
}


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[setup] Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pip_name])


def ensure_dependencies() -> None:
    ensure_package("google.genai", "google-genai")
    ensure_package("PIL", "Pillow")


ensure_dependencies()

from google import genai
from google.genai import types


def load_style_outputs(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    outputs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] Skipping invalid JSONL line {line_number}: {exc}")
                continue

            output = str(row.get("output", "")).strip()
            if output:
                outputs.append(output)

    if not outputs:
        raise ValueError("No valid style descriptions found in dataset.")

    return outputs


def select_style_texts(style_texts: List[str], sample_count: int) -> List[str]:
    chosen_count = min(sample_count, len(style_texts))
    return random.sample(style_texts, chosen_count)


def build_prompt(subject: str, sampled_texts: List[str], style_mode: str, composition: str) -> str:
    style_block = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(sampled_texts))
    preset = STYLE_PRESETS[style_mode]
    composition_rule = COMPOSITION_PRESETS[composition]

    return (
        "\u4f60\u73b0\u5728\u8981\u4e3a\u6211\u751f\u6210\u4e00\u5f20\u7b26\u5408\u4e2a\u4eba\u89c6\u89c9\u5ba1\u7f8e\u7684\u7167\u7247\u3002\n\n"
        "\u4e3b\u4f53\u9700\u6c42\uff1a\n"
        f"{subject}\n\n"
        "\u98ce\u683c\u786c\u7ea6\u675f\uff1a\n"
        "- \u5149\u5f71\u5fc5\u987b\u81ea\u7136\uff0c\u6709\u5c42\u6b21\uff0c\u4e0d\u8981\u68da\u62cd\u611f\n"
        "- \u4f18\u5148\u4fdd\u7559\u771f\u5b9e\u6444\u5f71\u8d28\u611f\uff0c\u907f\u514d\u660e\u663e AI \u63d2\u753b\u611f\n"
        "- \u4e0d\u8981\u8fc7\u5ea6\u7cbe\u81f4\uff0c\u4e0d\u8981\u7f51\u7ea2\u6ee4\u955c\u611f\uff0c\u4e0d\u8981\u5546\u4e1a\u6d77\u62a5\u611f\n"
        "- \u6784\u56fe\u548c\u7a7a\u95f4\u8981\u6709\u5ba1\u7f8e\u5224\u65ad\uff0c\u65e0\u8bba\u662f\u590d\u6742\u573a\u666f\u8fd8\u662f\u5355\u4e00\u4e3b\u4f53\uff0c\u90fd\u8981\u6709\u751f\u547d\u529b\u548c\u60c5\u7eea\n"
        f"{composition_rule}\n"
        f"{preset}\n\n"
        "\u4ee5\u4e0b\u662f\u6211\u4e2a\u4eba\u5ba1\u7f8e\u84b8\u998f\u51fa\u6765\u7684\u53c2\u8003\u98ce\u683c\u63cf\u8ff0\uff0c"
        "\u8bf7\u5438\u6536\u8fd9\u4e9b\u7279\u5f81\u540e\u518d\u51fa\u56fe\uff1a\n"
        f"{style_block}\n\n"
        "\u8f93\u51fa\u8981\u6c42\uff1a\n"
        "- \u76f4\u63a5\u751f\u6210\u56fe\u7247\n"
        "- \u6574\u4f53\u6c14\u8d28\u4f18\u5148\u670d\u4ece\u6211\u7684\u98ce\u683c\u5e93\uff0c\u4f46\u4e0d\u5fc5\u5f3a\u884c\u505a\u6210\u8857\u62cd\n"
        "- \u753b\u9762\u771f\u5b9e\u3001\u677e\u5f1b\u3001\u590d\u6742\u3001\u6709\u547c\u5438\u611f"
    )


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY. Please set it before running.")
    return api_key


def save_image_response(response, output_path: Path) -> bool:
    candidates = []

    if getattr(response, "candidates", None):
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            candidates.extend(parts)
    elif getattr(response, "parts", None):
        candidates.extend(response.parts)

    for part in candidates:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None:
            image = part.as_image()
            image.save(output_path)
            return True

    return False


def write_generation_log(log_path: Path, rows: List[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def generate_one_image(
    client: genai.Client,
    subject: str,
    style_texts: List[str],
    samples: int,
    style_mode: str,
    composition: str,
    aspect_ratio: str,
    output_path: Path,
) -> Tuple[Path, List[str]]:
    sampled_texts = select_style_texts(style_texts, samples)
    prompt = build_prompt(subject, sampled_texts, style_mode, composition)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(aspect_ratio=ASPECT_RATIOS[aspect_ratio]),
        ),
    )

    if not save_image_response(response, output_path):
        text_fallback = getattr(response, "text", "") or "No image returned by the model."
        raise RuntimeError(text_fallback)

    return output_path, sampled_texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images using your distilled photo style library."
    )
    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help="The scene or subject you want the model to generate.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_STYLE_SAMPLES,
        help="How many style samples to draw from the JSONL dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible style-sample selection.",
    )
    parser.add_argument(
        "--style-mode",
        choices=sorted(STYLE_PRESETS.keys()),
        default="balanced",
        help="Apply an extra style steering preset on top of the style library.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="How many image variants to generate in one run.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=sorted(BUDGET_MODES.keys()),
        default="draft",
        help="draft uses fewer style samples and fewer outputs by default; final is for fuller exploration.",
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=sorted(ASPECT_RATIOS.keys()),
        default="1:1",
        help="Aspect ratio for image generation.",
    )
    parser.add_argument(
        "--composition",
        choices=sorted(COMPOSITION_PRESETS.keys()),
        default="auto",
        help="Apply a composition rule such as central framing, symmetry, leading lines, or negative space.",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0.")

    if args.seed is not None:
        random.seed(args.seed)

    api_key = get_api_key()
    style_texts = load_style_outputs(DATASET_FILE)
    if args.samples == DEFAULT_STYLE_SAMPLES and args.budget_mode == "draft":
        args.samples = BUDGET_MODES["draft"]["default_samples"]
    if args.count is None:
        args.count = BUDGET_MODES[args.budget_mode]["default_count"]
    if args.count <= 0:
        raise ValueError("--count must be greater than 0.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    client = genai.Client(api_key=api_key)
    results = []
    for index in range(1, args.count + 1):
        output_path = OUTPUT_DIR / f"photostyle_{batch_id}_{index:02d}.png"
        saved_path, sampled_texts = generate_one_image(
            client=client,
            subject=args.subject,
            style_texts=style_texts,
            samples=args.samples,
            style_mode=args.style_mode,
            composition=args.composition,
            aspect_ratio=args.aspect_ratio,
            output_path=output_path,
        )
        results.append(
            {
                "image_path": str(saved_path.resolve()),
                "subject": args.subject,
                "style_mode": args.style_mode,
                "composition": args.composition,
                "style_samples": sampled_texts,
            }
        )
        print(f"Saved image {index}/{args.count}: {saved_path.resolve()}")

    log_path = LOG_DIR / f"generation_{batch_id}.json"
    write_generation_log(log_path, results)

    print(f"Model: {MODEL_NAME}")
    print(f"Subject: {args.subject}")
    print(f"Style samples used: {min(args.samples, len(style_texts))}")
    print(f"Style mode: {args.style_mode}")
    print(f"Composition: {args.composition}")
    print(f"Aspect ratio: {args.aspect_ratio}")
    print(f"Variants: {args.count}")
    print(f"Budget mode: {args.budget_mode}")
    print(f"Log: {log_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] {exc}")
        sys.exit(1)
