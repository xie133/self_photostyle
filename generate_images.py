#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate images in your distilled personal visual style by reusing
entries from photo_style_dataset.jsonl as a style library.
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
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "photo_style_dataset.jsonl"
OUTPUT_DIR = BASE_DIR / "generated_images"
LOG_DIR = BASE_DIR / "logs"
MODEL_NAME = "gemini-2.5-flash-image"
DEFAULT_STYLE_SAMPLES = 5
DEFAULT_DRAFT_STYLE_SAMPLES = 3
DEFAULT_SUBJECT = "A rainy Tokyo street at dusk, photographed with a candid cinematic feeling."

STYLE_PRESETS = {
    "balanced": (
        "- Keep the image readable and emotionally grounded.\n"
        "- Let the subject stand out without making the frame feel artificial.\n"
        "- Preserve realistic light, texture, and spatial believability."
    ),
    "messy_ordered_real": (
        "- Let the frame feel messy but intentionally organized.\n"
        "- Keep the subject clear inside a complex environment.\n"
        "- Favor real materials, lived-in surfaces, and believable documentary energy."
    ),
    "single_subject_poetic": (
        "- Favor a cleaner frame built around one clear subject.\n"
        "- Let the image feel calm, poetic, and emotionally focused.\n"
        "- Allow softness and atmosphere, but keep it photographic rather than illustrative."
    ),
    "experimental_floral_motion": (
        "- Favor layered petals, rotational blur, vortex-like motion, and optical abstraction.\n"
        "- Keep flowers recognizable while letting them dissolve into motion and rhythm.\n"
        "- The result should feel like experimental photography, not digital illustration."
    ),
    "graphic_documentary_silhouette": (
        "- Favor strong geometry, clear silhouettes, and disciplined visual rhythm.\n"
        "- Use hard backlight and bold contrast when appropriate.\n"
        "- Let the image feel like a documentary moment with graphic force."
    ),
}

BUDGET_MODES = {
    "draft": {"default_count": 2, "default_samples": DEFAULT_DRAFT_STYLE_SAMPLES},
    "final": {"default_count": 4, "default_samples": DEFAULT_STYLE_SAMPLES},
}

ASPECT_RATIOS = {
    "1:1": "1:1",
    "3:4": "3:4",
    "4:3": "4:3",
    "9:16": "9:16",
    "16:9": "16:9",
}

COMPOSITION_PRESETS = {
    "auto": "Choose the composition based on the subject, but keep subject placement, space, and rhythm clearly intentional.",
    "central": "Prefer central composition so the subject sits clearly and steadily in the middle of the frame.",
    "rule_of_thirds": "Prefer rule-of-thirds composition so the subject or visual weight sits near third lines or intersections.",
    "symmetry": "Prefer symmetry, emphasizing axis, mirroring, and geometric order.",
    "leading_lines": "Prefer leading lines so roads, edges, architecture, or light guide the eye toward the subject.",
    "negative_space": "Prefer negative space so the subject has breathing room and the mood feels spacious and intentional.",
    "layered_depth": "Prefer foreground, midground, and background layering for stronger depth and visual progression.",
    "repetition": "Prefer repetition and rhythm, emphasizing repeated forms and graphic structure.",
    "frame_within_frame": "Prefer frame-within-frame composition so doors, windows, structures, or environmental elements contain the subject.",
}

TOPIC_HINTS = {
    "architecture": {
        "keywords": ["building", "architecture", "apartment", "church", "house", "tower", "atrium", "hotel", "balcony", "建筑", "教堂", "公寓", "中庭", "房子"],
        "recommended_compositions": ["central", "symmetry", "repetition", "leading_lines"],
        "recommended_style_modes": ["balanced", "single_subject_poetic"],
    },
    "street": {
        "keywords": ["street", "city", "pedestrian", "traffic", "crossing", "alley", "东京", "街拍", "街头", "路口", "车流", "城市"],
        "recommended_compositions": ["leading_lines", "layered_depth", "rule_of_thirds"],
        "recommended_style_modes": ["messy_ordered_real", "balanced"],
    },
    "portrait_or_single_subject": {
        "keywords": ["flower", "portrait", "person", "girl", "boy", "single", "恋人", "女生", "男生", "人物", "花", "单一主体"],
        "recommended_compositions": ["central", "rule_of_thirds", "negative_space"],
        "recommended_style_modes": ["single_subject_poetic", "balanced"],
    },
    "landscape_or_minimal": {
        "keywords": ["mountain", "snow", "sunset", "lake", "field", "sky", "山", "雪", "落日", "天空", "旷野", "孤立"],
        "recommended_compositions": ["negative_space", "central", "layered_depth"],
        "recommended_style_modes": ["balanced", "single_subject_poetic"],
    },
    "silhouette_or_graphic": {
        "keywords": ["silhouette", "black and white", "逆光", "剪影", "黑白", "图形", "几何"],
        "recommended_compositions": ["symmetry", "repetition", "central"],
        "recommended_style_modes": ["graphic_documentary_silhouette", "balanced"],
    },
    "experimental_motion": {
        "keywords": ["vortex", "rotation", "swirl", "blur", "motion", "旋焦", "旋涡", "拖影", "波纹", "实验摄影"],
        "recommended_compositions": ["central", "repetition", "frame_within_frame"],
        "recommended_style_modes": ["experimental_floral_motion", "single_subject_poetic"],
    },
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
    return random.sample(style_texts, min(sample_count, len(style_texts)))


def detect_topics(subject: str) -> List[str]:
    lowered = subject.lower()
    matched: List[str] = []
    for topic_name, info in TOPIC_HINTS.items():
        if any(keyword.lower() in lowered for keyword in info["keywords"]):
            matched.append(topic_name)
    return matched


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recommend_settings(subject: str) -> Dict[str, List[str]]:
    matched_topics = detect_topics(subject)
    recommended_compositions: List[str] = []
    recommended_style_modes: List[str] = []

    for topic_name in matched_topics:
        info = TOPIC_HINTS[topic_name]
        recommended_compositions.extend(info["recommended_compositions"])
        recommended_style_modes.extend(info["recommended_style_modes"])

    return {
        "topics": matched_topics,
        "recommended_compositions": unique_preserve_order(recommended_compositions),
        "recommended_style_modes": unique_preserve_order(recommended_style_modes),
    }


def effective_composition(requested_composition: str, recommendations: Dict[str, List[str]]) -> str:
    if requested_composition != "auto":
        return requested_composition
    recs = recommendations["recommended_compositions"]
    return recs[0] if recs else "auto"


def build_prompt(
    subject: str,
    sampled_texts: List[str],
    style_mode: str,
    requested_composition: str,
    recommendations: Dict[str, List[str]],
) -> Tuple[str, str]:
    style_block = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(sampled_texts))
    preset = STYLE_PRESETS[style_mode]
    chosen_composition = effective_composition(requested_composition, recommendations)
    composition_rule = COMPOSITION_PRESETS[chosen_composition]
    topic_line = ", ".join(recommendations["topics"]) if recommendations["topics"] else "no strong topic match"
    rec_comp_line = ", ".join(recommendations["recommended_compositions"]) or "none"
    rec_style_line = ", ".join(recommendations["recommended_style_modes"]) or "none"

    prompt = (
        "You are generating a photograph that should align with my personal visual taste.\n\n"
        f"Subject request:\n{subject}\n\n"
        "Core constraints:\n"
        "- Keep the image photographic, not obviously illustrative or AI-glossy.\n"
        "- Preserve believable light, materials, and spatial relationships.\n"
        "- Avoid generic influencer-filter aesthetics and overly commercial polish.\n"
        "- Composition must feel intentional, emotionally readable, and visually disciplined.\n"
        f"- Composition guidance: {composition_rule}\n"
        f"- Style mode guidance:\n{preset}\n\n"
        "Automatic recommendation context:\n"
        f"- Detected topic hints: {topic_line}\n"
        f"- Recommended compositions: {rec_comp_line}\n"
        f"- Recommended style modes: {rec_style_line}\n\n"
        "Reference style descriptions from my distilled library:\n"
        f"{style_block}\n\n"
        "Output requirement:\n"
        "Generate the image directly and prioritize the emotional, compositional, and photographic qualities above."
    )
    return prompt, chosen_composition


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
    requested_composition: str,
    recommendations: Dict[str, List[str]],
    aspect_ratio: str,
    output_path: Path,
) -> Tuple[Path, List[str], str]:
    sampled_texts = select_style_texts(style_texts, samples)
    prompt, chosen_composition = build_prompt(
        subject=subject,
        sampled_texts=sampled_texts,
        style_mode=style_mode,
        requested_composition=requested_composition,
        recommendations=recommendations,
    )
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

    return output_path, sampled_texts, chosen_composition


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images using your distilled photo style library.")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT, help="The scene or subject you want the model to generate.")
    parser.add_argument("--samples", type=int, default=DEFAULT_STYLE_SAMPLES, help="How many style samples to draw from the JSONL dataset.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible style-sample selection.")
    parser.add_argument("--style-mode", choices=sorted(STYLE_PRESETS.keys()), default="balanced", help="Apply an extra style steering preset on top of the style library.")
    parser.add_argument("--count", type=int, default=None, help="How many image variants to generate in one run.")
    parser.add_argument("--budget-mode", choices=sorted(BUDGET_MODES.keys()), default="draft", help="draft uses fewer style samples and fewer outputs by default; final is for fuller exploration.")
    parser.add_argument("--aspect-ratio", choices=sorted(ASPECT_RATIOS.keys()), default="1:1", help="Aspect ratio for image generation.")
    parser.add_argument("--composition", choices=sorted(COMPOSITION_PRESETS.keys()), default="auto", help="Apply a composition rule such as central framing, symmetry, leading lines, or negative space.")
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

    recommendations = recommend_settings(args.subject)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    client = genai.Client(api_key=api_key)
    results = []
    for index in range(1, args.count + 1):
        output_path = OUTPUT_DIR / f"photostyle_{batch_id}_{index:02d}.png"
        saved_path, sampled_texts, chosen_composition = generate_one_image(
            client=client,
            subject=args.subject,
            style_texts=style_texts,
            samples=args.samples,
            style_mode=args.style_mode,
            requested_composition=args.composition,
            recommendations=recommendations,
            aspect_ratio=args.aspect_ratio,
            output_path=output_path,
        )
        results.append(
            {
                "image_path": str(saved_path.resolve()),
                "subject": args.subject,
                "style_mode": args.style_mode,
                "requested_composition": args.composition,
                "effective_composition": chosen_composition,
                "recommended_compositions": recommendations["recommended_compositions"],
                "recommended_style_modes": recommendations["recommended_style_modes"],
                "detected_topics": recommendations["topics"],
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
    print(f"Requested composition: {args.composition}")
    print(f"Recommended compositions: {', '.join(recommendations['recommended_compositions']) or 'none'}")
    print(f"Recommended style modes: {', '.join(recommendations['recommended_style_modes']) or 'none'}")
    print(f"Detected topics: {', '.join(recommendations['topics']) or 'none'}")
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
