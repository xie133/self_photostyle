# Self PhotoStyle

[English](./README.md) | [简体中文](./README.zh-CN.md)

Distill your personal visual taste from reference photos and use it to guide AI image generation.

`self_photostyle` is a small Python workflow built around Gemini. It helps you turn a folder of reference images into a reusable style library, then use that style language to generate and score new images.

## What It Does

1. Analyze your local reference photos with Gemini 2.5 Flash.
2. Export a reusable JSONL style dataset.
3. Generate new images with Gemini image models using your distilled style language.
4. Score generated images against the same style library.

## Repository Layout

```text
self_photostyle/
|- configs/
|  |- generation_presets.json
|- generated_images/
|  |- .gitkeep
|- logs/
|  |- .gitkeep
|- my_photos/
|  |- .gitkeep
|- prompts/
|  |- aesthetic_analysis_prompt.txt
|- .env.example
|- .gitignore
|- LICENSE
|- README.md
|- README.zh-CN.md
|- requirements.txt
|- build_style_dataset.py
|- generate_images.py
`- score_images.py
```

## Setup

Create and activate a virtual environment if you want an isolated setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set your Gemini API key for the current terminal session:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

## 1. Build Your Style Dataset

Put your own reference photos into `my_photos/`, then run:

```powershell
python .\build_style_dataset.py
```

This creates:

- `photo_style_dataset.jsonl`

Each line looks like:

```json
{"instruction":"Generate an image in my visual style","input":"","output":"...style description..."}
```

## 2. Generate Images

Write your own prompt inside `--subject`:

```powershell
python .\generate_images.py --subject "Describe the image you want to create here" --style-mode balanced --composition auto --aspect-ratio 3:4 --budget-mode draft
```

Replace the text inside `--subject` with your own image idea.

Useful options:

- `--subject`: your image prompt
- `--style-mode`: choose a built-in visual direction
- `--composition`: choose a composition rule such as `central`, `symmetry`, `rule_of_thirds`, or `negative_space`
- `--aspect-ratio`: choose image framing such as `3:4`, `9:16`, or `16:9`
- `--budget-mode`: use `draft` for cheaper exploration or `final` for fuller batches
- `--count`: number of variants to generate

## 3. Score Generated Images

Score all generated images:

```powershell
python .\score_images.py
```

Score only the latest generated batch:

```powershell
python .\score_images.py --latest-batch
```

After scoring, the script will ask which images you want to copy back into `my_photos/`.

## Style Modes

Current built-in modes include:

- `balanced`
- `messy_ordered_real`
- `single_subject_poetic`
- `experimental_floral_motion`
- `graphic_documentary_silhouette`

## Notes For Open Source Users

- This repository does not include private source photos, generated outputs, logs, or a real dataset file.
- You need to create your own `photo_style_dataset.jsonl` by running `build_style_dataset.py` on your own references.
- Gemini API tuning is not required here; the project uses retrieval-style prompt steering instead.

## Suggested Workflow

1. Curate a small folder of reference images you genuinely love.
2. Build the dataset with `build_style_dataset.py`.
3. Start with one custom prompt in `draft` mode.
4. Score only the latest batch.
5. Select the images you want to feed back into `my_photos/`.
6. Rebuild the dataset and refine prompts or style modes.

## License

MIT
