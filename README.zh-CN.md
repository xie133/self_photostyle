# Self PhotoStyle

[English](./README.md) | [简体中文](./README.zh-CN.md)

从参考照片中蒸馏你的个人视觉审美，并用它来引导 AI 图片生成。

`self_photostyle` 是一个基于 Gemini 的小型 Python 工作流。它可以帮你把一组参考图片整理成可复用的风格库，再用这套风格语言去生成和筛选新的图片。

## 项目能做什么

1. 用 Gemini 2.5 Flash 分析你的本地参考照片。
2. 导出可复用的 JSONL 风格数据集。
3. 用蒸馏出的风格语言引导 Gemini 图片生成模型出图。
4. 用同一套风格库给生成结果打分。

## 目录结构

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

## 环境准备

如果你想隔离环境，可以先创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt
```

在当前终端会话中设置 Gemini API Key：

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

## 1. 构建风格数据集

把你自己的参考照片放进 `my_photos/`，然后运行：

```powershell
python .\build_style_dataset.py
```

这会生成：

- `photo_style_dataset.jsonl`

每一行结构大致如下：

```json
{"instruction":"Generate an image in my visual style","input":"","output":"...style description..."}
```

## 2. 生成图片

把你自己的图片需求写进 `--subject`：

```powershell
python .\generate_images.py --subject "在这里描述你想生成的画面" --style-mode balanced --composition auto --aspect-ratio 3:4 --budget-mode draft
```

把 `--subject` 里的内容替换成你自己的 prompt 即可。

常用参数：

- `--subject`：你的图片描述
- `--style-mode`：选择内置风格方向
- `--composition`：选择构图法则，比如 `central`、`symmetry`、`rule_of_thirds`、`negative_space`
- `--aspect-ratio`：选择画幅，比如 `3:4`、`9:16`、`16:9`
- `--budget-mode`：`draft` 更省成本，`final` 更适合认真挑图
- `--count`：一次生成多少张变体

如果使用 `--composition auto`，脚本会根据题材自动推荐构图方式和风格模式，并把这些推荐记录进生成日志。

## 3. 给生成结果打分

给全部生成结果打分：

```powershell
python .\score_images.py
```

只给最近一批生成结果打分：

```powershell
python .\score_images.py --latest-batch
```

打分完成后，脚本会要求你把图片分别归入 `like`、`okay`、`dislike` 三个反馈桶。

- `like` 会被复制回 `my_photos/`
- 所有反馈都会追加到 `logs/feedback_history.jsonl`
- 脚本会输出“哪种构图”和“哪种风格+构图组合”更容易得到喜欢结果的统计摘要

## 风格模式

当前内置的风格模式包括：

- `balanced`
- `messy_ordered_real`
- `single_subject_poetic`
- `experimental_floral_motion`
- `graphic_documentary_silhouette`

## 给开源用户的说明

- 本仓库不包含私有参考照片、生成结果、日志或真实数据集文件。
- 你需要先用自己的参考图片运行 `build_style_dataset.py`，生成自己的 `photo_style_dataset.jsonl`。
- 这个项目不依赖 Gemini 原生微调，而是采用检索式的 prompt 风格引导流程。

## 建议工作流

1. 先整理一小批你真正喜欢的参考图。
2. 用 `build_style_dataset.py` 构建风格库。
3. 先用 `draft` 模式跑一条自定义 prompt。
4. 只给最近一批结果打分。
5. 选出想回流的图片复制进 `my_photos/`。
6. 重新构建数据集，并继续优化 prompt 或风格模式。

## License

MIT
