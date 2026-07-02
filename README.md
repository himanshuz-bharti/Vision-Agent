# Local Desktop Agent MVP

This is a first-pass local desktop-control agent:

```text
user task
-> screenshot
-> OCR + optional local VLM summary
-> local Qwen planner chooses one JSON action
-> safety gate approves or blocks
-> PyAutoGUI executes
-> verifier checks visible result
-> repeat up to 10 steps
```

The default mode is dry-run. It captures the screen, asks the local planner what it would do, logs the result, and stops without moving the mouse or typing.

## What You Have

From the checks I could run in this workspace:

```text
GPU: NVIDIA GeForce GTX 1650 Ti, 4096 MiB VRAM
CPU threads visible to Python: 8
Python: 3.13.12
Ollama client: installed, but the Ollama server was not running
Tesseract OCR: not installed or not on PATH
```

That GPU is usable for small/quantized local models. For this project, lean on OCR and small VLMs first.

## Setup

```powershell
cd C:\Vision_Chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Ollama in another terminal:

```powershell
ollama serve
```

Pull the low-resource default models:

```powershell
ollama pull qwen2.5-coder:3b
ollama pull moondream
```

Optional OCR install on Windows:

```powershell
winget install --id UB-Mannheim.TesseractOCR
```

Restart your terminal after installing Tesseract so it is on `PATH`.

## Run

Dry-run first:

```powershell
python -m desktop_agent "Open Notepad and type hello from my local desktop agent"
```

Actually execute low-risk actions:

```powershell
python -m desktop_agent --execute "Open Notepad and type hello from my local desktop agent"
```

Safer execution with manual approval before every action:

```powershell
python -m desktop_agent --execute --confirm-each-action "Open Notepad and type hello"
```

Use a stronger planner if your machine can tolerate it:

```powershell
ollama pull qwen2.5-coder:7b
python -m desktop_agent --planner-model qwen2.5-coder:7b --execute "Open Notepad and type hello"
```

Force VLM use:

```powershell
python -m desktop_agent --use-vlm always --vlm-model moondream "Describe what app is open"
```

Each run writes screenshots and JSON logs under `runs\YYYYMMDD_HHMMSS\`.

## Local VLM Recommendations

For your detected GTX 1650 Ti 4 GB VRAM:

| Model | Command | Fit | Notes |
| --- | --- | --- | --- |
| Moondream 2 | `ollama pull moondream` | Best first choice | Small 1.8B-class vision model, light enough for low VRAM, good for basic screen descriptions. |
| Gemma 3 4B | `ollama pull gemma3:4b` | Good next test | Multimodal and still relatively small, but may spill to CPU on 4 GB VRAM. |
| MiniCPM-V 2.6 | `ollama pull minicpm-v` | Better quality, heavier | Strong OCR/vision reputation, but the Ollama model is about 5.5 GB, so expect slower CPU/RAM fallback on your GPU. |
| Llama 3.2 Vision 11B | `ollama pull llama3.2-vision:11b` | Quality test only | About 7.8 GB in Ollama; likely too heavy for smooth use on 4 GB VRAM. |

Practical recommendation: use `moondream` for the VLM and `qwen2.5-coder:3b` for planning first. Upgrade the planner to `qwen2.5-coder:7b` if latency is acceptable.

## How To Check Your Machine Specs

PowerShell:

```powershell
nvidia-smi
[Environment]::ProcessorCount
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion
```

Windows UI:

```text
Ctrl + Shift + Esc -> Performance tab -> CPU, Memory, GPU
Win + R -> dxdiag -> Display tab
```

## Safety Limits

The MVP blocks or asks for user involvement around passwords, OTPs, banking, purchases, deletes, installs, security settings, unknown targets, low confidence, and destructive hotkeys. `--execute` is still real desktop control, so test in Notepad, Calculator, local files, and harmless webpages first.

## Next Features

- Draw an overlay showing the next click before executing.
- Add Playwright for browser tasks instead of using raw mouse clicks.
- Add a local verifier model that compares before/after screenshots.
- Add voice pause/resume with whisper.cpp and Silero VAD.
- Add teach-by-demonstration to record a workflow and replay it.
- Add app-specific adapters for VS Code, File Explorer, browser, and terminal.
