# Local Desktop Agent

A human-in-the-loop desktop automation agent. You give it a task in plain English; it **plans** the task into ordered subgoals using an LLM, then **executes one subgoal at a time**, pausing for your approval before each.

It reads the screen using **Windows UI Automation (UIA)** to build a list of interactive controls, overlays them as a "Set of Marks" (`*_marks.png`), and passes the marked screenshot to a vision model (VLM) for precise grounding. The planner and VLM can run via **Groq** (recommended), **Hugging Face Serverless**, or **Ollama** (local).

```text
task text
  -> LLM plans ordered subgoals
  -> for each subgoal:
       ask you: [E]xecute / [S]kip / [Q]uit
       loop:  screenshot -> UIA -> Set of Marks image -> VLM
              -> planner picks ONE action
              -> safety gate -> execute (PyAutoGUI) -> verify
       until the subgoal is done, blocked, or the step budget is spent
  -> summary
```

## Install

1. Clone the repository and navigate into it:
   ```powershell
   cd C:\Vision_Chatbot
   ```
2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

> [!NOTE]
> Main dependencies are `Pillow`, `PyAutoGUI`, `requests`, and `uiautomation` (Windows-only, used to retrieve interactive desktop controls). This agent must be run in an interactive Windows desktop session (not via SSH or background service).

---

## Config & Model Providers

The agent picks the first available provider in this order: **Groq** > **Hugging Face** > **Ollama** (local).

### 1. Groq Cloud (Recommended)

Groq offers the most generous free tier: **~30 requests/minute**, **~14,400 requests/day** — orders of magnitude more than Hugging Face's free tier.

1. Create a free API key at [Groq Console](https://console.groq.com/keys).
2. Add it to your `.env` file in the project root:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```
3. When `GROQ_API_KEY` is set, the agent automatically defaults to the following Groq models:
   - **Planner:** `llama-3.3-70b-versatile`
   - **VLM:** `llama-3.2-90b-vision-preview`

*Alternatively, provide your key at runtime: `--groq-key gsk_...`*

### 2. Hugging Face Serverless Inference

> [!WARNING]
> The Hugging Face free tier has very aggressive rate limits (~100 requests/month per model). A single task can exhaust this quota. Consider using Groq instead.

1. Create a free API token at [Hugging Face Tokens](https://huggingface.co/settings/tokens).
2. Add it to your `.env` file:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```
3. When `HF_TOKEN` is set (and no `GROQ_API_KEY`), the agent defaults to:
   - **Planner:** `Qwen/Qwen2.5-7B-Instruct`
   - **VLM:** `Qwen/Qwen3-VL-30B-A3B-Instruct`

*Alternatively, provide your token at runtime: `--hf-token hf_...`*

### 3. Local Ollama

If no cloud API key is detected, the agent falls back to Ollama.
1. Download and start [Ollama](https://ollama.com).
2. Start the Ollama server:
   ```powershell
   ollama serve
   ```
3. Pull the default planner and vision models:
   ```powershell
   ollama pull qwen2.5:7b       # planner (strong at structured JSON)
   ollama pull qwen2.5vl:7b     # vision model (screen understanding + click grounding)
   ```

---

## Run

Run the agent from the project root (`C:\Vision_Chatbot`) with your virtual environment activated.

### Quick Start (with Groq or HF key in `.env`)

**See the plan only** (no screen control or VLM invocation):
```powershell
python -m desktop_agent --plan-only "Open Notepad and type hello"
```

**Dry run** (plans and identifies target controls, without moving/clicking mouse/keyboard):
```powershell
python -m desktop_agent "Open Notepad and type hello"
```

**Execute for real** (asks you for confirmation before executing each subgoal):
```powershell
python -m desktop_agent --execute "Open Notepad and type hello"
```

**Override default models**:
```powershell
python -m desktop_agent --execute --planner-model "meta-llama/Llama-3.2-3B-Instruct" --vlm-model "llama-3.2-11b-vision-preview" "Open Notepad and type hello"
```

### Using Ollama (If no API keys are set)

```powershell
python -m desktop_agent --plan-only "Open Notepad and type hello"
python -m desktop_agent "Open Notepad and type hello"
python -m desktop_agent --execute "Open Notepad and type hello"
```

---

## Options

```text
--plan-only                 produce and print the plan, then stop
--execute                   actually move/click/type (default is dry-run)
--hitl subgoal|action|off   confirm before each subgoal (default), each action, or nothing
--max-steps-per-subgoal N   max actions per subgoal (default 6)
--use-vlm auto|always|never use the vision model never, always, or auto (default always)
--grounding uia             where click targets come from (default uia)
--planner-model NAME        model name (default: llama-3.3-70b-versatile for Groq, Qwen/Qwen2.5-7B-Instruct for HF, qwen2.5:7b for Ollama)
--vlm-model NAME            vision model (default: llama-3.2-90b-vision-preview for Groq, Qwen/Qwen3-VL-30B-A3B-Instruct for HF, qwen2.5vl:7b for Ollama)
--ollama-host URL           Ollama server URL (default http://localhost:11434)
--hf-token TOKEN            Hugging Face API token (overrides HF_TOKEN in .env)
--groq-key KEY              Groq API key (overrides GROQ_API_KEY in .env)
--verbose-logs              stream structured logs to console
```

---

## How it works

- **Planning** (`planner.py`) — The LLM decomposes the task into 2–6 verifiable subgoals, each with a `done_when` condition. If parsing/generation fails, it falls back to a single-step plan.
- **Perception** (`perception/`) — Captures a screenshot, walks the active window's Windows UI Automation (UIA) tree to collect interactive candidates (buttons, edits, menus, checkboxes), assigns integer marks starting from 1, draws numbered boundaries on the screenshot (`*_marks.png`), and (if enabled) runs the vision model (VLM) to describe the layout or ground instructions.
- **Action planning** (`planner.py`) — The LLM selects the next action (e.g. `open_app`, `click`, `type`, `hotkey`, `scroll`, `wait`, `ask_user`, `finish`). For clicks/interaction, it specifies the target mark ID. The VLM acts as a grounding bridge to map natural-language target descriptions to UIA control coordinates.
- **Safety** (`safety.py`) — Inspects proposed actions before execution, hard-blocking sensitive data entry (passwords, credentials, payment fields) and requiring approval for critical system events.
- **Verification** (`verifier.py`) — Compares screenshots and control states before and after an action to check for progress.

---

## Project layout

```text
desktop_agent/
  cli.py            argument parsing -> Config -> Agent
  config.py         Config dataclass (environment variable loading + validation)
  agent.py          execution orchestration and perceive/act/verify loops
  planner.py        task planner (subgoal planning and next action planner)
  actions.py        atomic actions wrapper using PyAutoGUI
  safety.py         safety policy evaluation
  verifier.py       post-action state transition verifier
  models.py         clients for Ollama, Hugging Face, and Groq APIs
  logging_utils.py  logging configuration and structured event tracer
  perception/
    screen.py       screenshot capture and sizing helper
    uia.py          Windows UI Automation tree traversal
    perceiver.py    perception controller, candidate builder, set-of-marks drawer
    vlm.py          VLM query controller for grounding and layout summary
tests/              unit tests (discoverable via: python -m unittest discover -s tests -v)
```

## Running Tests

Run the unit tests to verify the agent's core functionalities:
```powershell
python -m unittest discover -s tests -v
```
