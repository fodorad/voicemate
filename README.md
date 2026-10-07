<div align="center">

<img src="https://raw.githubusercontent.com/fodorad/voicemate/main/docs/assets/logo.svg" alt="voicemate" width="200"/>

# voicemate

**A private voice assistant that runs entirely on your Mac, in English and Hungarian**

[![CI](https://github.com/fodorad/voicemate/actions/workflows/ci.yml/badge.svg)](https://github.com/fodorad/voicemate/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fodorad/voicemate/graph/badge.svg)](https://codecov.io/gh/fodorad/voicemate)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=githubpages)](https://fodorad.github.io/voicemate/)
[![GitHub Release](https://img.shields.io/github/v/release/fodorad/voicemate?color=purple)](https://github.com/fodorad/voicemate/releases)
<br/>
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
<br/>
[![LangGraph](https://img.shields.io/badge/agent-LangGraph-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![NiceGUI](https://img.shields.io/badge/UI-NiceGUI-5898D4)](https://nicegui.io)
[![Apple Silicon](https://img.shields.io/badge/runs%20on-Apple%20Silicon%20(MLX)-000000?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)

</div>

---

A private voice assistant that runs entirely on your Mac and speaks Hungarian and English.
You talk, it listens, thinks, looks things up when needed, remembers what you already
researched together, and answers out loud. You can interrupt it at any time by speaking.

- **Local:** speech recognition, language model, memory and speech synthesis run on the
  machine. Only explicit tool calls (web search, page fetch, arXiv, weather) go online, and
  each one is marked in the UI.
- **Streaming:** speech starts while the answer is still being written; typical chat turns
  start speaking about 1-3 s after you stop talking.
- **Natural turn-taking:** interrupt it any time; it stops, remembers what you already heard,
  and treats your words as a correction or a new question. Asks before risky actions.
- **Remembers:** facts about you, past conversations and a research cache. Asking the same
  research question again is answered from memory, with the date it was found.
- **Transparent:** a browser UI shows each stage live: listening, transcribing, checking
  memory, thinking, using a tool, writing, turning it into speech, speaking.

The assistant's name ("Ava" by default) is a setting in `config/voicemate.toml`.

## How it works

```
browser mic ─► VAD (Silero) ─► ASR (Parakeet v3, MLX) ─► language detection
                                                              │
      ┌───────────────────────── LangGraph agent (Ollama) ◄───┘
      │   recall memory (LanceDB + bge-m3) → model ⇄ tools
      │   tools: web search, fetch page, arXiv + paper download, weather, time,
      │          calculator, notes/reminders, files in ~/Documents and ~/Downloads
      ▼
sentence chunker ─► TTS (Piper for Hungarian, Kokoro for English) ─► browser speaker
```

See [docs/architecture.md](docs/architecture.md) for diagrams of the LangGraph agent, a
spoken turn, interruptions and the research memory.

## What you can ask it

- "What time is it?", "What's 18% of 2,450?", "What's the weather tomorrow?" (time,
  calculator and Open-Meteo weather for your configured place)
- "Remember that I'm looking for ML jobs in Budapest." / "What do you know about me?"
- "Add a todo: call the dentist." / "Remind me to call Anna at 5." (notes, todos and spoken
  reminders)
- "Search for recent papers on mixture of experts, and save the best one." (web search, page
  fetch, arXiv; a PDF lands in `~/Downloads/voicemate-papers` and its text in memory)
- "Save these notes to plan.md in Documents." (asks before replacing an existing file)

## Requirements

- Apple Silicon Mac (developed on an M4 with 32 GB), macOS
- [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com)
- About 15 GB of free disk space for the models

## Quick start

```bash
make dev       # create .venv (Python 3.13) and install everything
make models    # pull the Ollama model and download ASR, TTS and embedding weights
make run       # start the UI on http://127.0.0.1:8080 and open it
```

Click **Start microphone** (or press **Ctrl+M**), allow microphone access, and talk. You can
also type.

- **Ctrl+M** turns the microphone on and off. Turning it off means "I'm done": whatever it
  heard so far is answered. (Cmd+M can't be used: Chrome minimizes the window before the
  page sees it.)
- **Hold** (**Ctrl+H**) lets you think out loud: pauses no longer end your turn, and your
  words collect in one bubble. Press **I'm finished** (or turn the microphone off) and
  everything is answered at once. Pressing Hold also cuts off an answer in progress.
- Speaking while Ava talks interrupts her: she stops at once, remembers what you heard, and
  treats your words as a correction or a new question.
- The conversation continues across page reloads and restarts; **New conversation** starts
  a fresh one. Long-term memory (facts, research, notes) is kept either way.

Other commands:

```bash
make chat      # text-only chat in the terminal (same agent, tools and memory)
make report    # latency and error summary of your recent turns
make bench     # ASR / TTS / LLM benchmarks; voice samples in temp/bench/tts/index.html
```

## Configuration

Everything lives in [`config/voicemate.toml`](config/voicemate.toml) (or the file named in
`$VOICEMATE_CONFIG`). The most useful settings:

| Setting | Default | Notes |
|---|---|---|
| `llm.profile` | `fast` | one model for everything: `fast` (gemma4:e4b, ~11 GB), `gemma` (gemma4:26b-mlx, ~24 GB, best answers), `qwen` (qwen3.8:27b-mlx, ~27 GB, slowest). Try one with `make run PROFILE=gemma`; a startup memory check falls back to `fast` if it would not fit |
| `assistant.name` | `Ava` | persona name, plus per-language pronunciation hints |
| `assistant.default_language` | `en` | language before your language is detected, and the fallback voice |
| `tts.hu.voice` | `hu_HU-anna-medium` | other Piper voices: `hu_HU-berta-medium`, `hu_HU-imre-medium` |
| `vad.min_silence_ms` | `700` | pause that ends your turn; lower is snappier, higher interrupts you less |
| `search.backend` | `ddgs` | DuckDuckGo, no key; `searxng` for a self-hosted instance |
| `files.roots` | `~/Documents`, `~/Downloads` | the only folders the file tools may read and write |
| `files.papers_dir` | `~/Downloads/voicemate-papers` | where downloaded papers are saved |
| `location` | Gyöngyös | default place for weather |

## Privacy

- Conversations, memory, notes and logs stay in `data/` (git-ignored). Raw audio is not
  stored unless `save_audio = true`.
- Hugging Face Hub is switched to offline mode once the models are cached.
- Web content is treated as untrusted data; the assistant is instructed never to follow
  instructions found in pages or files.
- File tools are confined to the configured folders and cannot delete. Replacing an
  existing file pauses the agent and asks you (dialog or voice): that is LangGraph's
  human-in-the-loop `interrupt()`, so a malicious web page cannot make it overwrite files.
- Opening a page that no search result listed pauses the agent and asks you first, so a
  malicious page cannot make it request an address that carries your data out.
- The server only answers to `127.0.0.1`/`localhost` (DNS-rebinding protection), and
  fetched URLs must resolve to public addresses (no access to your LAN or local services).

## Development

The project is built with an AI-native SDLC: [intent](docs/sdlc/intent.md) →
[spec with measurements](docs/sdlc/spec.md) → build (TDD) → test → deploy → maintain.

```bash
make check             # lint (ruff) + type-check (ty) + unit tests with coverage + docs
make test-integration  # also runs the tests that need Ollama, models, network and `say`
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Documentation: https://fodorad.github.io/voicemate/

## Troubleshooting

- **Replies are slow:** run `make report`. With the `gemma` or `qwen` profile, check memory
  pressure in Activity Monitor; use `PROFILE=fast` or close memory-hungry apps.
- **Stopping and restarting the microphone:** if it ever stops reacting, reload the page;
  your conversation is kept.
- **"Turn failed: … connection"** in the UI: Ollama is not running; start the Ollama app.
- **No sound:** browsers only play audio after you interact with the page; click
  **Start microphone** or send a message once.
- **It interrupts itself:** use headphones, or raise `vad.barge_in_threshold`.

## License

MIT, Ádám Fodor (fodorad201@gmail.com)
