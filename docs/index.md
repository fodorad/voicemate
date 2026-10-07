```{image} assets/logo.svg
:alt: voicemate
:width: 160px
:align: center
```

# voicemate

A private voice assistant that runs entirely on your Mac and speaks English and Hungarian.
You talk, it listens, thinks, looks things up when needed, remembers what you already
researched together, and answers out loud.

- **Local:** speech recognition, language model, memory and speech synthesis run on the
  machine. Only explicit tool calls (web search, page fetch, arXiv, weather) go online, and
  each one is marked in the UI.
- **Streaming:** speech starts while the answer is still being written (about 1-3 s after
  you stop talking for chat turns).
- **Natural turn-taking:** interrupt it any time; it keeps what you heard and treats your
  words as a correction. **Hold** (Ctrl+H) lets you pause and think out loud; **I'm
  finished** answers everything at once. Turning the microphone off (Ctrl+M) answers what
  it heard so far.
- **Asks before risky actions:** replacing a file pauses the LangGraph agent and asks you.
- **Remembers:** facts, past conversations, notes and a research cache; the conversation
  itself survives page reloads and restarts.
- **One model for everything:** choose `fast`, `gemma` or `qwen`; a startup memory check
  falls back to a model that fits.
- **Transparent:** the browser UI shows each pipeline stage live.

```{image} ui/early-look.png
:alt: voicemate in the browser
:width: 100%
```

*A real session: pipeline stages with timings on the left, memory and tool trace on the right.*

## Quick start

```bash
make dev       # create .venv (Python 3.13) and install everything
make models    # pull the Ollama model and download ASR, TTS and embedding weights
make run       # start the UI on http://127.0.0.1:8080
```

Configuration lives in `config/voicemate.toml`; the README on GitHub lists the most useful
settings.

## How it works and how it was built

```{toctree}
:maxdepth: 1

architecture
sdlc/intent
sdlc/spec
```

The API reference below is generated from the docstrings.
