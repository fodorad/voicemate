# voicemate: notes for Claude

Private, local, bilingual (HU/EN) voice assistant. Package `voicemate/`; the persona name
is configuration only and must never appear in code.

## Workflow

- AI-native SDLC: intent in `docs/sdlc/intent.md`, design and measurements in
  `docs/sdlc/spec.md` (update it when a decision changes). Local plan in `temp/plan.md`
  (gitignored).
- `make check` = ruff lint + format check, ty, unittest with coverage (≥ 80 %), Sphinx `-W`.
- `make test-integration` runs model/network/`say` tests (`VOICEMATE_INTEGRATION=1`).
- Tests use `unittest`, random arrays and the deterministic protocol implementations in
  `tests/helpers.py` (ScriptedChatModel, ScriptedRecognizer, ToneSynthesizer,
  HashingEmbedder, StaticSearchBackend). No mocks or patches.
- Git: GitHub Flow (`feat/*` → PR → squash-merge); never add AI/Claude attribution.

## Architecture in one breath

`ui/app.py` (NiceGUI page + `/ws/audio`) → `pipeline/orchestrator.py` (`VoiceSession`: VAD →
ASR → LangGraph stream → `SpeechChunker` → TTS → browser) → `agent/graph.py` (recall →
agent ⇄ ToolNode) with tools in `agent/tools/`, memory in `memory/` (LanceDB + SQLite).
All UI state flows through `events.EventBus`.

## Lessons that cost time (keep them)

- **KV cache:** the system prompt must stay byte-identical across turns. Put anything that
  changes (time, language, memories) in `prompts.turn_context`, attached to the newest user
  message by `graph.with_context`. Breaking this costs 5-20 s per turn.
- **GPU is for the LLM:** keep ASR (`asr.device`) and embeddings (`memory.embed_device`)
  on the CPU. Embeddings never go through Ollama (it evicts the chat model).
- **RAM:** qwen3.6:35b (22 GB) swaps on the 32 GB Mac with normal apps open; the default is
  gemma4:e4b. Check `sysctl vm.swapusage` before blaming code for slowness.
- **Numbers stay digits** in replies; espeak reads them correctly, LLM spelling does not.
- LangGraph injects the run config only into a parameter literally named `config`.
- LangChain tools read annotations at runtime: do not move `RunnableConfig` imports into
  `TYPE_CHECKING` (ruff `TC` rules are disabled for this reason).
- `kokoro-onnx`'s espeak loader is broken on macOS; phonemes come from Piper's espeak.
- `huggingface_hub` reads `HF_HUB_OFFLINE` at import; `runtime.use_offline_hub_if_cached`
  also flips the module constant.
- NiceGUI serves static files as immutable for a year: asset URLs carry `asset_version()`.
- Never `await AudioContext.resume()` in the browser; it hangs without a user gesture.
- Stopping the mic must tear down the whole capture graph (source, worklet node, port);
  a half-disconnected worklet keeps sending silent frames that corrupt the next start.
- Cmd+M can't be a page shortcut: Chrome minimizes the window first. The mic uses Ctrl+M.
- Turning the mic off means "I'm done": flush the segmenter and answer; never drop speech.
  Hold (`VoiceSession.hold`/`release`) collects utterances into one turn.
- Small local models sometimes return an empty answer after a tool round; the agent
  retries once with `ANSWER_NUDGE`. Keep that when changing `graph.py`.
- Risky tools ask first via `agent.hitl.ask_user` (LangGraph `interrupt()`); the session
  resumes the graph with the user's words (`VoiceSession.answer_confirmation`).
- UI tests build `ConversationPage` in an offline NiceGUI `Client` (set `core.loop`);
  NiceGUI's `user_simulation` only works under pytest.
- Every session task goes through `VoiceSession._spawn` (the loop keeps only weak refs).
- Diagnostics: `kill -USR1 <pid>` dumps thread stacks, `kill -USR2 <pid>` dumps asyncio
  tasks (server only).
