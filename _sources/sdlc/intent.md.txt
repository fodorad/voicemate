# voicemate — Intent

- **Status:** Approved by delegation (Adam, 2026-09-28: "decide & document")
- **Owner:** Adam
- **Date:** 2026-09-28
- **SDLC stage:** 1 — Plan

## Problem

I want a personal assistant I can simply *talk to*, in Hungarian or English, the way I would talk to a colleague. It should listen, understand, look things up when needed, remember what we already discussed or researched, and answer out loud.

Cloud assistants don't fit:

- they send my conversations to third parties;
- they forget context between sessions;
- they don't handle Hungarian well.

The current prototype (repo formerly `NEXA`) runs locally, but it is:

- a blocking 5-second record → transcribe → generate → speak loop;
- Hungarian-only;
- without tools, memory or a UI;
- too slow to feel conversational.

## User

A single user (Adam) on a local Apple Silicon machine (M4, 32 GB), talking through a browser tab on localhost.

Typical topics:

- machine-learning research (papers, methods, comparisons);
- job search (roles, companies, preparation);
- health and lifestyle advice (informational);
- everyday questions (time, weather, calculations, notes, reminders).

## Goals

1. **Natural voice conversation.** Always listening: I speak, it notices when I stop, and it answers. I can interrupt it mid-sentence and it stops and listens.
2. **Bilingual.** It understands Hungarian and English and replies in the language I used.
3. **Local and private.** All models (speech recognition, language models, embeddings, speech synthesis) run on my machine through Ollama and local libraries. No cloud LLM or cloud speech service.
4. **Capable.** It can use tools:
   - web search, reading web pages and arXiv search;
   - weather, time/date and a calculator;
   - notes, todos and reminders;
   - read-only access to folders I whitelist.
5. **Remembers.**
   - It keeps short-term conversation context.
   - It builds long-term memory of facts about me and my preferences, summaries of past conversations, and a research cache.
   - What it has already researched is answered from memory, with the date it was found, instead of being searched again.
6. **Transparent.** A browser UI shows live what it is doing: listening, transcribing, recalling memory, thinking, using a tool, generating the answer, synthesizing speech and speaking. It also shows the transcript, the full written answer with sources, and which data left the machine.

## Success metrics (acceptance criteria)

| # | Metric | Target |
|---|---|---|
| M1 | End of my speech → first audio of the reply, chat turns, p50 | ≤ 2.5 s |
| M2 | Same, turns that use a tool or web research, p50 | ≤ 8 s, with a spoken filler within 2.5 s |
| M3 | Barge-in: I start talking → its audio stops | ≤ 300 ms |
| M4 | Reply language matches my language | ≥ 95 % of turns (20-turn HU/EN test session) |
| M5 | Repeated research question, in a new session, is answered from memory | 100 % with no new web request, and the stored date cited |
| M6 | Network egress | Only web search, page fetch, arXiv and weather requests leave the machine; each is visible in the UI |
| M7 | Tool calling works for scripted prompts (HU + EN) | ≥ 90 % correct tool choice and arguments |
| M8 | Hungarian speech output is intelligible and acceptable to me | Mean opinion score ≥ 3.5 / 5 in the TTS bake-off |
| M9 | Code quality | CI green; ruff, ty, unittest coverage ≥ 80 % on new code; Sphinx builds with no warnings |

## Privacy stance

- Conversations, memory, notes, embeddings and audio stay on disk under `data/`, which is gitignored. Raw audio is not stored by default.
- Outbound traffic is limited to search queries, fetched URLs, arXiv queries and weather lookups. Each one is logged and shown in the UI.
- Web content is treated as untrusted data: the assistant never follows instructions embedded in fetched pages.
- File access is read-only and confined to folders I whitelist in config.
- No secrets, API keys or absolute local paths are committed to the repository.

## Behavioural expectations

- Spoken replies are concise; the full answer, links and sources appear in the UI.
- Health answers are informational and recommend consulting a professional for anything medical or urgent.
- When it doesn't know, or a tool fails, it says so instead of making things up.
- If a turn exceeds its time budget, it apologises briefly and recovers; it never goes silent or hangs.

## Non-goals (v1)

- Wake word ("Hey Nexa").
- Access from a phone, remotely, or by several users.
- Voice cloning or custom voices.
- Shell commands, app control, or writing or deleting files outside its own notes store.
- Any cloud model or cloud speech API.
- Running on non-Apple-Silicon hardware (nice to have, not a target).

## Open questions (resolved in Stage 2 — Design)

1. Which speech-recognition model: parakeet-mlx, mlx-whisper-turbo, or gemma4 native audio? Decided by a word-error-rate and speed bake-off.
2. Which Hungarian voice: Piper, F5-TTS-hu, or MMS? Decided by a speed test plus my listening rating.
3. LLM split and memory strategy: gemma4:e4b resident and qwen3.6:35b loaded on demand, or another split? Decided by latency, tool-calling and peak-memory measurements.
4. Silence threshold for "I finished speaking": 500, 700 or 900 ms?
5. **Assistant persona name.** Renamed on 2026-10-07 (easier to say and write; recognized 7/7 in both languages, see spec §11). It is only the default persona name. It is a config value (`assistant.name` plus per-language pronunciation hints such as HU "Neksza"), fully decoupled from the code. The bake-off checks how ASR and TTS handle it; changing it later costs one config edit.

## Naming (decided 2026-09-28)

- **Project, repo and package: `voicemate`** (Python package `voicemate`, free on PyPI). `nexa` was rejected for two reasons: the name is taken on PyPI, and it clashes with Nexa AI's on-device SDK.
- **Assistant persona name:** configurable, default "Nexa". The code never refers to the persona name.

## Legacy code

The existing prototype (`voicemate/chat.py`, `voicemate/LLM|STT|TTS/`, `test/`, `requirements.txt`) is throwaway. It is deleted when the new project skeleton is scaffolded, and git history keeps it for reference.
