"""Command line: ``python -m voicemate {serve,chat,models,bench,report}``."""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import logging
import signal
import sys
from collections.abc import Callable
from typing import TextIO

from voicemate.config import Config, load_config
from voicemate.events import ConfirmEvent, ErrorEvent, Event, TokenEvent, ToolEvent

logger = logging.getLogger("voicemate")

#: Process-level stderr; diagnostics still work when ``sys.stderr`` is redirected.
STDERR_FD: int = 2


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def render_event(event: Event, out: TextIO) -> None:
    """Show one session event in the terminal (text chat)."""
    if isinstance(event, TokenEvent):
        out.write(event.text)
    elif isinstance(event, ToolEvent) and event.phase == "end":
        summary = f", {event.summary}" if event.summary else ""
        out.write(f"\n  [{event.name}: {event.source}{summary}]\n")
    elif isinstance(event, ConfirmEvent) and event.open:
        out.write(f"\n? {event.question} (answer below)\n")
    elif isinstance(event, ErrorEvent):
        out.write(f"\n! {event.message}\n")
    out.flush()


async def _chat(config: Config) -> None:  # pragma: no cover - interactive, needs Ollama
    """Text-only REPL on the same session logic as the voice UI (no ASR/TTS)."""
    from voicemate.pipeline.orchestrator import VoiceSession
    from voicemate.runtime import Runtime

    runtime = await Runtime.load(config, speech=False)
    session = VoiceSession(runtime)
    events = session.bus.subscribe()
    out = sys.stdout

    async def show() -> None:
        while True:
            render_event(await events.get(), out)

    printer = asyncio.create_task(show())
    out.write(f"{config.assistant.name} (text mode). Empty line to quit.\n")
    try:
        while text := (await asyncio.to_thread(input, "\n> ")).strip():
            await session.submit_text(text)
            await session.wait_idle()
    finally:
        printer.cancel()
        await session.close()
        await runtime.aclose()


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(prog="voicemate", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="run the voice UI on localhost (default)")
    serve.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    chat = commands.add_parser("chat", help="text-only chat in the terminal")
    for sub in (serve, chat):
        sub.add_argument("--profile", help="LLM profile from [llm.profiles], e.g. fast or gemma")
    commands.add_parser("models", help="download every model the configuration needs")
    bench = commands.add_parser("bench", help="run benchmarks (macOS)")
    bench.add_argument("what", choices=["asr", "tts", "llm", "all"], nargs="?", default="all")
    commands.add_parser("report", help="summarize turn metrics against the latency budgets")
    return parser


def _report(config: Config, _args: argparse.Namespace) -> None:
    from voicemate.pipeline.orchestrator import turns_log_path
    from voicemate.report import summarize

    sys.stdout.write(summarize(turns_log_path(config.data_path)).render() + "\n")


def _serve(config: Config, args: argparse.Namespace) -> None:  # pragma: no cover - blocking
    from voicemate.ui.app import serve

    serve(config, open_browser=not getattr(args, "no_browser", False))


def _chat_command(config: Config, _args: argparse.Namespace) -> None:  # pragma: no cover
    asyncio.run(_chat(config))


def _models(config: Config, _args: argparse.Namespace) -> None:  # pragma: no cover - downloads
    from voicemate.models import ensure_all

    ensure_all(config)


def _bench(config: Config, args: argparse.Namespace) -> None:  # pragma: no cover - needs models
    from voicemate.bench import run

    for name in ("asr", "tts", "llm") if args.what == "all" else (args.what,):
        getattr(run, f"bench_{name}")(config)


#: Command name → handler.
COMMANDS: dict[str, Callable[[Config, argparse.Namespace], None]] = {
    "serve": _serve,
    "chat": _chat_command,
    "models": _models,
    "bench": _bench,
    "report": _report,
}


def main(argv: list[str] | None = None) -> None:
    """Entry point of the ``voicemate`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    # Diagnostics for a stuck process: `kill -USR1 <pid>` prints every thread's stack.
    faulthandler.register(signal.SIGUSR1, file=STDERR_FD, all_threads=True)
    config = load_config()
    if profile := getattr(args, "profile", None):
        if profile not in config.llm.profiles:
            parser.error(f"unknown profile {profile!r}; choose from {sorted(config.llm.profiles)}")
        config.llm.profile = profile
    COMMANDS[args.command or "serve"](config, args)


if __name__ in ("__main__", "__mp_main__"):  # pragma: no cover
    main()
