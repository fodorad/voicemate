"""NiceGUI front end: one page per conversation plus the ``/ws/audio`` WebSocket.

The page renders the pipeline rail (one station per :class:`~voicemate.events.Lane`), the
conversation, an activity trace that marks every request that left the machine, the notes
panel and, when a tool needs permission, a confirmation dialog. Microphone and speaker audio
travel over a plain WebSocket so the binary stream stays out of NiceGUI's own socket.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import signal
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from nicegui import app, background_tasks, ui
from starlette.middleware.trustedhost import TrustedHostMiddleware

from voicemate.audio.codec import pack_audio_frame
from voicemate.config import Config
from voicemate.events import (
    AudioEvent,
    ConfirmEvent,
    ErrorEvent,
    Event,
    HoldEvent,
    InterruptEvent,
    Lane,
    LaneState,
    MetricsEvent,
    NotesChangedEvent,
    ReminderEvent,
    ReplyEvent,
    StatusEvent,
    TokenEvent,
    ToolEvent,
    TranscriptEvent,
)
from voicemate.pipeline.orchestrator import ChatLine, VoiceSession
from voicemate.runtime import Runtime

logger = logging.getLogger(__name__)

#: Static assets (CSS, audio worklet).
STATIC_DIR: Path = Path(__file__).parent / "static"

#: Rail stations: lane, phase colour group, label.
RAIL: tuple[tuple[Lane, str, str], ...] = (
    (Lane.LISTENING, "listen", "Listening"),
    (Lane.TRANSCRIBING, "listen", "Transcribing"),
    (Lane.RECALLING, "think", "Checking memory"),
    (Lane.THINKING, "think", "Thinking"),
    (Lane.TOOL, "think", "Using a tool"),
    (Lane.GENERATING, "think", "Writing the answer"),
    (Lane.SYNTHESIZING, "speak", "Turning it into speech"),
    (Lane.SPEAKING, "speak", "Speaking"),
)

#: Trace badge text per data source.
BADGES: dict[str, str] = {"web": "sent to web", "memory": "from memory", "local": "on device"}

#: Keyboard shortcuts, shown in the button tooltips (bound in ``static/audio.js``).
#: Cmd+M would be the Mac habit, but Chrome minimizes the window before the page sees it.
MIC_SHORTCUT: str = "Ctrl+M"
HOLD_SHORTCUT: str = "Ctrl+H"

#: Test audio file names accepted by ``?fake_mic=`` (no paths, no URLs).
_FAKE_MIC_NAME = re.compile(r"^[\w-]+\.wav$")


def asset_version() -> str:
    """Cache-busting token that changes whenever any static asset changes.

    NiceGUI serves static files as immutable for a year, so edited assets need new URLs.
    """
    return str(max(int(path.stat().st_mtime) for path in STATIC_DIR.iterdir()))


def _dump_tasks() -> None:  # pragma: no cover - manual diagnostics
    """Log the stack of every pending asyncio task (used for diagnosing stalls)."""
    for task in asyncio.all_tasks():
        buffer = io.StringIO()
        task.print_stack(file=buffer)
        logger.warning("Task %s:\n%s", task.get_name(), buffer.getvalue())


def handle_control_message(session: VoiceSession, text: str) -> None:
    """Apply a JSON control message from the browser; malformed messages are ignored."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed control message")
        return
    if not isinstance(data, dict):
        return
    kind, state = data.get("type"), data.get("state")
    if kind == "playback" and state == "started":
        session.playback_started()
    elif kind == "playback" and state == "ended":
        session.playback_ended()
    elif kind == "mic" and state == "off":
        session.microphone_stopped()


async def serve_audio(websocket: WebSocket, session: VoiceSession) -> None:
    """Bridge one browser's audio WebSocket and its session until it disconnects."""
    await websocket.accept()
    queue = session.bus.subscribe()

    async def forward() -> None:
        while True:
            event = await queue.get()
            if isinstance(event, AudioEvent):
                await websocket.send_bytes(pack_audio_frame(event.pcm, event.sample_rate))
            elif isinstance(event, InterruptEvent):
                await websocket.send_text(json.dumps({"type": "stop"}))

    sender = background_tasks.create(forward(), name="audio-out")
    try:
        while (message := await websocket.receive())["type"] != "websocket.disconnect":
            if message.get("bytes"):
                await session.feed_audio(message["bytes"])
            elif message.get("text"):
                handle_control_message(session, message["text"])
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        session.bus.unsubscribe(queue)


class ConversationPage:
    """Widgets of one browser tab and the event pump that updates them.

    Args:
        session: The tab's voice session.
        config: Application configuration.
        on_new_conversation: Starts a fresh conversation thread (the page then reloads).
    """

    def __init__(
        self,
        session: VoiceSession,
        config: Config,
        on_new_conversation: Callable[[], object] | None = None,
    ) -> None:
        self.session = session
        self.config = config
        self.on_new_conversation = on_new_conversation
        self.stations: dict[Lane, tuple[ui.element, ui.label, ui.label]] = {}
        self.started: dict[Lane, float] = {}
        self.reply: ui.label | None = None
        self.last_user: ui.label | None = None
        self.mic_on = False
        # Subscribe now, not when the pump task first runs, so no early event is lost.
        self._events = session.bus.subscribe()
        self._handlers: dict[type, Callable[[Any], None]] = {
            StatusEvent: self._status,
            TranscriptEvent: self._transcript,
            TokenEvent: self._token,
            ReplyEvent: self._reply_done,
            ToolEvent: self._tool,
            ErrorEvent: self._error,
            ReminderEvent: self._reminder,
            NotesChangedEvent: self._notes_changed,
            MetricsEvent: self._metrics,
            ConfirmEvent: self._confirm,
            HoldEvent: self._hold_changed,
        }
        self._build()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.root = ui.element("div").classes("vm-shell")
        with self.root:
            self._build_header()
            self._build_rail()
            self._build_conversation()
            with ui.element("aside").classes("vm-side"):
                with ui.element("section"):
                    ui.html("<h2>What happened</h2>", sanitize=False)
                    self.trace = ui.element("div")
                with ui.element("section"):
                    ui.html("<h2>Notes and reminders</h2>", sanitize=False)
                    self.notes_view()
            self._build_confirm_dialog()

    def _build_header(self) -> None:
        with ui.element("header").classes("vm-top"):
            ui.label(self.config.assistant.name).classes("vm-name")
            self.lang = ui.label(self.config.assistant.default_language).classes("vm-lang")
            self.lang.props('aria-label="Language"')
            choice = self.session.runtime.model_choice
            if choice is not None:
                with ui.label(choice.model).classes("vm-lang"):
                    ui.tooltip(f"{choice.profile} profile: {choice.reason}")
            ui.element("div").classes("vm-spacer")
            new = ui.button("New conversation", on_click=self._new_conversation)
            new.classes("vm-btn").props("flat no-caps")
            new.tooltip("Forget this conversation's history; long-term memory stays")
            self.hold = ui.button("Hold", on_click=self._toggle_hold)
            self.hold.classes("vm-btn vm-hold").props("flat no-caps")
            self.hold.tooltip(
                f"Hold until you're finished ({HOLD_SHORTCUT}): pause as long as you like, then "
                "press again and everything you said is answered at once"
            )
            self.mic = ui.button("Start microphone").classes("vm-btn vm-mic")
            self.mic.props("flat no-caps")
            self.mic.tooltip(f"Turn the microphone on or off ({MIC_SHORTCUT})")
            self.mic.on(
                "click",
                self._mic_toggled,
                js_handler="async () => emit(await window.voicemate.toggleMic())",
            )

    def _build_rail(self) -> None:
        rail = (
            ui.element("nav").classes("vm-rail").props('aria-label="What the assistant is doing"')
        )
        with rail:
            for index, (lane, phase, label) in enumerate(RAIL):
                station = ui.element("div").classes("vm-station").props(f'data-phase="{phase}"')
                if index == len(RAIL) - 1:
                    station.classes(add="is-last")
                with station:
                    ui.element("span").classes("vm-dot")
                    ui.label(label)
                    ms = ui.label("").classes("vm-ms")
                    detail = ui.label("").classes("vm-detail")
                self.stations[lane] = (station, ms, detail)
            self.latency = ui.label("").classes("vm-latency").style("white-space: pre-line")

    def _build_conversation(self) -> None:
        with ui.element("main").classes("vm-talk"):
            self.log = ui.element("div").classes("vm-log").props('aria-live="polite"')
            with self.log:
                self.empty = ui.label(
                    f"Start the microphone ({MIC_SHORTCUT}) and talk, in English or "
                    "Hungarian. You can also type below. Interrupt any time by speaking. "
                    f"Need time to think? Press Hold ({HOLD_SHORTCUT}) and pause freely."
                ).classes("vm-empty")
            self.partial = ui.label("").classes("vm-partial")
            with ui.element("div").classes("vm-compose"):
                self.input = ui.input(placeholder="Type a message").classes("vm-input")
                self.input.props("borderless dense")
                gesture = "() => { window.voicemate.ensureAudio(); emit(); }"
                self.input.on("keydown.enter", self.send, js_handler=gesture)
                send = ui.button("Send").classes("vm-btn").props("flat no-caps")
                send.on("click", self.send, js_handler=gesture)

    def _build_confirm_dialog(self) -> None:
        with ui.dialog().props("persistent") as self.dialog, ui.card():
            self.question = ui.label("")
            with ui.row():
                ui.button("Allow", on_click=lambda: self.answer("yes")).props("no-caps")
                ui.button("Deny", on_click=lambda: self.answer("no")).props("flat no-caps")

    @ui.refreshable_method
    def notes_view(self) -> None:
        """Open notes, todos and reminders with a checkbox to complete them."""
        notes = self.session.runtime.notes
        items = notes.entries() if notes is not None else []
        if not items:
            ui.label("Nothing yet. Try “remind me to call Anna at 5”.").classes("vm-note")
        for note in items:
            with ui.element("div").classes("vm-note"):
                ui.checkbox(on_change=lambda _e, i=note.id: self._complete(i))
                ui.label(note.text)
                if note.due:
                    ui.label(note.due.replace("T", " ")).classes("text-xs")

    # ── user actions ─────────────────────────────────────────────────────────

    def _mic_toggled(self, event: Any) -> None:
        wanted_on = not self.mic_on
        self.mic_on = bool(event.args)
        self.mic.set_text("Stop microphone" if self.mic_on else "Start microphone")
        if self.mic_on:
            self.mic.classes(add="is-on")
        else:
            self.mic.classes(remove="is-on")
        if wanted_on and not self.mic_on:
            ui.notify("The browser did not allow microphone access.", type="warning")

    async def _toggle_hold(self) -> None:
        if self.session.holding:
            await self.session.release()
        else:
            await self.session.hold()

    def _hold_changed(self, event: HoldEvent) -> None:
        """Follow the session, which also ends a hold when the microphone is turned off."""
        self.hold.set_text("I'm finished" if event.holding else "Hold")
        if event.holding:
            self.hold.classes(add="is-on")
        else:
            self.hold.classes(remove="is-on")

    def _new_conversation(self) -> None:
        if self.on_new_conversation is not None:
            self.on_new_conversation()
        ui.navigate.reload()

    def show_history(self, lines: list[ChatLine]) -> None:
        """Render the earlier messages of a reopened conversation."""
        for line in lines:
            label = self._message(line.text, line.role)
            if line.interrupted:
                label.classes(add="interrupted")
            if line.role == "user":
                self.last_user = label

    async def send(self) -> None:
        """Submit the text box as a turn."""
        text = (self.input.value or "").strip()
        self.input.value = ""
        if text:
            await self.session.submit_text(text)

    async def answer(self, text: str) -> None:
        """Answer the pending confirmation (dialog buttons)."""
        self.dialog.close()
        await self.session.answer_confirmation(text)

    def _complete(self, note_id: int) -> None:
        if self.session.runtime.notes is not None:
            self.session.runtime.notes.complete(note_id)
        self.notes_view.refresh()

    # ── events ───────────────────────────────────────────────────────────────

    async def pump(self) -> None:
        """Apply every session event to the page until the page goes away."""
        try:
            while True:
                event = await self._events.get()
                try:
                    with self.root:  # background tasks have no slot context of their own
                        self.apply(event)
                except Exception:  # a UI glitch must never stop the conversation
                    logger.exception("Failed to render %s", type(event).__name__)
        finally:
            self.session.bus.unsubscribe(self._events)

    def apply(self, event: Event) -> None:
        """Render one event (events without a handler, such as audio, are ignored)."""
        handler = self._handlers.get(type(event))
        if handler is not None:
            handler(event)

    def _status(self, event: StatusEvent) -> None:
        if event.lane is Lane.LISTENING and event.state is LaneState.ACTIVE:
            for lane in self.stations:
                self._set_station(lane, LaneState.IDLE)
            self.started.clear()
        self._set_station(event.lane, event.state)
        _, ms, detail = self.stations[event.lane]
        if event.state is LaneState.ACTIVE:
            self.started.setdefault(event.lane, event.ts)
            ms.set_text("")
        elif event.state is LaneState.DONE and event.lane in self.started:
            ms.set_text(f"{(event.ts - self.started[event.lane]) * 1000:.0f} ms")
        if event.lane is Lane.TOOL and event.detail:
            detail.set_text(event.detail)

    def _set_station(self, lane: Lane, state: LaneState) -> None:
        station, ms, detail = self.stations[lane]
        station.classes(remove="is-active is-done")
        if state is LaneState.ACTIVE:
            station.classes(add="is-active")
        elif state is LaneState.DONE:
            station.classes(add="is-done")
        else:
            ms.set_text("")
            detail.set_text("")

    def _transcript(self, event: TranscriptEvent) -> None:
        if not event.final:
            self.partial.set_text(event.text)
            return
        self.partial.set_text("")
        self.lang.set_text(event.lang or self.lang.text)
        if event.merged and self.last_user is not None:
            self.last_user.set_text(event.text)  # the cut-off question, now completed
        else:
            self.last_user = self._message(event.text, "user")

    def _token(self, event: TokenEvent) -> None:
        if self.reply is None:
            self.reply = self._message("", "assistant")
        self.reply.set_text(self.reply.text + event.text)
        self._scroll()

    def _reply_done(self, event: ReplyEvent) -> None:
        if self.reply is not None and event.interrupted:
            self.reply.classes(add="interrupted")
        self.reply = None

    def _tool(self, event: ToolEvent) -> None:
        if event.phase == "end":
            text = f"{event.name}: {event.summary}" if event.summary else event.name
            self._trace(BADGES.get(event.source, event.source), event.source, text)

    def _error(self, event: ErrorEvent) -> None:
        self._trace("error", "error", event.message)
        ui.notify(event.message, type="negative")

    def _reminder(self, event: ReminderEvent) -> None:
        ui.notify(f"Reminder: {event.text}", type="info", timeout=0, close_button=True)
        self.notes_view.refresh()

    def _confirm(self, event: ConfirmEvent) -> None:
        if event.open:
            self.question.set_text(event.question)
            self.dialog.open()
        else:
            self.dialog.close()

    def _message(self, text: str, role: str) -> ui.label:
        if self.empty.visible:
            self.empty.set_visibility(False)
        with self.log:
            label = ui.label(text).classes(f"vm-msg {role}")
        self._scroll()
        return label

    def _scroll(self) -> None:
        self.log.client.run_javascript(
            f"const el = getHtmlElement({self.log.id}); if (el) el.scrollTop = el.scrollHeight"
        )

    def _trace(self, badge: str, kind: str, text: str) -> None:
        with self.trace:
            item = ui.element("div").classes("vm-trace-item")
            with item:
                ui.label(badge).classes(f"vm-badge {kind}")
                ui.label(text).style("display: inline")
        item.move(target_index=0)

    def _notes_changed(self, _event: NotesChangedEvent) -> None:
        self.notes_view.refresh()

    def _metrics(self, event: MetricsEvent) -> None:
        metrics = event.metrics
        lines = []
        if "first_audio_s" in metrics:
            lines.append(f"First words after {metrics['first_audio_s']:.1f} s")
        if "total_s" in metrics:
            lines.append(f"Whole turn {metrics['total_s']:.1f} s")
        if metrics.get("tools"):
            lines.append("Tools: " + ", ".join(metrics["tools"]))
        self.latency.set_text("\n".join(lines))


class Server:
    """Holds the runtime and the live sessions of this process.

    Args:
        config: Application configuration.
        runtime: An already loaded runtime (tests); otherwise loaded at startup.
    """

    def __init__(self, config: Config, runtime: Runtime | None = None) -> None:
        self.config = config
        self.runtime = runtime
        self.sessions: dict[str, VoiceSession] = {}

    @property
    def _thread_file(self) -> Path:
        return self.config.data_path / "conversation.json"

    def thread_id(self) -> str:
        """The current conversation; it survives page reloads and server restarts."""
        try:
            return str(json.loads(self._thread_file.read_text())["thread_id"])
        except (OSError, ValueError, KeyError, TypeError):
            return self.new_conversation()

    def new_conversation(self) -> str:
        """Start a fresh conversation thread (long-term memory is kept)."""
        thread = f"conversation-{uuid.uuid4().hex[:12]}"
        self._thread_file.parent.mkdir(parents=True, exist_ok=True)
        self._thread_file.write_text(json.dumps({"thread_id": thread}))
        return thread

    async def open_session(self) -> tuple[str, VoiceSession]:
        """A session for a new page, continuing the current conversation."""
        if self.runtime is None:
            raise RuntimeError("models are not loaded yet")
        sid = uuid.uuid4().hex
        session = VoiceSession(self.runtime, thread_id=self.thread_id())
        self.sessions[sid] = session
        session.start()
        return sid, session

    async def startup(self) -> None:  # pragma: no cover - loads every model
        """Load every model before the first page is served."""
        # Diagnostics: `kill -USR2 <pid>` logs the stack of every pending asyncio task.
        asyncio.get_running_loop().add_signal_handler(signal.SIGUSR2, _dump_tasks)
        if self.runtime is None:
            self.runtime = await Runtime.load(self.config)

    async def shutdown(self) -> None:
        """Close sessions and release the runtime."""
        for session in list(self.sessions.values()):
            await session.close()
        if self.runtime is not None:
            await self.runtime.aclose()

    def fake_mic_name(self, requested: str | None) -> str | None:
        """The ``?fake_mic=`` file to replay, if test audio is enabled and the name is safe."""
        if not requested or not self.config.ui.test_audio or not _FAKE_MIC_NAME.match(requested):
            return None
        return requested if (self.config.data_path / "fake_mic" / requested).is_file() else None

    async def build_page(self, fake_mic: str | None = None) -> ConversationPage | None:
        """Build the conversation page for the current client (the ``/`` route)."""
        client = ui.context.client
        version = asset_version()
        ui.add_head_html(f'<link rel="stylesheet" href="/static/app.css?v={version}">')
        ui.add_body_html(f'<script src="/static/audio.js?v={version}"></script>')
        if self.runtime is None:  # models are still loading: wait, then reload the page
            ui.label("Loading the speech and language models. This page opens by itself.")
            ui.timer(1.0, lambda: self.runtime is not None and ui.navigate.reload())
            return None
        sid, session = await self.open_session()
        page = ConversationPage(session, self.config, self.new_conversation)
        page.show_history(await session.restore())
        pump = background_tasks.create(page.pump(), name=f"ui-{sid}")
        choice = self.runtime.model_choice
        if choice is not None and (choice.fell_back or not choice.fits):
            ui.notify(f"Using {choice.model}: {choice.reason}", type="warning", multi_line=True)

        async def cleanup() -> None:
            pump.cancel()
            self.sessions.pop(sid, None)
            await session.close()

        client.on_delete(cleanup)
        await client.connected()
        ui.run_javascript(f"window.voicemate.connect({json.dumps(sid)})")
        if name := self.fake_mic_name(fake_mic):
            ui.run_javascript(f"window.voicemate.playFakeMic({json.dumps('/fake-mic/' + name)})")
        return page


def register(server: Server) -> None:  # pragma: no cover - wiring only, exercised by E2E
    """Attach middleware, routes, static files and the page to the NiceGUI app."""
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=server.config.ui.allowed_hosts)
    app.add_static_files("/static", STATIC_DIR)
    fake_mic_dir = server.config.data_path / "fake_mic"
    if server.config.ui.test_audio and fake_mic_dir.is_dir():
        app.add_static_files("/fake-mic", fake_mic_dir)
    app.on_startup(server.startup)
    app.on_shutdown(server.shutdown)

    @app.websocket("/ws/audio")
    async def audio_socket(websocket: WebSocket, sid: str) -> None:
        session = server.sessions.get(sid)
        if session is None:
            await websocket.close(code=4404)
            return
        await serve_audio(websocket, session)

    @ui.page("/")
    async def index(fake_mic: str | None = None) -> None:
        await server.build_page(fake_mic)


def serve(config: Config, open_browser: bool = True) -> None:  # pragma: no cover - blocking
    """Run the web UI (blocks until Ctrl+C)."""
    register(Server(config))
    ui.run(
        host=config.ui.host,
        port=config.ui.port,
        title=config.assistant.name,
        favicon=STATIC_DIR / "icon.svg",
        dark=None,
        reload=False,
        show=open_browser,
        show_welcome_message=True,
    )
