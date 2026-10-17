"""Control messages between the browser and the server, defined in one place.

Audio itself travels as binary WebSocket frames (see :mod:`voicemate.audio.codec`); everything
else is a small JSON text message. ``static/audio.js`` implements the browser side, and a
contract test keeps it in step with the definitions here.

Browser → server: :class:`MicOff`, :class:`PlaybackStarted`, :class:`PlaybackEnded`.
Server → browser: :class:`Stop`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MicOff:
    """The user turned the microphone off."""

    type = "mic"
    state = "off"


@dataclass(frozen=True)
class PlaybackStarted:
    """The browser started playing assistant audio."""

    type = "playback"
    state = "started"


@dataclass(frozen=True)
class PlaybackEnded:
    """The browser finished (or dropped) all queued assistant audio."""

    type = "playback"
    state = "ended"


@dataclass(frozen=True)
class Stop:
    """Barge-in: the browser must drop queued speech."""

    type = "stop"
    state = None


ControlMessage = MicOff | PlaybackStarted | PlaybackEnded | Stop

#: Every control message, for lookup and for the contract test against ``audio.js``.
CONTROL_MESSAGES: tuple[ControlMessage, ...] = (
    MicOff(),
    PlaybackStarted(),
    PlaybackEnded(),
    Stop(),
)


def dump(message: ControlMessage) -> str:
    """The JSON text sent over the WebSocket for ``message``."""
    body = {"type": message.type}
    if message.state is not None:
        body["state"] = message.state
    return json.dumps(body)


def parse(text: str) -> ControlMessage | None:
    """Decode a JSON text message.

    Returns:
        The message, or ``None`` (and a log line) when the text is malformed or unknown, so
        a bad client can never break the session.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed control message")
        return None
    if not isinstance(data, dict):
        logger.warning("Ignoring control message that is not an object")
        return None
    for message in CONTROL_MESSAGES:
        if data.get("type") == message.type and data.get("state") == message.state:
            return message
    logger.warning("Ignoring unknown control message %r", data)
    return None
