"""voicemate: a private, local, bilingual (Hungarian/English) voice assistant."""

from importlib.metadata import PackageNotFoundError, version

try:
    #: Installed package version.
    __version__: str = version("voicemate")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0"
