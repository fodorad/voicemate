"""Sphinx configuration for the voicemate documentation."""

project = "voicemate"
copyright = "2026, Ádám Fodor"
author = "Ádám Fodor"
release = "latest"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# ── sphinx-autoapi (reads the source, so heavy model dependencies need not import) ────────

autoapi_dirs = ["../voicemate"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"
autoapi_add_toctree_entry = True
# Browser assets are not Python modules.
autoapi_ignore = ["*/ui/static/*"]

# ── Napoleon (Google docstrings) ───────────────────────────────────────────────────────────

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render `Attributes:` sections as fields; autoapi already documents the attributes.
napoleon_use_ivar = True

# ── MyST ───────────────────────────────────────────────────────────────────────────────────

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
# Render ```mermaid fences (also shown natively on GitHub) as diagrams.
myst_fence_as_directive = ["mermaid"]
suppress_warnings = ["myst.xref_missing"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── Theme ──────────────────────────────────────────────────────────────────────────────────

html_theme = "furo"
html_title = "voicemate"
html_baseurl = "https://fodorad.github.io/voicemate/"
