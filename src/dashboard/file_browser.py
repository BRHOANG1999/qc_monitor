"""Reusable server-side file browser (modal) for the analysis tabs.

The analysis tabs historically scanned one configured folder
(``config.chronic_evoked.evoked_output_dir`` etc.). This lets a signed-in
user instead navigate the server's filesystem and pick a **folder** (the tab
then globs it) or specific **files**, per-view (nothing is written to config).

Usage in a tab:

    from src.dashboard import file_browser as fb
    # in layout():
    ...,
    fb.modal("chronic"),               # one modal instance per id-prefix
    html.Button("📂 Browse…", id="chronic-fb-open"),
    dcc.Store(id="chronic-fb-result"),  # provided by modal(); read this
    # in register_callbacks():
    fb.register(app, "chronic", open_btn_id="chronic-fb-open",
                exts=(".mat",), initial_path=_EVOKED_DIR)

The tab then adds its own callback on ``Input("chronic-fb-result","data")``
to apply ``{"path": <folder>}`` or ``{"files": [<paths>]}``.

SECURITY: reads run in the server process and browsing is intentionally
unrestricted (whole filesystem / all drives), per product decision. Every
listing is wrapped so a permission error just yields an empty/!partial list
rather than crashing. Do NOT expose this build to untrusted users.
"""

from __future__ import annotations

import os
import string

from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update

# ----- backend: directory listing ------------------------------------ #


def list_drives() -> list[str]:
    """Mounted drive roots (Windows ``C:\\`` ...). On POSIX returns ['/']."""
    if os.name != "nt":
        return ["/"]
    out: list[str] = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.isdir(root):
            out.append(root)
    return out


def list_dir(path: str | None, exts: tuple | None = None) -> dict:
    """``{cwd, parent, dirs, files}`` for *path*. Empty/None path -> the drive
    roots. *exts* (lowercase suffixes) filters the file list. Permission /
    missing-dir errors degrade to empty lists, never raise."""
    if not path:
        return {"cwd": "", "parent": None, "dirs": list_drives(), "files": []}
    cwd = os.path.abspath(path)
    dirs: list[str] = []
    files: list[str] = []
    try:
        with os.scandir(cwd) as it:
            for i, e in enumerate(it):
                if i > 50000:                       # bound a pathological dir
                    break
                try:
                    if e.is_dir(follow_symlinks=False):
                        dirs.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        if not exts or e.name.lower().endswith(exts):
                            files.append(e.path)
                except OSError:
                    continue
    except (PermissionError, OSError, FileNotFoundError, ValueError):
        pass
    dirs.sort(key=str.lower)
    files.sort(key=str.lower)
    # abspath keeps a trailing slash only on drive roots ("D:\\"), so
    # dirname(cwd) == cwd identifies a root -> parent is the drive list ("").
    p = os.path.dirname(cwd)
    parent = "" if (not p or p == cwd) else p
    return {"cwd": cwd, "parent": parent, "dirs": dirs, "files": files}


# ----- UI: modal + per-prefix callback wiring ------------------------- #

_OVERLAY = {
    "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
    "background": "rgba(0,0,0,0.6)", "zIndex": 2000,
    "display": "none", "alignItems": "center", "justifyContent": "center",
}
_PANEL = {
    "background": "#16161f", "border": "1px solid #3a3d4a",
    "borderRadius": "8px", "width": "min(760px, 92vw)",
    "maxHeight": "82vh", "display": "flex", "flexDirection": "column",
    "padding": "14px 16px", "boxShadow": "0 10px 40px rgba(0,0,0,0.5)",
}
_BTN = {"background": "#2a2d3a", "color": "#cfd0d6",
        "border": "1px solid #3a3d4a", "borderRadius": "6px",
        "padding": "6px 12px", "cursor": "pointer", "fontSize": "12px"}
_BTN_ACCENT = {**_BTN, "background": "#5e7ce2", "color": "white",
               "border": "none", "fontWeight": "700"}


def modal(prefix: str) -> html.Div:
    """The browser overlay + its Stores for *prefix*. Read ``{prefix}-fb-result``
    (``{"path": ...}`` or ``{"files": [...]}``) in the tab to apply a pick."""
    assert prefix, "prefix required"
    return html.Div([
        dcc.Store(id=f"{prefix}-fb-path"),
        dcc.Store(id=f"{prefix}-fb-result"),
        dcc.Store(id=f"{prefix}-fb-exts"),
        html.Div([
            html.Div([
                html.Span("📂 Browse server files", style={
                    "color": "white", "fontWeight": "700",
                    "fontSize": "14px"}),
                html.Button("✕", id=f"{prefix}-fb-cancel", n_clicks=0,
                            style={**_BTN, "padding": "2px 10px"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "8px"}),
            html.Div([
                html.Button("⤴ Up", id=f"{prefix}-fb-up", n_clicks=0,
                            style=_BTN),
                html.Span(id=f"{prefix}-fb-cwd", style={
                    "color": "#a0a0b0", "fontSize": "12px",
                    "fontFamily": "ui-monospace, monospace",
                    "marginLeft": "10px", "wordBreak": "break-all"}),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "8px"}),
            html.Div(id=f"{prefix}-fb-dirs", style={
                "overflowY": "auto", "maxHeight": "28vh",
                "borderBottom": "1px solid #2a2d3a", "marginBottom": "8px"}),
            html.Div("Files (tick to select):", style={
                "color": "#8a8d99", "fontSize": "11px", "marginBottom": "4px"}),
            dcc.Checklist(id=f"{prefix}-fb-files", options=[], value=[],
                          style={"overflowY": "auto", "maxHeight": "24vh"},
                          labelStyle={"display": "block", "color": "#cfd0d6",
                                      "fontSize": "12px",
                                      "fontFamily": "ui-monospace, monospace"},
                          inputStyle={"marginRight": "6px"}),
            html.Div([
                html.Button("Use this folder", id=f"{prefix}-fb-usefolder",
                            n_clicks=0, style=_BTN_ACCENT),
                html.Button("Use selected files", id=f"{prefix}-fb-usefiles",
                            n_clicks=0, style=_BTN),
            ], style={"display": "flex", "gap": "8px", "marginTop": "10px",
                      "justifyContent": "flex-end"}),
        ], style=_PANEL),
    ], id=f"{prefix}-fb-modal", style=_OVERLAY)


def _dir_row(prefix: str, path: str) -> html.Button:
    name = os.path.basename(path.rstrip("\\/")) or path
    return html.Button(
        "📁 " + name, id={"type": f"{prefix}-fb-dir", "path": path},
        n_clicks=0, style={
            "display": "block", "width": "100%", "textAlign": "left",
            "background": "transparent", "color": "#cfd0d6", "border": "none",
            "padding": "4px 6px", "cursor": "pointer", "fontSize": "12px",
            "fontFamily": "ui-monospace, monospace"})


def register(app, prefix: str, *, open_btn_id: str,
             exts: tuple | None = None, initial_path: str = "") -> None:
    """Wire one browser instance. *open_btn_id* opens it at *initial_path*;
    *exts* filters the file list (lowercase suffixes, e.g. ('.mat',))."""
    assert prefix and open_btn_id, "prefix + open_btn_id required"
    shown = {**_OVERLAY, "display": "flex"}
    exts_t = tuple(e.lower() for e in (exts or ()))

    @app.callback(
        Output(f"{prefix}-fb-modal", "style"),
        Output(f"{prefix}-fb-path", "data"),
        Output(f"{prefix}-fb-exts", "data"),
        Input(open_btn_id, "n_clicks"),
        prevent_initial_call=True,
    )
    def _open(_n):
        return shown, (initial_path or ""), list(exts_t)

    @app.callback(
        Output(f"{prefix}-fb-modal", "style", allow_duplicate=True),
        Input(f"{prefix}-fb-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def _close(_n):
        return _OVERLAY

    @app.callback(
        Output(f"{prefix}-fb-path", "data", allow_duplicate=True),
        Input({"type": f"{prefix}-fb-dir", "path": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _click_dir(_clicks):
        trig = callback_context.triggered_id
        if not trig or not any(_clicks):
            return no_update
        return trig.get("path")

    @app.callback(
        Output(f"{prefix}-fb-path", "data", allow_duplicate=True),
        Input(f"{prefix}-fb-up", "n_clicks"),
        State(f"{prefix}-fb-path", "data"),
        prevent_initial_call=True,
    )
    def _up(_n, path):
        return list_dir(path, exts_t or None)["parent"]

    @app.callback(
        Output(f"{prefix}-fb-cwd", "children"),
        Output(f"{prefix}-fb-dirs", "children"),
        Output(f"{prefix}-fb-files", "options"),
        Output(f"{prefix}-fb-files", "value"),
        Input(f"{prefix}-fb-path", "data"),
        prevent_initial_call=True,
    )
    def _render(path):
        info = list_dir(path, exts_t or None)
        rows = [_dir_row(prefix, d) for d in info["dirs"]]
        if not rows:
            rows = [html.Div("— no sub-folders —", style={
                "color": "#666", "fontSize": "11px", "padding": "4px 6px"})]
        opts = [{"label": os.path.basename(f), "value": f}
                for f in info["files"]]
        return (info["cwd"] or "Drives", rows, opts, [])

    @app.callback(
        Output(f"{prefix}-fb-result", "data"),
        Output(f"{prefix}-fb-modal", "style", allow_duplicate=True),
        Input(f"{prefix}-fb-usefolder", "n_clicks"),
        Input(f"{prefix}-fb-usefiles", "n_clicks"),
        State(f"{prefix}-fb-path", "data"),
        State(f"{prefix}-fb-files", "value"),
        prevent_initial_call=True,
    )
    def _use(_f, _s, path, files):
        trig = callback_context.triggered_id
        if trig == f"{prefix}-fb-usefolder":
            if not path:
                return no_update, no_update
            return {"path": path}, _OVERLAY
        if trig == f"{prefix}-fb-usefiles":
            if not files:
                return no_update, no_update
            return {"files": list(files)}, _OVERLAY
        return no_update, no_update
