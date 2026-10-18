"""Terminal look and feel: banner, arrow-key prompts, panels and progress bars."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
from InquirerPy.utils import get_style
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

PINK, PURPLE, CYAN, DIM = "#ff79c6", "#bd93f9", "#8be9fd", "#7f849c"

PROMPT_STYLE = get_style({
    "questionmark": f"{PINK} bold",
    "answermark": f"{PINK} bold",
    "question": "bold",
    "answered_question": "",
    "answer": f"{CYAN} bold",
    "pointer": f"{PINK} bold",
    "marker": CYAN,
    "fuzzy_prompt": PINK,
    "fuzzy_info": DIM,
    "fuzzy_match": f"{CYAN} bold",
    "fuzzy_border": DIM,
    "instruction": f"{DIM} italic",
    "long_instruction": f"{DIM} italic",
    "separator": DIM,
    "skipped": DIM,
    "validator": "#f38ba8",
}, style_override=False)
def _short(title: str) -> str:
    """Menu titles are 'Label    explanation'; answers echo just the label."""
    return title.split("  ")[0].strip()


COMMON = dict(style=PROMPT_STYLE, qmark="◆", amark="◇", pointer="❯", border=False, mandatory=True)

LOGO = [
    "██╗    ██╗ █████╗ ██╗███████╗██╗   ██╗",
    "██║    ██║██╔══██╗██║██╔════╝██║   ██║",
    "██║ █╗ ██║███████║██║█████╗  ██║   ██║",
    "██║███╗██║██╔══██║██║██╔══╝  ██║   ██║",
    "╚███╔███╔╝██║  ██║██║██║     ╚██████╔╝",
    " ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚═╝      ╚═════╝ ",
]
GRADIENT = ["#ff79c6", "#f077d0", "#e075da", "#d173e4", "#c771ee", "#bd93f9"]


class Cancelled(Exception):
    """The user said no at the final confirmation."""


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def banner() -> None:
    width = len(LOGO[0])
    if console.width < width + 4:
        console.print(f"[bold {PINK}]WaifuExtractor[/] [{DIM}]v{__version__}[/]\n")
        return
    console.print()
    for line, color in zip(LOGO, GRADIENT):
        console.print(f"  [{color}]{line}[/]")
    console.print(f"  [bold {PURPLE}]{'E X T R A C T O R':^{width}}[/]")
    console.print(f"  [{DIM}]{f'v{__version__} · cut a character out of anime episodes':^{width}}[/]\n")


# --- messages ------------------------------------------------------------------

def step(msg: str) -> None:
    console.print(f"[{PINK}]›[/] {msg}")


def ok(msg: str) -> None:
    console.print(f"[green]✓[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]![/] {msg}")


def error(msg: str) -> None:
    err_console.print(f"[bold red]✗[/] {msg}")


def panel(rows: Sequence[tuple[str, str]], title: str, color: str = PINK) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {DIM}", justify="right")
    grid.add_column()
    for key, value in rows:
        grid.add_row(key, value)
    console.print(Panel(grid, title=f"[bold]{title}[/]", title_align="left", border_style=color,
                        padding=(1, 2), expand=False))


def table(columns: Sequence[tuple[str, str]], rows: Sequence[Sequence[Any]], title: str | None = None) -> None:
    """columns: (header, justify)."""
    t = Table(title=title, title_justify="left", title_style="bold", header_style=f"bold {PURPLE}",
              border_style=DIM, show_edge=False, pad_edge=False, box=None)
    for header, justify in columns:
        t.add_column(header, justify=justify)
    for row in rows:
        t.add_row(*(str(c) for c in row))
    console.print(t)


def progress(transient: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(style=PINK),
        TextColumn("{task.description}"),
        BarColumn(complete_style=PINK, finished_style="green", pulse_style=PURPLE),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
    )


# --- prompts -------------------------------------------------------------------
# Ctrl+C in any prompt raises KeyboardInterrupt, handled once in cli.main.

def select(message: str, choices: Sequence[tuple[Any, str]], default: Any = None, *,
           keys: dict[str, Callable[[Any], None]] | None = None, instruction: str = "↑↓ Enter",
           notes: dict[Any, list[str]] | None = None) -> Any:
    """Arrow-key menu. `keys` maps extra keys (e.g. "tab") to a function of the highlighted choice's value;
    `notes` adds dim rows under a choice that the cursor skips."""
    opts: list[Any] = []
    for value, title in choices:
        opts.append(Choice(value, name=title))
        opts += [Separator(note) for note in (notes or {}).get(value, [])]
    prompt = inquirer.select(
        message, opts, default=default,
        instruction=instruction, max_height="60%", transformer=_short,
        # Tab/Shift+Tab tick-and-move by default (a no-op tick here) and would win over our own keys
        keybindings={"toggle-down": [], "toggle-up": []} if keys else None, **COMMON,
    )
    for key, action in (keys or {}).items():
        prompt.register_kb(key)(lambda _event, act=action: act(prompt.content_control.selection["value"]))
    return prompt.execute()


def checkbox(message: str, choices: Sequence[tuple[Any, str]], *, checked: set = frozenset(),
             unavailable: dict[Any, str] | None = None, required: bool = False,
             answer: Callable[[list[str]], str] | None = None) -> list:
    """Short tick list. `unavailable` choices are shown greyed out with the reason and can't be ticked."""
    unavailable = unavailable or {}
    opts: list[Any] = [Choice(value, name=title, enabled=value in checked)
                       for value, title in choices if value not in unavailable]
    opts += [Separator(f"  ✗ {title}  ({unavailable[value]})") for value, title in choices if value in unavailable]
    return inquirer.checkbox(
        message, opts, instruction="Space to tick · Enter to confirm", max_height="60%",
        enabled_symbol="◉", disabled_symbol="○",
        validate=(lambda picked: bool(picked)) if required else None,
        invalid_message="Tick at least one with Space",
        transformer=answer or (lambda titles: ", ".join(map(_short, titles)) or "none"), **COMMON,
    ).execute()


def search(message: str, choices: Sequence[tuple[Any, str]], *, multiselect: bool = True,
           answer: Callable[[list[str]], str] | None = None) -> list:
    """Fuzzy-search list for long choice sets; scrolls instead of flooding the terminal."""
    return inquirer.fuzzy(
        message, [Choice(value, name=title) for value, title in choices], multiselect=multiselect,
        instruction="type to search · Tab to tick more · Enter to confirm" if multiselect else "type to search · Enter",
        max_height="50%", marker="◉ ", marker_pl="  ", info=True, match_exact=False,
        transformer=answer or (lambda titles: ", ".join(map(_short, titles))), **COMMON,
    ).execute()


def text(message: str, *, validate: Callable[[str], bool], invalid: str, instruction: str = "") -> str:
    return inquirer.text(message, validate=validate, invalid_message=invalid, instruction=instruction,
                         style=PROMPT_STYLE, qmark="◆", amark="◇").execute()


def confirm(message: str, default: bool = True) -> bool:
    return inquirer.confirm(message, default=default, style=PROMPT_STYLE, qmark="◆", amark="◇").execute()


def path(message: str) -> str:
    return inquirer.filepath(
        message, validate=lambda p: bool(p.strip()) and Path(p).expanduser().exists(),
        invalid_message="No such file or folder", style=PROMPT_STYLE, qmark="◆", amark="◇",
        instruction="Tab completes",
    ).execute()
