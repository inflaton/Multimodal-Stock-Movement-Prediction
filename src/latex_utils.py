"""Minimal LaTeX helpers used by the notebooks.

`pandas.DataFrame.to_latex` in pandas >= 2.0 routes through the Styler and
therefore requires ``jinja2 >= 3.1.2``. The notebooks shouldn't fail just
because the active env has an older jinja2 (or none at all), so we provide a
small booktabs-style emitter here. The output matches what
``df.to_latex(index=False, escape=True, float_format=...)`` would have produced
on pre-Styler pandas, modulo cosmetic whitespace.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


_LATEX_ESCAPES = (
    ("\\", r"\textbackslash{}"),  # must run first
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


def _escape(s: str) -> str:
    for a, b in _LATEX_ESCAPES:
        s = s.replace(a, b)
    return s


def to_booktabs_latex(
    df: pd.DataFrame,
    float_format: str = "%.4f",
    escape: bool = True,
    column_format: str | None = None,
) -> str:
    """Render a DataFrame as a booktabs LaTeX ``tabular`` (no jinja2 dependency).

    Parameters
    ----------
    df: DataFrame to render.
    float_format: printf-style format string applied to numpy/Python floats.
    escape: when True, escape LaTeX-special characters in string cells/headers.
    column_format: override the column spec (e.g. ``"lrrrr"``). Defaults to
        ``"l"`` per column.
    """
    cols = list(df.columns)

    def _cell(v: object) -> str:
        # Honour NaN -> empty cell for readability; otherwise pandas prints "nan".
        if isinstance(v, float):
            if v != v:  # NaN
                return ""
            return float_format % v
        s = str(v)
        return _escape(s) if escape else s

    def _header(c: object) -> str:
        return _escape(str(c)) if escape else str(c)

    spec = column_format if column_format is not None else "l" * len(cols)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + spec + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(_header(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(_cell(v) for v in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


__all__: Iterable[str] = ("to_booktabs_latex",)
