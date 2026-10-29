#!/usr/bin/env python
"""
Typeset ``docs/manual.md`` as ``docs/CMF_manual.pdf``.

A deliberately small Markdown subset is understood -- headings, paragraphs, bullet and numbered
lists, fenced code blocks, pipe tables, block quotes, horizontal rules and the inline markup
``**bold**``, ``*italic*`` and ``code`` -- which is all the manual uses and all that is needed to
keep the source readable on its own.  Display equations between ``$$`` lines are numbered by
chapter in order of appearance; a comment ``<!-- #eq:name -->`` on the line after an equation
labels it, and ``(#eq:name)`` in the text becomes its number, such as ``(2.3)``.  Chapters,
sections and equations are all numbered here, never in the source, so no number can go stale.
The point of writing the manual in Markdown and converting
it here, rather than building flowables by hand, is that the text stays editable by anyone and the
layout lives in one place.

The DejaVu fonts that ship with matplotlib are registered if they can be found, because the manual
is full of Greek letters, angstroms and superscripts that the built-in Type 1 fonts do not carry
and would silently print as black boxes.

Usage:  python docs/make_manual.py [--out docs/CMF_manual.pdf] [--source docs/manual.md]
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, HRFlowable, Image,
                                KeepTogether,
                                ListFlowable, ListItem, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BODY, BOLD, ITALIC, MONO = "Body", "Body-Bold", "Body-Italic", "Mono"
INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5f6b7a")
RULE = colors.HexColor("#c9d2dd")
ACCENT = colors.HexColor("#1f4e79")
BAND = colors.HexColor("#eef2f7")
CODE_BG = colors.HexColor("#f5f6f8")


def register_fonts() -> bool:
    """DejaVu if it can be found, so that Greek letters and angstroms print."""
    folders = []
    try:
        import matplotlib
        folders.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts",
                                    "ttf"))
    except Exception:                                 # noqa: BLE001 - matplotlib is optional
        pass
    folders += ["C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu",
                "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
    wanted = {BODY: "DejaVuSans.ttf", BOLD: "DejaVuSans-Bold.ttf",
              ITALIC: "DejaVuSans-Oblique.ttf", MONO: "DejaVuSansMono.ttf"}
    found = {}
    for name, filename in wanted.items():
        for folder in folders:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                found[name] = path
                break
    if len(found) < len(wanted):
        missing = ", ".join(sorted(set(wanted) - set(found)))
        print(f"[warn] DejaVu not found ({missing}); falling back to Helvetica, which has no "
              f"Greek letters", file=sys.stderr)
        return False
    for name, path in found.items():
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(BODY, normal=BODY, bold=BOLD, italic=ITALIC, boldItalic=BOLD)
    return True


def styles(have_dejavu: bool):
    body, bold, italic, mono = ((BODY, BOLD, ITALIC, MONO) if have_dejavu else
                                ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"))
    ss = getSampleStyleSheet()
    out = {
        "title": ParagraphStyle("title", parent=ss["Title"], fontName=bold, fontSize=30,
                                leading=36, textColor=ACCENT, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=ss["Title"], fontName=body, fontSize=14,
                                   leading=20, textColor=MUTED, spaceAfter=2),
        "cover": ParagraphStyle("cover", parent=ss["Normal"], fontName=body, fontSize=10.5,
                                leading=16, textColor=INK, alignment=TA_CENTER),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName=bold, fontSize=20, leading=25,
                             textColor=ACCENT, spaceBefore=0, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=13.5,
                             leading=18, textColor=INK, spaceBefore=14, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName=bold, fontSize=11,
                             leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=3),
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontName=body, fontSize=9.6,
                               leading=14.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7),
        "bullet": ParagraphStyle("bullet", parent=ss["BodyText"], fontName=body, fontSize=9.6,
                                 leading=14, textColor=INK, spaceAfter=3),
        "code": ParagraphStyle("code", parent=ss["Code"], fontName=mono, fontSize=8.1,
                               leading=11.4, textColor=colors.HexColor("#17202b")),
        "quote": ParagraphStyle("quote", parent=ss["BodyText"], fontName=italic, fontSize=9.4,
                                leading=13.6, textColor=MUTED, leftIndent=10, rightIndent=6,
                                spaceAfter=7),
        "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontName=body, fontSize=8.4,
                               leading=11.4, textColor=INK, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=ss["BodyText"], fontName=bold, fontSize=8.4,
                                   leading=11.4, textColor=colors.white, spaceAfter=0),
        "toc0": ParagraphStyle("toc0", fontName=bold, fontSize=10.2, leading=16, textColor=INK),
        "toc1": ParagraphStyle("toc1", fontName=body, fontSize=9.3, leading=13.4,
                               leftIndent=14, textColor=INK),
        "ref": ParagraphStyle("ref", parent=ss["BodyText"], fontName=body, fontSize=8.9,
                              leading=12.4, textColor=INK, leftIndent=17, firstLineIndent=-17,
                              spaceAfter=5.5, alignment=TA_JUSTIFY),
        "caption": ParagraphStyle("caption", parent=ss["BodyText"], fontName=italic,
                                  fontSize=8.6, leading=12, textColor=MUTED, alignment=TA_CENTER,
                                  spaceAfter=0),
        "footer": ParagraphStyle("footer", fontName=body, fontSize=7.8, leading=10,
                                 textColor=MUTED),
    }
    out["_fonts"] = (body, bold, italic, mono)
    return out


# --------------------------------------------------------------------------- #
# Inline markup
# --------------------------------------------------------------------------- #

#: the numbers of the display equations in order of appearance, and the number of every labelled
#: one, filled by number_equations() before the source is parsed, so that a reference may come
#: before the equation it points at
EQUATIONS = {"numbers": [], "labels": {}}

_EQ_LABEL = re.compile(r"^<!--\s*#eq:([\w-]+)\s*-->$")


def number_equations(lines):
    """Number every display equation as (chapter.n), counting from 1 in each chapter, with an
    appendix keeping its letter; returns the numbers in order and the labels.

    An equation is labelled by an HTML comment on the line after its closing ``$$``, written
    ``<!-- #eq:name -->``, and the text refers to it as ``(#eq:name)``, which becomes ``(2.3)``.
    The chapter count mirrors parse(): a ``# `` heading starts a chapter unless its title starts
    with ``Appendix``, and headings inside code blocks do not count.
    """
    numbers, labels = [], {}
    chapter, number, n, i = 0, "0", 0, 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("# "):
            m = re.match(r"^Appendix ([A-Z])(?![A-Za-z])", s[2:])
            if m:
                number = m.group(1)
            else:
                chapter += 1
                number = str(chapter)
            n = 0
        elif s.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
        elif s.startswith("$$"):
            j = i
            if not (s.endswith("$$") and len(s) > 4):
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("$$"):
                    j += 1
            n += 1
            numbers.append(f"{number}.{n}")
            m = _EQ_LABEL.match(lines[j + 1].strip()) if j + 1 < len(lines) else None
            if m:
                if m.group(1) in labels:
                    print(f"[warn] equation label {m.group(1)} used twice", file=sys.stderr)
                labels[m.group(1)] = numbers[-1]
            i = j
        i += 1
    return numbers, labels


def inline(text: str, mono: str) -> str:
    """Markdown inline markup to reportlab's mini-HTML, escaping what must be escaped."""
    # maths first, so that its backslashes and braces never meet the escaping below
    maths, order = {}, []

    def stash(m):
        key = f"\x00M{len(order)}\x00"
        order.append(key)
        maths[key] = inline_math(m.group(1), mono)
        return key

    def equation(m):
        label = m.group(1)
        if label in EQUATIONS["labels"]:
            return f"({EQUATIONS['labels'][label]})"
        print(f"[warn] no equation is labelled {label}", file=sys.stderr)
        return m.group(0)

    out = re.sub(r"(?<!\\)\$([^$]+)\$", stash, text)
    out = re.sub(r"\(#eq:([\w-]+)\)", equation, out)
    out = out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = re.sub(r"`([^`]+)`",
                 lambda m: f'<font name="{mono}" size="8.6">{m.group(1)}</font>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1f4e79">\1</link>', out)
    # a DOI is a reference the reader will want to follow, so make it one
    out = re.sub(r"doi:(10\.\S+?)(?=[\s,;)]|$)",
                 r'<link href="https://doi.org/\1" color="#1f4e79">doi:\1</link>', out)
    for key in order:
        out = out.replace(key, maths[key])
    return out


class Rule(Flowable):
    """A thin rule the width of the frame, used under chapter titles."""

    def __init__(self, thickness=0.8, colour=RULE, space=4):
        super().__init__()
        self.thickness, self.colour, self.space = thickness, colour, space
        self.width, self.height = 0, thickness + space

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.height

    def draw(self):
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space, self.width, self.space)


# --------------------------------------------------------------------------- #
# Block parsing
# --------------------------------------------------------------------------- #

def split_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


#: rendered equations, keyed by (latex, size); typesetting one costs about 40 ms
_MATH_CACHE = {}


#: LaTeX commands that inline maths may use, as the characters DejaVu actually carries
GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "lambda": "λ",
    "mu": "μ", "nu": "ν", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Sigma": "Σ", "Phi": "Φ", "Delta": "Δ",
    "Omega": "Ω", "Lambda": "Λ",
}
SYMBOL = {
    "sum": "Σ", "int": "∫", "infty": "∞", "to": "→", "approx": "≈", "neq": "≠", "le": "≤",
    "ge": "≥", "times": "×", "cdot": "·", "pm": "±", "ldots": "…", "langle": "⟨",
    "rangle": "⟩", "partial": "∂", "nabla": "∇", "in": "∈", "propto": "∝", "sqrt": "√",
    "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋",
    "sim": "~", "mid": "|", "perp": "⊥", "equiv": "≡", "subset": "⊂",
}
#: accents, as the combining marks that follow the letter they sit on
MATH_ACCENT = {"hat": "̂", "bar": "̄", "tilde": "̃", "dot": "̇",
               "vec": "⃗"}
#: upright in mathematics, however italic the surrounding letters are
UPRIGHT = {"exp", "log", "ln", "max", "min", "arg", "det", "tr", "const"}


def read_group(expr: str, i: int):
    """The argument at `expr[i]` and the index after it: a balanced {...} or one character.

    The braces nest -- a transpose is written ^{\\mathrm{T}} -- so the group cannot be matched
    with a regular expression that stops at the first closing brace.
    """
    if i >= len(expr) or expr[i] != "{":
        return expr[i:i + 1], i + 1
    depth, j = 0, i
    while j < len(expr):
        if expr[j] == "{":
            depth += 1
        elif expr[j] == "}":
            depth -= 1
            if depth == 0:
                return expr[i + 1:j], j + 1
        j += 1
    return expr[i + 1:], len(expr)


def inline_math(expr: str, mono: str) -> str:
    """A short formula as reportlab markup: italic variables, real sub- and superscripts.

    Display equations go through matplotlib, but a symbol sitting inside a sentence has to share
    the paragraph's baseline and line breaking, which an image cannot do.  This covers what such
    symbols need -- Greek letters, indices, powers, upright function names -- and anything more
    elaborate belongs on a line of its own anyway.
    """
    out, i, n = [], 0, len(expr)

    def emit_group(text):
        return inline_math(text, mono)

    while i < n:
        ch = expr[i]
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+)", expr[i:])
            if m:
                name = m.group(1)
                i += len(m.group(0))
                if name == "mathbf":
                    m2 = re.match(r"\s*\{([^}]*)\}", expr[i:])
                    if m2:
                        out.append(f"<b>{m2.group(1)}</b>")
                        i += len(m2.group(0))
                    continue
                if name in MATH_ACCENT:
                    m2 = re.match(r"\s*\{([^}]*)\}|\s*(\w)", expr[i:])
                    if m2:
                        body = m2.group(1) if m2.group(1) is not None else m2.group(2)
                        # a combining mark is drawn where the pen stands, so it lands correctly
                        # only when Unicode has one character for the pair -- which it has for
                        # the lower-case letters and for no capital, and that is why a hatted
                        # capital belongs in a display equation rather than in a sentence
                        composed = unicodedata.normalize("NFC", body + MATH_ACCENT[name])
                        if len(composed) == len(body):
                            out.append(f"<i>{composed}</i>" if composed.isalpha()
                                       else emit_group(composed))
                        else:
                            out.append(emit_group(body) + MATH_ACCENT[name])
                        i += len(m2.group(0))
                    continue
                if name in ("sqrt", "frac"):
                    # the only commands inline maths may use that take arguments; without this
                    # the braces of their groups reach the page as braces
                    pat = (r"\s*\{([^}]*)\}\s*\{([^}]*)\}" if name == "frac"
                           else r"\s*\{([^}]*)\}|\s*(\w)")
                    m2 = re.match(pat, expr[i:])
                    if not m2:
                        out.append("\u221a" if name == "sqrt" else "/")
                        continue
                    i += len(m2.group(0))
                    if name == "frac":
                        out.append(f"{emit_group(m2.group(1))}/{emit_group(m2.group(2))}")
                    else:
                        body = m2.group(1) if m2.group(1) is not None else m2.group(2)
                        inner = emit_group(body)
                        out.append("\u221a" + (inner if len(body) < 2 else f"({inner})"))
                    continue
                if name in ("text", "mathrm", "operatorname", "mathcal", "mathbb"):
                    m2 = re.match(r"\s*\{([^}]*)\}", expr[i:])
                    if m2:
                        out.append(m2.group(1))
                        i += len(m2.group(0))
                    continue
                out.append(GREEK.get(name, SYMBOL.get(name, name)))
                continue
            if expr[i:i + 2] in ("\\,", "\\;", "\\!", "\\ "):
                out.append("" if expr[i + 1] == "!" else " ")
                i += 2
                continue
            i += 1
            continue
        if ch in "_^":
            tag = "sub" if ch == "_" else "super"
            i += 1
            body, i = read_group(expr, i)
            out.append(f"<{tag}>{emit_group(body)}</{tag}>")
            continue
        m = re.match(r"[A-Za-z]+", expr[i:])
        if m:
            word = m.group(0)
            i += len(word)
            out.append(word if word in UPRIGHT else
                       "".join(f"<i>{c}</i>" if c.isalpha() else c for c in word))
            continue
        # a proper minus sign, not the hyphen the keyboard offers
        out.append({"-": "−", "&": "&amp;", "<": "&lt;", ">": "&gt;"}.get(ch, ch))
        i += 1
    return "".join(out)


def math_drawing(latex: str, fontsize: float = 13.0, colour: str = "#1a1a1a"):
    """One display equation as a vector drawing, or None if it cannot be typeset.

    matplotlib's mathtext understands enough LaTeX for everything in this manual and needs no TeX
    installation; the formula is written to SVG with the glyphs converted to paths, so the result
    carries no font dependency, and svglib turns that into reportlab's own vector primitives.
    Nothing is rasterised anywhere along the way.
    """
    key = (latex, fontsize, colour)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["mathtext.default"] = "it"
        matplotlib.rcParams["svg.fonttype"] = "path"
        from matplotlib.figure import Figure
        from svglib.svglib import svg2rlg
    except Exception as exc:                      # noqa: BLE001 - fall back to the plain text
        print(f"[warn] equations stay unformatted ({exc})", file=sys.stderr)
        _MATH_CACHE[key] = None
        return None
    try:
        fig = Figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=colour)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.012, transparent=True)
        buf.seek(0)
        drawing = svg2rlg(buf)
    except Exception as exc:                      # noqa: BLE001 - one bad formula, not the book
        print(f"[warn] cannot typeset {latex[:40]!r}: {exc}", file=sys.stderr)
        drawing = None
    _MATH_CACHE[key] = drawing
    return drawing


def math_block(lines, st, frame_width, number=None):
    """A display equation, centred, with its number at the right margin and the source kept as
    a fallback."""
    from reportlab.graphics.shapes import Drawing, Group, String
    latex = " ".join(l.strip() for l in lines if l.strip())
    drawing = math_drawing(latex)
    if drawing is None:
        return code_block(lines, st, frame_width)
    # the number takes a margin on the right, and the same margin is kept on the left so that
    # the equation stays centred on the page; a long equation is scaled into the room, never clipped
    margin = 44 if number else 0
    scale = min(1.0, (frame_width - 2 * margin) / drawing.width)
    width, height = drawing.width * scale, drawing.height * scale
    group = Group(*drawing.contents)
    group.transform = (scale, 0, 0, scale, (frame_width - width) / 2, 0)
    block = Drawing(frame_width, height)
    block.add(group)
    if number:
        size = 9.5
        block.add(String(frame_width, (height - 0.7 * size) / 2, f"({number})",
                         fontName=st["_fonts"][0], fontSize=size, textAnchor="end",
                         fillColor=INK))
    block.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 5), block, Spacer(1, 9)])


def figure(path, caption, st, frame_width):
    """A screenshot or diagram, scaled to the text width, with its caption under it."""
    full = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(full):
        full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        print(f"[warn] no such figure: {path}", file=sys.stderr)
        return Spacer(1, 1)
    if full.lower().endswith(".svg"):
        # a vector figure, as the equations: scaled to the text width, never rasterised
        from svglib.svglib import svg2rlg
        img = svg2rlg(full)
        scale = min(1.0, frame_width / img.width)
        img.scale(scale, scale)
        img.width, img.height = img.width * scale, img.height * scale
    else:
        reader = ImageReader(full)
        iw, ih = reader.getSize()
        # a 150 dpi raster is drawn at its natural size, or narrower to fit
        width = min(frame_width, float(iw) * 72.0 / 150.0)
        img = Image(full, width=width, height=width * ih / iw)
    img.hAlign = "CENTER"
    parts = [Spacer(1, 4), img]
    if caption:
        parts += [Spacer(1, 3), Paragraph(inline(caption, st["_fonts"][3]), st["caption"])]
    parts.append(Spacer(1, 9))
    return KeepTogether(parts)


def make_table(rows, st, frame_width):
    mono = st["_fonts"][3]
    head, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c, mono), st["cellhead"]) for c in head]]
    for row in body:
        data.append([Paragraph(inline(c, mono), st["cell"]) for c in row])
    n = max(len(r) for r in rows)
    # the first column carries the labels and gets more room than the numbers after it
    first = min(0.34, max(0.16, 0.42 - 0.03 * n))
    widths = [frame_width * first] + [frame_width * (1 - first) / (n - 1)] * (n - 1) if n > 1 \
        else [frame_width]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def code_block(lines, st, frame_width):
    mono = st["_fonts"][3]
    text = "\n".join(lines)
    pre = Preformatted(text, st["code"], maxLineLength=104)
    t = Table([[pre]], colWidths=[frame_width], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.25, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    assert mono  # the style already carries it; kept for clarity
    return t


def _starts_block(line: str) -> bool:
    """True when a line opens a block of its own rather than continuing a paragraph.

    A bibliography entry starts with "[1] ", so the test has to be that and not merely "[":
    an ordinary paragraph may wrap onto a line that begins with a Markdown link, and treating
    that as a block start leaves the line to no branch at all.
    """
    return (line.startswith(("#", "|", "```", ">", "- ", "* ", "---", "<!--", "$$", "!["))
            or bool(re.match(r"^\[\d+\]\s", line))
            or bool(re.match(r"^\d+\.\s+", line)))


def parse(source: str, st, frame_width):
    """Markdown subset to a list of flowables."""
    mono = st["_fonts"][3]
    story, lines, i = [], source.replace("\r\n", "\n").split("\n"), 0
    # chapters and sections are numbered here, never in the source, so the numbers cannot go
    # stale; an appendix keeps the letter its title already carries, and its sections take it
    chapter, section, number, heads, equations = 0, 0, "0", 0, 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if re.match(r"^\[\d+\]\s", stripped):
            # a bibliography entry: it runs until the next one or a blank line, and is set with a
            # hanging indent so that the numbers line up down the margin
            block = [stripped]
            i += 1
            while (i < len(lines) and lines[i].strip()
                   and not re.match(r"^\[\d+\]\s", lines[i].strip())):
                block.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(block), mono), st["ref"]))
            continue
        # the caption may carry a citation in brackets, so it runs to the last "](" of the line
        m_img = re.match(r"^!\[(.*)\]\(([^()]+)\)\s*$", stripped)
        if m_img:
            story.append(figure(m_img.group(2), m_img.group(1), st, frame_width))
            i += 1
            continue
        if stripped.startswith("$$"):
            numbers = EQUATIONS["numbers"]
            eq_number = numbers[equations] if equations < len(numbers) else None
            equations += 1
            if stripped.endswith("$$") and len(stripped) > 4:      # a one-line equation
                story.append(math_block([stripped[2:-2]], st, frame_width, eq_number))
                i += 1
            else:
                j, block = i + 1, []
                while j < len(lines) and not lines[j].strip().startswith("$$"):
                    block.append(lines[j])
                    j += 1
                story.append(math_block(block, st, frame_width, eq_number))
                i = j + 1
            if i < len(lines) and _EQ_LABEL.match(lines[i].strip()):   # the label line
                i += 1
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            i += 1                                               # any other comment
            continue
        if stripped.startswith("```"):
            j = i + 1
            block = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                block.append(lines[j])
                j += 1
            story.append(Spacer(1, 2))
            story.append(code_block(block, st, frame_width))
            story.append(Spacer(1, 8))
            i = j + 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\|[\s:|-]+\|?$", lines[i + 1].strip()):
            rows = [split_row(stripped)]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            story.append(Spacer(1, 2))
            story.append(make_table(rows, st, frame_width))
            story.append(Spacer(1, 9))
            i = j
            continue
        if stripped.startswith("# "):
            heads += 1
            if heads > 1:
                story.append(PageBreak())
            title = stripped[2:]
            m = re.match(r"^Appendix ([A-Z])(?![A-Za-z])", title)
            if m:
                number = m.group(1)
            else:
                chapter += 1
                number = str(chapter)
                title = f"{number}  {title}"
            section = 0
            text = inline(title, mono)
            para = Paragraph(f'<a name="ch{heads}"/>{text}', st["h1"])
            para._toc = (0, title, f"ch{heads}")
            story += [para, Rule(1.1, ACCENT, 5), Spacer(1, 6)]
            i += 1
            continue
        if stripped.startswith("## ") or stripped.startswith("### "):
            level = 1 if stripped.startswith("## ") else 2
            text = stripped[3:] if level == 1 else stripped[4:]
            if level == 1:
                section += 1
                text = f"{number}.{section}  {text}"
            para = Paragraph(inline(text, mono), st["h2" if level == 1 else "h3"])
            if level == 1:
                para._toc = (1, text, None)
            story.append(para)
            i += 1
            continue
        if stripped == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=4,
                                    spaceAfter=8))
            i += 1
            continue
        if stripped.startswith("> "):
            block = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                block.append(lines[i].strip()[2:])
                i += 1
            story.append(Paragraph(inline(" ".join(block), mono), st["quote"]))
            continue
        if re.match(r"^([-*]|\d+\.)\s+", stripped):
            numbered = bool(re.match(r"^\d+\.\s+", stripped))
            items, pattern = [], r"^\d+\.\s+" if numbered else r"^[-*]\s+"
            while i < len(lines) and re.match(pattern, lines[i].strip()):
                text = re.sub(pattern, "", lines[i].strip())
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and \
                        not re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()):
                    text += " " + lines[i].strip()
                    i += 1
                item = Paragraph(inline(text, mono), st["bullet"])
                items.append(ListItem(item, leftIndent=16, value=len(items) + 1) if numbered
                             else ListItem(item, leftIndent=16))
            kw = {"bulletType": "1", "start": 1} if numbered else {"bulletType": "bullet",
                                                                   "start": "•"}
            story.append(ListFlowable(items, bulletFontName=st["_fonts"][0], bulletFontSize=8.6,
                                      leftIndent=16, spaceBefore=1, spaceAfter=7, **kw))
            continue
        block = []
        while i < len(lines) and lines[i].strip() and not _starts_block(lines[i].strip()):
            block.append(lines[i].strip())
            i += 1
        if not block:
            # Nothing consumed the line and the paragraph scanner would not take it either, so
            # the loop would spin here for ever -- which it did, on a wrapped line that began
            # with a Markdown link, because "[" marks a bibliography entry. Take the line as
            # ordinary text and move on: a surprise in the source should cost a line of odd
            # typesetting, not a hung build.
            block = [lines[i].strip()]
            i += 1
        story.append(Paragraph(inline(" ".join(block), mono), st["body"]))
    return story


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #

class Manual(BaseDocTemplate):
    """Two page templates: a bare cover and the body with a running footer."""

    def __init__(self, path, st, title, **kw):
        super().__init__(path, pagesize=A4, title=title, author="Igor I. Baskin",
                         subject="CMF 3D-QSAR desktop application", **kw)
        self.st = st
        self.chapter = ""
        self.entries = []
        margin = 20 * mm
        frame = Frame(margin, margin, A4[0] - 2 * margin, A4[1] - 2 * margin - 6 * mm,
                      id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        cover = Frame(margin, margin, A4[0] - 2 * margin, A4[1] - 2 * margin, id="cover",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[cover]),
            PageTemplate(id="body", frames=[frame], onPageEnd=self.decorate),
        ])

    def beforeDocument(self):
        # the document is built twice -- once to find out which page everything lands on, once
        # with the contents filled in -- and neither pass may inherit the other's state
        self.chapter = ""
        self.entries = []

    def decorate(self, canv, doc):
        # drawn at the end of the page, not its beginning: by then the running chapter is the one
        # whose text is actually on this page, not the one before it
        canv.saveState()
        canv.setFont(self.st["_fonts"][0], 7.8)
        canv.setFillColor(MUTED)
        y = 13 * mm
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(20 * mm, y + 4 * mm, A4[0] - 20 * mm, y + 4 * mm)
        canv.drawString(20 * mm, y, "CMF 3D-QSAR — user manual")
        right = self.chapter[:70]
        canv.drawRightString(A4[0] - 20 * mm, y, f"{right}   ·   {doc.page}" if right
                             else str(doc.page))
        canv.restoreState()

    def afterFlowable(self, flowable):
        toc = getattr(flowable, "_toc", None)
        if toc is None:
            return
        level, text, anchor = toc
        if level == 0:
            self.chapter = text
            self.canv.bookmarkPage(anchor)
            self.canv.addOutlineEntry(text, anchor, level=0, closed=True)
        self.entries.append((level, text, self.page, anchor))


def cover(st, title, subtitle, lines):
    story = [Spacer(1, 46 * mm), Paragraph(title, st["title"]),
             Spacer(1, 2), Paragraph(subtitle, st["subtitle"]), Spacer(1, 6),
             HRFlowable(width="38%", thickness=1.2, color=ACCENT, hAlign="CENTER",
                        spaceBefore=6, spaceAfter=14)]
    for line in lines:
        story.append(Paragraph(line, st["cover"]))
    story.append(Spacer(1, 1))
    return story


def contents_flowables(entries, st, frame_width):
    """The contents, as a plain two-column table: label on the left, page on the right.

    reportlab's own TableOfContents refers to itself -- its length changes the page numbers it
    prints, and with a contents list that lands near a page boundary the two never settle, which
    is how the first draft of this manual came to promise chapter one on a page it was not on.
    Here the document is simply built twice: the first build reports where everything landed, the
    second prints those numbers into a contents of fixed length, so nothing can shift underneath
    it.
    """
    rows = []
    for level, text, page, _anchor in entries:
        style = st["toc0"] if level == 0 else st["toc1"]
        rows.append([Paragraph(inline(text, st["_fonts"][3]), style),
                     Paragraph(f"{page}", ParagraphStyle(
                         "page", parent=style, alignment=2))])
    table = Table(rows, colWidths=[frame_width - 16 * mm, 16 * mm], hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
             ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
             ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]
    for i, (level, *_rest) in enumerate(entries):
        if level == 0:
            style.append(("LINEBELOW", (0, i), (-1, i), 0.25, RULE))
            style.append(("TOPPADDING", (0, i), (-1, i), 6))
    table.setStyle(TableStyle(style))
    return [Paragraph("Contents", st["h1"]), Rule(1.1, ACCENT, 5), Spacer(1, 8), table]


def build(source_path, out_path):
    have = register_fonts()
    st = styles(have)
    with open(source_path, encoding="utf-8") as fh:
        text = fh.read()
    # the first line of the source is the document title, the second its subtitle
    head, rest = text.split("\n", 1) if "\n" in text else (text, "")
    sub, rest = rest.split("\n", 1) if "\n" in rest else (rest, "")
    title, subtitle = head.lstrip("# ").strip(), sub.strip()
    EQUATIONS["numbers"], EQUATIONS["labels"] = number_equations(
        rest.replace("\r\n", "\n").split("\n"))
    lines = ["Desktop application, library and command-line tool",
             "for the Continuous Molecular Fields approach to 3D-QSAR", "", "User manual", "",
             '<font name="%s" size="9">github.com/igbaskin/cmf-qsar</font>' % st["_fonts"][3]]

    width = Manual(os.devnull, st, title).width

    def assemble(front):
        story = cover(st, title, subtitle, lines)
        story += [NextPageTemplate("body"), PageBreak()]
        story += front + [PageBreak()]
        story += parse(rest, st, width)
        return story

    def pages_of(front):
        """How many pages a contents of this shape occupies."""
        probe = Manual(os.devnull, st, title)
        probe.build(cover(st, title, subtitle, lines) +
                    [NextPageTemplate("body"), PageBreak()] + front)
        return probe.page - 1                              # minus the cover

    slot = 1
    for _ in range(6):
        # build with the contents slot empty but of the right length, so that the body lands on
        # the pages it finally will, and read off where every heading went
        measure = Manual(os.devnull, st, title)
        measure.build(assemble([Spacer(1, 1)] + [PageBreak()] * (slot - 1)))
        entries = list(measure.entries)
        front = contents_flowables(entries, st, width)
        need = pages_of(front)
        if need <= slot:
            doc = Manual(out_path, st, title)
            doc.build(assemble(front + [PageBreak()] * (slot - need)))
            return out_path
        slot = need
    raise RuntimeError("the contents refuses to settle")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.join(HERE, "manual.md"))
    ap.add_argument("--out", default=os.path.join(HERE, "CMF_manual.pdf"))
    args = ap.parse_args()
    path = build(args.source, args.out)
    size = os.path.getsize(path)
    print(f"written {path}  ({size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
