#!/usr/bin/env python
"""
Write ``docs/api.md``, the function-by-function reference of the ``cmfqsar`` package, from the
docstrings in the source.

The reference is generated, not written: a hand-written list of 900 functions, classes and
methods would be out of date after the next feature and nobody would notice.  The source files
are read with the ``ast`` module and never imported, so the generator needs no RDKit, no Qt and
no compiled extension, and nothing runs as a side effect.

For every module, in alphabetical order, the file carries the module docstring, then every
public class with its constructor signature, its annotated attributes and its public methods,
then every public function with its signature.  A name is public when it does not start with an
underscore.  The Sphinx cross-reference markup the docstrings use (``:func:`name```, ``:class:``,
double backticks) is rewritten as Markdown code so that the file reads on GitHub and in any
editor.

Usage:  python docs/make_api.py            # rewrite docs/api.md
        python docs/make_api.py --check    # exit 1 if docs/api.md is not what the source gives
"""
from __future__ import annotations

import argparse
import ast
import inspect
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "cmfqsar")
OUT = os.path.join(ROOT, "docs", "api.md")

_ROLE = re.compile(r":(?:func|meth|class|mod|attr|data|obj|exc|const):`~?([^`]+)`")
_DOUBLE = re.compile(r"``([^`]+)``")


def docstring_to_markdown(doc: str | None) -> str:
    """The docstring as Markdown: Sphinx roles and double backticks become code spans."""
    if not doc:
        return ""
    text = inspect.cleandoc(doc)
    text = _ROLE.sub(lambda m: f"`{m.group(1)}`", text)
    text = _DOUBLE.sub(lambda m: f"`{m.group(1)}`", text)
    # a literal block introduced by "::" keeps its indentation, which Markdown reads as code;
    # reStructuredText turns "text::" into "text:" and drops " ::" and a bare "::" entirely
    text = re.sub(r"^::$", "", text, flags=re.M)
    text = re.sub(r"\s+::$", "", text, flags=re.M)
    text = re.sub(r"(\S)::$", r"\1:", text, flags=re.M)
    return text.strip()


def first_sentence(doc: str | None) -> str:
    """The first sentence of a docstring, on one line, for the module table."""
    text = docstring_to_markdown(doc)
    if not text:
        return ""
    para = text.split("\n\n")[0]
    para = " ".join(para.split())
    m = re.match(r"(.+?[.!?])(?:\s|$)", para)
    return (m.group(1) if m else para).replace("|", "\\|")


def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_first: bool = False) -> str:
    """``(a, b=1, *, c: int = 2) -> ret`` from the function node, without ``self``."""
    args = fn.args
    if drop_first and args.args:
        positional = list(args.posonlyargs) + list(args.args)
        defaults = list(args.defaults)
        # defaults belong to the last positional arguments; the first argument (self or cls)
        # carries one only when every positional argument does
        if len(defaults) == len(positional):
            defaults = defaults[1:]
        args = ast.arguments(posonlyargs=list(args.posonlyargs), args=list(args.args[1:]),
                             vararg=args.vararg, kwonlyargs=list(args.kwonlyargs),
                             kw_defaults=list(args.kw_defaults), kwarg=args.kwarg,
                             defaults=defaults)
    text = ast.unparse(args)
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns is not None else ""
    return f"({text}){ret}"


def decorators(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for d in fn.decorator_list:
        node = d.func if isinstance(d, ast.Call) else d
        names.add(ast.unparse(node).split(".")[-1])
    return names


def is_public(name: str) -> bool:
    return not name.startswith("_")


def module_name(path: str) -> str:
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    rel = rel[:-3]  # .py
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    return rel.replace("/", ".")


def render_function(fn: ast.FunctionDef | ast.AsyncFunctionDef, owner: str | None,
                    level: str) -> str:
    decs = decorators(fn)
    drop = owner is not None and "staticmethod" not in decs
    if owner and "property" in decs:
        head = f"{owner}.{fn.name}"
        ret = f" -> {ast.unparse(fn.returns)}" if fn.returns is not None else ""
        title = f"{level} `{head}`{ret}  (property)"
    else:
        head = f"{owner}.{fn.name}" if owner else fn.name
        tag = ""
        if "classmethod" in decs:
            tag = "  (class method)"
        elif "staticmethod" in decs:
            tag = "  (static method)"
        title = f"{level} `{head}{signature(fn, drop_first=drop)}`{tag}"
    body = docstring_to_markdown(ast.get_docstring(fn))
    return title + ("\n\n" + body if body else "") + "\n"


def render_class(cls: ast.ClassDef) -> tuple[str, int]:
    decs = {ast.unparse(d.func if isinstance(d, ast.Call) else d).split(".")[-1]
            for d in cls.decorator_list}
    init = next((n for n in cls.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"),
                None)
    bases = ", ".join(ast.unparse(b) for b in cls.bases)
    if init is not None:
        head = f"class {cls.name}{signature(init, drop_first=True)}"
    else:
        head = f"class {cls.name}"
    if bases:
        head += f"  — bases: {bases}"
    if "dataclass" in decs:
        head += "  (dataclass)"
    parts = [f"### `{head}`"]
    body = docstring_to_markdown(ast.get_docstring(cls))
    if body:
        parts.append(body)
    attrs = []
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and is_public(n.target.id):
            line = f"`{n.target.id}: {ast.unparse(n.annotation)}`"
            if n.value is not None:
                line += f" = `{ast.unparse(n.value)}`"
            attrs.append("- " + line)
    if attrs:
        parts.append("Attributes:\n\n" + "\n".join(attrs))
    n_methods = 0
    for n in cls.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(n.name):
            parts.append(render_function(n, cls.name, "####").rstrip("\n"))
            n_methods += 1
    return "\n\n".join(parts) + "\n", n_methods


def render_module(path: str) -> tuple[str, str, dict]:
    """(markdown, first sentence, counts) of one source file."""
    name = module_name(path)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    doc = ast.get_docstring(tree)
    counts = {"functions": 0, "classes": 0, "methods": 0}
    parts = [f"## {name}", f"Source: `{os.path.relpath(path, ROOT).replace(os.sep, '/')}`"]
    body = docstring_to_markdown(doc)
    if body:
        parts.append(body)
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and is_public(n.name):
            text, m = render_class(n)
            parts.append(text.rstrip("\n"))
            counts["classes"] += 1
            counts["methods"] += m
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(n.name):
            parts.append(render_function(n, None, "###").rstrip("\n"))
            counts["functions"] += 1
    return "\n\n".join(parts) + "\n", first_sentence(doc), counts


def source_files() -> list[str]:
    files = []
    for folder in (PACKAGE, os.path.join(PACKAGE, "gui")):
        for fn in sorted(os.listdir(folder)):
            if fn.endswith(".py"):
                files.append(os.path.join(folder, fn))
    return files


def generate() -> str:
    sections, rows = [], []
    total = {"modules": 0, "functions": 0, "classes": 0, "methods": 0}
    package_doc = ""
    for path in source_files():
        text, first, counts = render_module(path)
        name = module_name(path)
        if name == "cmfqsar":
            with open(path, encoding="utf-8") as fh:
                package_doc = docstring_to_markdown(ast.get_docstring(ast.parse(fh.read())))
        sections.append(text)
        rows.append(f"| [{name}](#{name.replace('.', '')}) | {first} |")
        total["modules"] += 1
        for k in counts:
            total[k] += counts[k]
    head = [
        "# cmfqsar API reference",
        "",
        "This file is generated from the docstrings of the source by `python docs/make_api.py`; "
        "do not edit it by hand. `python docs/make_api.py --check` fails when it is out of date. "
        f"It covers {total['modules']} modules, {total['classes']} classes with "
        f"{total['methods']} public methods, and {total['functions']} public functions. "
        "A name is public when it does not start with an underscore. The user manual "
        "(`docs/manual.md`, chapter *Using it as a library*) explains what the modules are for "
        "and shows worked scripts; this file says what every function takes and returns.",
        "",
        package_doc,
        "",
        "## Modules",
        "",
        "| Module | What it is for |",
        "|---|---|",
        *rows,
        "",
    ]
    return "\n".join(head) + "\n" + "\n".join(sections)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="compare with the existing file instead of writing it")
    args = ap.parse_args(argv)
    text = generate()
    if args.check:
        try:
            with open(args.out, encoding="utf-8") as fh:
                current = fh.read().replace("\r\n", "\n")
        except FileNotFoundError:
            current = ""
        if current == text:
            print(f"{os.path.relpath(args.out, ROOT)} is up to date")
            return 0
        print(f"{os.path.relpath(args.out, ROOT)} is out of date: run python docs/make_api.py")
        return 1
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"written {os.path.relpath(args.out, ROOT)}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
