"""The docstrings show an own model in one line, and an author copies that line. Two mistakes are easy to make there. The first is a
temperature: a gpt-5 model can refuse one. The second is a ChatOpenAI( with no use_responses_api=False: it can then use OpenAI's
Responses API. With use_responses_api=False it uses chat completions, which OpenAI-compatible models speak too. This check reads each
docstring (module, class, def) of the installed package's modules:

    DOCSTRING no own-model example in a docstring sets a temperature, and each ChatOpenAI( in one sets use_responses_api=False; a
              failure names the file and the line

Standard library only.

  pip install .
  python tests/check_docstrings.py
"""
import ast
import sys
from pathlib import Path

import domino_agui

PACKAGE = Path(domino_agui.__file__).resolve().parent
failed = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failed.append(name)


def docstring_lines(path):
    """Each line of each docstring (module, class, def) of a Python file: (its line number, the line)."""
    lines = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines += [(first.lineno + i, line) for i, line in enumerate(first.value.value.splitlines())]
    return lines


print(f"domino_agui {domino_agui.__version__} from {PACKAGE}\n")
modules = sorted(PACKAGE.glob("*.py"))
check("DOCSTRING the installed package has its modules", len(modules) > 1, f"no modules in {PACKAGE}")
bad = [f"    domino_agui/{path.name}:{n}: {line.strip()}" for path in modules for n, line in docstring_lines(path)
       if "temperature=" in line or ("ChatOpenAI(" in line and "use_responses_api=False" not in line)]
check("DOCSTRING the own-model examples in domino_agui set no temperature, and each ChatOpenAI( sets use_responses_api=False", not bad,
      "take out the temperature, and add use_responses_api=False:\n" + "\n".join(bad))

print("\nall checks passed" if not failed else f"\n{len(failed)} failed: {failed}")
sys.exit(1 if failed else 0)
