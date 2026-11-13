"""The repo's text is public: authors outside Domino read the library's docstrings, its checks and its docs. So the text has no em dash,
and no reference to a design record that they can't read. This check reads each text file of the repo: the files that git lists
(tracked, and new files that .gitignore doesn't exclude), or each file in the folder when there is no git:

    EM-DASH no file has an em dash; a failure names the file and the line
    REFS    no file has an internal reference: an ADR number, "bootstrapper" (say "the agent service"), GLOSSARY, or a spec or
            story number; a failure names the file, the line and the reference. Say the rule in plain words instead

An issue or PR number (#12) isn't a REFS failure: this repo's own issues and PRs are public. Names of people, clusters and internal
hosts aren't in REFS either: a list of them here would publish them.

Standard library only. It reads the source folder, not the installed package.

  python tests/check_text.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()   # its own text names the references, so REFS doesn't read it
SKIP = {".git", "build", "dist", "__pycache__", ".venv", "venv"}   # with no git: .gitignore's folders, and git's own
EM_DASH = chr(0x2014)
REFS = [("an ADR number", re.compile(r"\bADRs?[- #]?\d", re.IGNORECASE)),
        ("the bootstrapper", re.compile(r"\bbootstrapper", re.IGNORECASE)),
        ("GLOSSARY", re.compile(r"\bGLOSSARY\b")),
        ("a spec number", re.compile(r"\bspec\b\S*\s*\(?#\d", re.IGNORECASE)),
        ("a story number", re.compile(r"\bstor(?:y|ies)[- ]#?\d", re.IGNORECASE))]
failed = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failed.append(name)


def listed():
    """The repo's files, from its root: the ones git lists, else each file in the folder that isn't in SKIP or *.egg-info."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True, check=True).stdout
        return sorted(set(out.split("\0")) - {""})
    except (OSError, subprocess.CalledProcessError):   # no git, or not a git checkout (an unpacked sdist)
        return sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
                      and not any(part in SKIP or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts))


def text_files():
    """Each text file of the repo, as (its path from the repo's root, its lines). A file that isn't UTF-8 text, or that git lists but
    the working tree doesn't have (deleted, not yet committed), is left out."""
    for name in listed():
        try:
            yield name, (ROOT / name).read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue


files = list(text_files())
print(f"{len(files)} text files in {ROOT}\n")
check("the repo's text files are read: the README, the library and the checks",
      {"README.md", "pyproject.toml", "domino_agui/run.py", "tests/check_domino_agui.py"} <= {name for name, _ in files},
      [name for name, _ in files])

dashes = [f"    {name}:{n}: {line.strip()}" for name, lines in files for n, line in enumerate(lines, 1) if EM_DASH in line]
check("EM-DASH no text file has an em dash", not dashes, "write a colon, a comma or two sentences:\n" + "\n".join(dashes))

refs = [f"    {name}:{n}: {what} ({m.group(0)!r}): {line.strip()}" for name, lines in files if name != SELF
        for n, line in enumerate(lines, 1) for what, rx in REFS for m in [rx.search(line)] if m]
check("REFS no text file names an ADR, the bootstrapper, GLOSSARY, a spec or a story", not refs,
      "say the rule in plain words, or take the reference out:\n" + "\n".join(refs))

print("\nall checks passed" if not failed else f"\n{len(failed)} failed: {failed}")
sys.exit(1 if failed else 0)
