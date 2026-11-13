"""The build job's tag check: scripts/release_tag.py <tag> stops the release when the tag isn't v plus the library's __version__, or
isn't vX.Y.Z. This check runs the script as the release workflow does:

    TAG the tag v plus __version__ passes. A tag that doesn't match fails with a message that names the tag and the version, and a
        tag that isn't vX.Y.Z fails too. In GitHub Actions the message is an error annotation.

Standard library only.

  pip install .
  python tests/check_release_tag.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import domino_agui

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_tag.py"
failed = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failed.append(name)


def release_tag(tag, script=SCRIPT, env=None):
    """Runs the script with `tag` as the workflow does, outside GitHub Actions unless `env` says so (CI sets GITHUB_ACTIONS); returns its
    exit code and its output."""
    clean = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"}
    done = subprocess.run([sys.executable, "-I", str(script), tag], capture_output=True, text=True, timeout=30,
                          env={**clean, **(env or {})})
    return done.returncode, done.stdout + done.stderr


print(f"domino_agui {domino_agui.__version__}, {SCRIPT}\n")
code, out = release_tag(f"v{domino_agui.__version__}")
check(f"TAG v{domino_agui.__version__}, v plus this package's __version__, passes", code == 0, f"exit {code}: {out.strip()}")

# A copy of the script in a folder like the repo, with a known __version__, so that the cases don't depend on this release's version
repo = Path(tempfile.mkdtemp(prefix="release-tag-"))
(repo / "scripts").mkdir()
shutil.copy(SCRIPT, repo / "scripts" / SCRIPT.name)
(repo / "domino_agui").mkdir()
(repo / "domino_agui" / "__init__.py").write_text('"""A package."""\n__version__ = "1.2.3"\nPROTOCOL_VERSION = "1.0"\n')
copy = repo / "scripts" / SCRIPT.name

code, out = release_tag("v1.2.3", copy)
check("TAG with __version__ 1.2.3, the tag v1.2.3 passes", code == 0, f"exit {code}: {out.strip()}")
for tag in ["v1.2.4", "1.2.3", "V1.2.3", "v1.2.3rc1", "refs/tags/v1.2.3", ""]:
    code, out = release_tag(tag, copy)
    check(f"TAG with __version__ 1.2.3, the tag {tag!r} fails, and the message names the tag and the version",
          code == 1 and f"'{tag}'" in out and "1.2.3" in out and "__version__" in out, f"exit {code}: {out.strip()}")
(repo / "domino_agui" / "__init__.py").write_text('__version__ = "1.2.3rc1"\n')
code, out = release_tag("v1.2.3rc1", copy)
check("TAG with __version__ 1.2.3rc1, the tag v1.2.3rc1 fails, because a release is vX.Y.Z only, and the message says so",
      code == 1 and "'v1.2.3rc1'" in out and "vX.Y.Z" in out, f"exit {code}: {out.strip()}")
(repo / "domino_agui" / "__init__.py").write_text('__version__ = "1.2.3"\n')
code, out = release_tag("v1.2.4", copy, env={"GITHUB_ACTIONS": "true"})
check("TAG in GitHub Actions, the message is an error annotation, so that the run's page shows it",
      code == 1 and out.startswith("::error title=Wrong release tag::the tag 'v1.2.4'"), f"exit {code}: {out.strip()}")

print("\nall checks passed" if not failed else f"\n{len(failed)} failed: {failed}")
sys.exit(1 if failed else 0)
