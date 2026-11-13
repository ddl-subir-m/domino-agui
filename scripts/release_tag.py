"""The build job's tag check: the tag of a release must be v plus the library's __version__, so that a release can't have the wrong
version. A release is a final version, so the tag must also be vX.Y.Z. The script reads __version__ from domino_agui/__init__.py in
this checkout, without an import, as the build does.

  python scripts/release_tag.py v0.2.0
"""
import ast
import os
import re
import sys
from pathlib import Path

INIT_PY = Path(__file__).resolve().parents[1] / "domino_agui" / "__init__.py"


def read_version():
    for node in ast.parse(INIT_PY.read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["__version__"]:
            return node.value.value


def stop(message):
    """Stops the release with `message`. In GitHub Actions, the message is an error annotation, so that the run's page shows it (a % is
    escaped as %25)."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        message = "::error title=Wrong release tag::" + message.replace("%", "%25")
    print(message, file=sys.stderr)
    sys.exit(1)


tag, version = sys.argv[1], read_version()
expected_tag = f"v{version}"
if tag != expected_tag:
    stop(f"the tag '{tag}' isn't v plus __version__ ('{version}' in domino_agui/__init__.py), so this release stops here. Delete the "
         f"release and its tag. Then publish a release with the tag '{expected_tag}', or bump __version__ in a PR first.")
if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
    stop(f"the tag '{tag}' isn't vX.Y.Z, so this release stops here. A release is a final version only. Delete the release and its "
         "tag, and set __version__ to X.Y.Z in a PR first.")
print(f"the tag '{tag}' is v plus __version__")
