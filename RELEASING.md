# Releasing

A GitHub Release starts a release to PyPI. The workflow `.github/workflows/release.yml` builds the wheel and the sdist, and publishes
them with PyPI's trusted publishing after the maintainer approves. No PyPI token is kept anywhere.

1. In a PR, bump `__version__` in `domino_agui/__init__.py` and add the version's entry to `CHANGELOG.md`. Versions follow semantic
   versioning.
2. Merge the PR to `main` when CI passes.
3. Publish a GitHub Release from `main` with a new tag: `v` plus `__version__`, for example `v0.2.0`.
4. Approve the job "Publish to PyPI": it waits for the required reviewer of the GitHub environment `pypi`.

If the tag isn't `v` plus `__version__`, or isn't `vX.Y.Z`, the build job stops and says so, and nothing goes to PyPI. Delete the
release and its tag. Then publish the release again with the correct tag.

PyPI doesn't take a version two times. To fix a published release, release a new version.

The setup, one time:

- On PyPI: a trusted publisher for the project `domino-agui`, with this repo, the workflow `release.yml` and the environment `pypi`.
- On GitHub: the environment `pypi`, with the maintainer as its required reviewer.
