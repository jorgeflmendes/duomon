from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExternalRepository:
    name: str
    path: Path
    url: str
    branch: str = "master"


REPOSITORIES = (
    ExternalRepository(
        name="Pokemon Showdown server",
        path=ROOT / "showdown-server",
        url="https://github.com/jorgeflmendes/pokemon-showdown-for-duomon.git",
    ),
    ExternalRepository(
        name="Pokemon Showdown client",
        path=ROOT / "showdown-client",
        url="https://github.com/jorgeflmendes/pokemon-showdown-client-for-duomon.git",
    ),
)


def run(args: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def has_local_changes(path: Path) -> bool:
    output = run(["git", "status", "--porcelain"], cwd=path)
    return bool(output)


def path_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def ensure_submodules() -> None:
    run(["git", "submodule", "sync", "--recursive"])
    run(["git", "submodule", "update", "--init", "--recursive"])


def clone_repo(repo: ExternalRepository) -> None:
    if path_has_files(repo.path):
        if is_git_repo(repo.path):
            return
        raise SystemExit(f"{repo.path} exists but is not a git repository.")
    run(["git", "clone", "--branch", repo.branch, repo.url, str(repo.path)], cwd=ROOT)


def ensure_external_repositories() -> None:
    if is_git_repo(ROOT):
        ensure_submodules()
    for repo in REPOSITORIES:
        if not is_git_repo(repo.path):
            clone_repo(repo)
        align_remote(repo)


def align_remote(repo: ExternalRepository) -> None:
    if not is_git_repo(repo.path):
        raise SystemExit(f"{repo.path} is not a git repository.")
    run(["git", "remote", "set-url", "origin", repo.url], cwd=repo.path)


def update_repo(repo: ExternalRepository) -> None:
    if has_local_changes(repo.path):
        raise SystemExit(
            f"{repo.path} has local changes. Commit/stash them before updating the submodule."
        )
    run(["git", "fetch", "origin", repo.branch], cwd=repo.path)
    run(["git", "checkout", f"origin/{repo.branch}"], cwd=repo.path)


def build_repo(repo: ExternalRepository) -> None:
    npm = "npm.cmd" if os.name == "nt" else "npm"
    run([npm, "ci"], cwd=repo.path)
    run([npm, "run", "build"], cwd=repo.path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up DuoMon external repositories.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Move clean external checkouts to origin/master.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run npm ci and npm run build inside each external repository.",
    )
    args = parser.parse_args()

    ensure_external_repositories()
    for repo in REPOSITORIES:
        if args.update:
            update_repo(repo)
        if args.build:
            build_repo(repo)
        print(f"{repo.name}: {repo.path.relative_to(ROOT)} -> {repo.url}")


if __name__ == "__main__":
    main()
