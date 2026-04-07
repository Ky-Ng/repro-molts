"""HuggingFace Hub upload utilities for MOLT checkpoints and activations."""

from __future__ import annotations

import os
from pathlib import Path


def upload_files(
    local_paths: list[Path],
    repo_id: str,
    path_prefix: str = "",
    commit_message: str = "Upload MOLT artifacts",
    token: str | None = None,
    private: bool = False,
) -> None:
    """Upload files to a HuggingFace Hub repository.

    Args:
        local_paths: list of local file paths to upload
        repo_id: HF repo ID (e.g. "username/repo-name")
        path_prefix: prefix for remote paths in the repo
        commit_message: commit message for the upload
        token: HF token (defaults to $HF_TOKEN)
        private: whether to create a private repo
    """
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No token found. Set HF_TOKEN or pass token argument.")

    api = HfApi(token=token)

    # Resolve namespace if needed
    if "/" not in repo_id:
        user = api.whoami()["name"]
        repo_id = f"{user}/{repo_id}"

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    print(f"Repo: https://huggingface.co/{repo_id}")

    operations = []
    for local_path in local_paths:
        remote_path = f"{path_prefix}/{local_path.name}" if path_prefix else local_path.name
        operations.append(CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(local_path)))

    api.create_commit(repo_id=repo_id, operations=operations, commit_message=commit_message)

    print(f"Uploaded {len(operations)} files to https://huggingface.co/{repo_id}")


def upload_experiment(
    experiment_dir: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> None:
    """Upload all result files from an experiment directory.

    Uploads everything in results/ and figures/ subdirectories.

    Args:
        experiment_dir: path to the experiment folder
        repo_id: HF repo ID
        token: HF token (defaults to $HF_TOKEN)
        private: whether to create a private repo
    """
    experiment_dir = Path(experiment_dir)
    exp_name = experiment_dir.name

    files = []
    for subdir in ["results", "figures"]:
        dir_path = experiment_dir / subdir
        if dir_path.exists():
            files.extend(f for f in dir_path.iterdir() if f.is_file() and f.name != ".gitkeep")

    if not files:
        print(f"No files found in {experiment_dir}")
        return

    upload_files(
        local_paths=files,
        repo_id=repo_id,
        path_prefix=exp_name,
        commit_message=f"Upload results from {exp_name}",
        token=token,
        private=private,
    )
