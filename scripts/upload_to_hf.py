"""Upload all MOLT checkpoints, histories, and results to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py                          # uses HF_TOKEN env var
    python scripts/upload_to_hf.py --token hf_xxx           # explicit token
    python scripts/upload_to_hf.py --dry-run                # list files without uploading
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REPO_ID = "auto-repro-molts"
ROOT = Path(__file__).resolve().parent.parent


def collect_files():
    """Collect all checkpoint, history, and result files to upload."""
    files = []

    # Checkpoints and histories
    for pt_or_json in sorted(ROOT.glob("checkpoints/**/*")):
        if pt_or_json.is_file():
            rel = pt_or_json.relative_to(ROOT)
            files.append((pt_or_json, str(rel)))

    # Results (plots, JSONs)
    for result_file in sorted(ROOT.glob("results/**/*")):
        if result_file.is_file():
            rel = result_file.relative_to(ROOT)
            files.append((result_file, str(rel)))

    return files


def main():
    parser = argparse.ArgumentParser(description="Upload MOLT artifacts to Hugging Face Hub")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (default: $HF_TOKEN)")
    parser.add_argument("--repo-id", default=REPO_ID, help=f"Repository name (default: {REPO_ID})")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    if not args.token and not args.dry_run:
        parser.error("No token found. Set HF_TOKEN or pass --token.")

    files = collect_files()

    if not files:
        print("No files found to upload.")
        return

    if args.dry_run:
        print(f"Would upload {len(files)} files to {{user}}/{args.repo_id}:\n")
        for local, remote in files:
            size_mb = local.stat().st_size / 1024 / 1024
            print(f"  {remote:<55} ({size_mb:.1f} MB)")
        total = sum(f[0].stat().st_size for f in files) / 1024 / 1024
        print(f"\nTotal: {total:.1f} MB")
        return

    api = HfApi(token=args.token)

    # Resolve full repo_id (namespace/repo)
    repo_id = args.repo_id
    if "/" not in repo_id:
        user = api.whoami()["name"]
        repo_id = f"{user}/{repo_id}"

    # Create repo if it doesn't exist
    create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)
    print(f"Repo: https://huggingface.co/{repo_id}")

    # Upload all files in one commit
    operations = []
    from huggingface_hub import CommitOperationAdd

    for local_path, remote_path in files:
        operations.append(CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(local_path)))

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="Upload MOLT checkpoints, training histories, and result plots",
    )

    print(f"\nUploaded {len(files)} files:")
    for _, remote in files:
        print(f"  {remote}")
    print(f"\nDone! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
