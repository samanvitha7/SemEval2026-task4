import argparse
from pathlib import Path

from huggingface_hub import HfApi


def find_sentence_transformer_dirs(root: Path):
    """Return checkpoint dirs that look like SentenceTransformer exports."""
    matches = []
    for model_file in root.rglob("model.safetensors"):
        parent = model_file.parent
        if (parent / "modules.json").exists() and (parent / "config.json").exists():
            matches.append(parent)

    # Sort and deduplicate while preserving stable order.
    unique = []
    seen = set()
    for p in sorted(matches):
        key = str(p.resolve()).lower()
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def default_repo_name(local_dir: Path):
    rel = str(local_dir).replace("\\", "-").replace("/", "-")
    rel = rel.strip("-").lower()
    return f"semeval2026-{rel}"


def push_folder(api: HfApi, username: str, local_dir: Path, repo_name: str, private: bool):
    repo_id = f"{username}/{repo_name}"
    print(f"\n[INFO] Pushing: {local_dir} -> {repo_id}")

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message=f"Upload model from {local_dir}",
    )
    print(f"[OK] https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push all SentenceTransformer checkpoints to Hugging Face Hub")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Hugging Face username (auto-detected if omitted)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repositories as private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be pushed",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    api = HfApi()

    username = args.username
    if not username:
        username = api.whoami()["name"]

    model_dirs = find_sentence_transformer_dirs(root)
    if not model_dirs:
        print("[WARN] No SentenceTransformer model directories found.")
        return

    print(f"[INFO] Found {len(model_dirs)} model directories")
    for d in model_dirs:
        repo_name = default_repo_name(d.relative_to(root))
        print(f"  - {d} -> {username}/{repo_name}")

    if args.dry_run:
        print("[INFO] Dry run complete. No uploads performed.")
        return

    for d in model_dirs:
        repo_name = default_repo_name(d.relative_to(root))
        push_folder(api, username, d, repo_name, args.private)


if __name__ == "__main__":
    main()
