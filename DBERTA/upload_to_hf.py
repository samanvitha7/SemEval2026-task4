
import os
import shutil
from huggingface_hub import HfApi, Repository, login

# User: fill in your Hugging Face username and token
HF_USERNAME = "samanvitha7"  # Your HF username
HF_TOKEN = os.environ.get("HF_TOKEN")  # Or paste your token here
REPO_NAME = "semeval-hcp-deberta"  # Name for your model repo on Hugging Face

# 1. Prepare export directory
EXPORT_DIR = "hf_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# 2. Copy model checkpoints (all folds)
ckpt_dir = "checkpoints"
for fname in os.listdir(ckpt_dir):
    if fname.startswith("best_fold_") and fname.endswith(".pt"):
        shutil.copy(os.path.join(ckpt_dir, fname), os.path.join(EXPORT_DIR, fname))

# 3. Copy config and model code
shutil.copy("configs/config.yaml", os.path.join(EXPORT_DIR, "config.yaml"))
shutil.copy("models/deberta_ranker.py", os.path.join(EXPORT_DIR, "deberta_ranker.py"))


# 4. Login to Hugging Face
login(token=HF_TOKEN)
api = HfApi()

# 5. Create repo if it doesn't exist
full_repo_name = f"{HF_USERNAME}/{REPO_NAME}"
if not api.repo_exists(full_repo_name, token=HF_TOKEN):
    api.create_repo(repo_id=REPO_NAME, token=HF_TOKEN, private=False)

# 6. Clone and push
repo = Repository(local_dir=EXPORT_DIR, clone_from=full_repo_name, use_auth_token=HF_TOKEN)
repo.git_add()
repo.git_commit("Initial commit: add all fold checkpoints, config, and model code")
repo.git_push()

print(f"Model and code uploaded to https://huggingface.co/{full_repo_name}")
