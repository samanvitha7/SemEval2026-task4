#!/bin/bash
#SBATCH --job-name=semeval_embeddings
#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embeddings.out
#SBATCH --error=logs/embeddings.err

echo "Job started on $(hostname)"
echo "Time: $(date)"

# Load CUDA module if available
module load cuda 2>/dev/null || module load cuda/12.1 2>/dev/null || echo "CUDA module not needed"

# Check CUDA availability
nvidia-smi || echo "nvidia-smi not available"

# Go to project directory
cd /home/pruthwikmishra/SemEval-hcp || exit 1


# Activate virtual environment
source .venv/bin/activate

# Verify CUDA is available
.venv/bin/python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None'); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('Using device:', 'cuda' if torch.cuda.is_available() else 'cpu')"

# Run ensemble embeddings extraction for Track-B
.venv/bin/python generate_embeddings.py --test_file data/testb.jsonl --output_file submission_trackb.jsonl --use_ensemble

echo "Job finished at $(date)"
