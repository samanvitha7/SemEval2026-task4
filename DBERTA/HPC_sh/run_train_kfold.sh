#!/bin/bash
#SBATCH --job-name=semeval_deberta
#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err

echo "Job started on $(hostname)"
echo "Time: $(date)"

# Load CUDA module if available
module load cuda 2>/dev/null || module load cuda/12.1 2>/dev/null || echo "CUDA module not needed"

# Check CUDA availability
nvidia-smi || echo "nvidia-smi not available"

# Go to project directory
cd /home/pruthwikmishra/SemEval-hcp || exit 1

# Set Python to not use system site-packages to avoid conflicts
export PYTHONNOUSERSITE=0
export PYTHONPATH="${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH}"

# Check if torch has CUDA support, if not, install it
python3 -c "import sys; sys.path.insert(0, '${HOME}/.local/lib/python3.11/site-packages'); import torch; exit(0 if torch.cuda.is_available() else 1)" || {
    echo "Installing PyTorch with CUDA support..."
    pip uninstall -y torch torchvision torchaudio
    pip install --user --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # Install compatible numpy version
    pip install --user "numpy<2.0"
}

# Verify CUDA is now available
python3 -c "import sys; sys.path.insert(0, '${HOME}/.local/lib/python3.11/site-packages'); import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'None'); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Use the same python you tested with
python3 train.py --config configs/config.yaml

echo "Job finished at $(date)"
