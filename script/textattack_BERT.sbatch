#!/bin/bash
#
# SBATCH script to run STaRK DPR evaluation with PRE-TRAINED EMBEDDINGS
# Uses downloaded OpenAI embeddings instead of real-time encoding
#
# Job name and output
#SBATCH --job-name=textattack_BERT
#SBATCH --output=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.out
#SBATCH --error=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.err

# Resources (Quad-GPU accelerated DPR encoding)
#SBATCH --time=7-00:00:00  # 7天
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1                # Request 4 GPUs for maximum parallel processing
#SBATCH --cpus-per-task=16       # More CPUs for data loading
#SBATCH --mem=256G               # More memory for parallel processing

set -euo pipefail

# Initialize and activate conda environment
echo "Initializing conda environment..."
if [ -f /etc/profile.d/conda.sh ]; then
  source /etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || {
    source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || {
      export PATH="$(conda info --base 2>/dev/null)/bin:$PATH" 2>/dev/null || true
    }
  }
fi

# Activate conda environment
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# Load CUDA module for GPU support
echo "Loading CUDA module for GPU acceleration..."
module load cuda || echo "Warning: CUDA module load failed"

# Verify environment activation
echo "Python executable: $(which python3)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-not_activated}"

# Set HuggingFace cache directory
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_cache
echo "HF_HOME set to: $HF_HOME"

# Set environment variables for optimal multi-GPU performance
export PYTHONUNBUFFERED=1
# Allow all GPUs to be visible (removed CUDA_VISIBLE_DEVICES restriction)
export TORCH_USE_CUDA_DSA=1
export NCCL_IB_DISABLE=1  # Disable InfiniBand for compatibility
export NCCL_SOCKET_IFNAME=ib0  # Use specific network interface if available

# Memory optimization for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Create logs directory if it doesn't exist
mkdir -p /home/wlia0047/ar57_scratch/wenyu/logs

# Run the textattack_BERT script
echo "Launching textattack_BERT script..."
echo "Current directory: $(pwd)"
echo "Python3 version: $(python3 --version 2>&1)"

# Change to STaRK directory and run the script
cd /home/wlia0047/ar57/wenyu/TextAttack/code
echo "Changed to directory: $(pwd)"
python3 personalized_rewriting_example.py