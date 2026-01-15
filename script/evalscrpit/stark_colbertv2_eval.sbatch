#!/bin/bash
#
# SBATCH script to run STaRK DPR evaluation with PRE-TRAINED EMBEDDINGS
# Uses downloaded OpenAI embeddings instead of real-time encoding
#
# Job name and output
#SBATCH --job-name=stark_colbertv2_eval
#SBATCH --output=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.out
#SBATCH --error=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.err

# Resources (Quad-GPU accelerated DPR encoding)
#SBATCH --time=7-00:00:00  # 7天
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1                # Request 4 GPUs for maximum parallel processing
#SBATCH --cpus-per-task=16       # More CPUs for data loading
#SBATCH --mem=1000G               # More memory for parallel processing
# Get current script name (without extension) for dynamic referencing
SCRIPT_NAME=stark_colbertv2_eval  

# Get current SLURM job ID for tracking and logging
JOB_ID=${SLURM_JOB_ID:-"unknown"}

# Clean up previous log files (excluding current job's logs)
# Remove old grammer_aware log files except current job
for log_file in /home/wlia0047/ar57_scratch/wenyu/logs/${SCRIPT_NAME}_*.out; do
    if [[ -f "$log_file" && "$log_file" != "/home/wlia0047/ar57_scratch/wenyu/logs/${SCRIPT_NAME}_${JOB_ID}.out" ]]; then
        rm -f "$log_file"
        echo "Removed old log file: $log_file"
    fi
done  

for log_file in /home/wlia0047/ar57_scratch/wenyu/logs/${SCRIPT_NAME}_*.err; do
    if [[ -f "$log_file" && "$log_file" != "/home/wlia0047/ar57_scratch/wenyu/logs/${SCRIPT_NAME}_${JOB_ID}.err" ]]; then
        rm -f "$log_file"
        echo "Removed old error file: $log_file"
    fi
done

# Display job information
echo "=========================================="
echo "SLURM Job Information:"
echo "Script Name: $SCRIPT_NAME"
echo "Job ID: $JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

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

# Set environment variables for optimal single-GPU performance
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0  # Force single GPU usage
export TORCH_USE_CUDA_DSA=1
export NCCL_IB_DISABLE=1  # Disable InfiniBand for compatibility
export NCCL_SOCKET_IFNAME=lo  # Use loopback interface for single GPU

# Create logs directory if it doesn't exist
mkdir -p /home/wlia0047/ar57_scratch/wenyu/logs

# Run the ColBERTv2 evaluation script
echo "Launching ColBERTv2 evaluation..."
echo "Current directory: $(pwd)"
echo "Python3 version: $(python3 --version 2>&1)"

# Change to STaRK directory and run the script
cd /home/wlia0047/ar57/wenyu/stark
echo "Changed to directory: $(pwd)"


# Execute with timeout to prevent hanging
python3 /home/wlia0047/ar57/wenyu/stark/code/run_stark_colbert_eval.py --strategy grammar_aware --force_rerun

