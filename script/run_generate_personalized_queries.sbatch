#!/bin/bash
#
# SBATCH script to run generate_item_profiles.py (user: semantic item profile generation)
# Minimal, single copy — adapted from run_load_skb.sbatch.
#
# Job name and output
#SBATCH --job-name=gen_personalized_queries
#SBATCH --output=/home/wlia0047/ar57_scratch/wenyu/logs/generate_personalized_queries_%j.out
#SBATCH --error=/home/wlia0047/ar57_scratch/wenyu/logs/generate_personalized_queries_%j.err

# Resources (adjust as needed)
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
set -euo pipefail

# Optional: activate conda/virtualenv if needed. Uncomment and edit the path below.
# source /path/to/conda.sh
# conda activate myenv

cd /home/wlia0047/ar57/wenyu/authorship-llm/code

# Export HF cache to avoid permission issues (optional)
export HF_HOME=/home/wlia0047/.cache/huggingface

# Run the script. Adjust arguments as desired.
python generate_personalized_queries.py \
  --items /home/wlia0047/ar57_scratch/wenyu/generate_item_profiles.json \
  --styles /home/wlia0047/ar57_scratch/wenyu/style_analysis_results_all_beauty_first5.json \
  --out /home/wlia0047/ar57_scratch/wenyu/generated_personalized_queries.json \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --max-tokens 256

echo "Job finished: $(date)"

