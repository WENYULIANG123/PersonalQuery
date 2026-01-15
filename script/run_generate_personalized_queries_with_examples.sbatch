#!/bin/bash
#
# SBATCH script to run generate_personalized_queries_with_examples.py
# Usage: sbatch /home/wlia0047/ar57_scratch/wenyu/run_generate_personalized_queries_with_examples.sbatch
#
#SBATCH --job-name=gen_personalized_queries_ex
#SBATCH --output=/home/wlia0047/ar57_scratch/wenyu/logs/generate_personalized_queries_with_examples_%j.out
#SBATCH --error=/home/wlia0047/ar57_scratch/wenyu/logs/generate_personalized_queries_with_examples_%j.err
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

# Ensure HF cache location
export HF_HOME=/home/wlia0047/.cache/huggingface

python generate_personalized_queries_with_examples.py \
  --items /home/wlia0047/ar57_scratch/wenyu/generate_item_profiles.json \
  --styles /home/wlia0047/ar57_scratch/wenyu/style_analysis_results_all_beauty_first5.json \
  --examples-csv /home/wlia0047/ar57_scratch/wenyu/stark_qa_human_generated_eval.csv \
  --out /home/wlia0047/ar57_scratch/wenyu/generated_personalized_queries_with_examples.json \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --max-tokens 256

echo "Job finished: $(date)"



