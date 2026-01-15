#!/bin/bash
#
# SBATCH script to run generate_item_profiles.py (user: semantic item profile generation)
# Minimal, single copy — adapted from run_load_skb.sbatch.
#
# Job name and output
#SBATCH --job-name=generate_item_profiles
#SBATCH --output=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.out
#SBATCH --error=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.err

# Resources (adjust as needed)
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Initialize conda and activate the target environment
eval "$(conda shell.bash hook)"
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# Run the script with unbuffered output and save a timestamped log.
# Set HF_TOKEN in the environment if needed; the script will read --hf-token if provided.
# Additional args: --style-file (style analysis JSON), --pick-method (first|random), --out (output JSON)
python -u /home/wlia0047/ar57/wenyu/authorship-llm/code/generate_item_profiles.py \
  --count 5 \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --num-factors 10 \
  --explain-tokens 250 \
  --hf-token "$HF_TOKEN" \
  --path /home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/raw/meta_categories/skb_output/processed/node_info.pkl \
  --style-file /home/wlia0047/ar57_scratch/wenyu/style_analysis_results_all_beauty_first5.json \
  --pick-method first \
  --out /home/wlia0047/ar57_scratch/wenyu/generate_item_profiles.json \
  2>&1 | tee /home/wlia0047/ar57_scratch/wenyu/logs/generate_item_profiles_$(date +%s).log

