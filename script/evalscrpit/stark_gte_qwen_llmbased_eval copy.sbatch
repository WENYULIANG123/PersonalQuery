#!/bin/bash
#
# SBATCH script to run GTE LLM-based evaluation on STaRK dataset
# Uses alibaba-nlp/gte-base-en-v1.5 model for document and query embeddings
# Optimized batch sizes for memory efficiency
#
# Job name and output
#SBATCH --job-name=stark_gte_llmbased_eval
#SBATCH --output=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.out
#SBATCH --error=/home/wlia0047/ar57_scratch/wenyu/logs/%x_%j.err

# Resources (adjusted for GTE-base-en-v1.5 model)
#SBATCH --time=7-00:00:00  # 7天
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G

set -euo pipefail

# Initialize and activate conda environment
if [ -f /etc/profile.d/conda.sh ]; then
  source /etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || {
    source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || {
      export PATH="$(conda info --base 2>/dev/null)/bin:$PATH" 2>/dev/null || true
    }
  }
fi

module load cuda || true

# Activate conda environment
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# Set HuggingFace cache directory
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_cache

# Change to STaRK directory
cd /home/wlia0047/ar57/wenyu/stark

echo "🏗️ Generating GTE-base-en-v1.5 embeddings for amazon dataset..."
echo "📚 This will process all documents and generate LLM-based embeddings"
echo "⚙️  Using optimized batch sizes for memory efficiency"
echo ""

# Check if embeddings already exist
EMB_PATH="emb/amazon/alibaba-nlp/gte-base-en-v1.5/doc/candidate_emb_dict.pt"
if [ -f "$EMB_PATH" ]; then
    echo "✅ Embeddings already exist at $EMB_PATH"
    echo "🔍 Checking embeddings validity..."
    python -c "
import torch
import sys
try:
    emb_dict = torch.load('$EMB_PATH', map_location='cpu')
    if isinstance(emb_dict, dict) and len(emb_dict) > 0:
        sample_emb = next(iter(emb_dict.values()))
        print(f'✅ Valid embeddings found: {len(emb_dict)} items, shape: {sample_emb.shape}')
        print('⏭️  Skipping embedding generation...')
        sys.exit(0)
    else:
        print('⚠️  Invalid embeddings found, will regenerate...')
        sys.exit(1)
except Exception as e:
    print(f'⚠️  Error loading embeddings: {e}')
    print('🔄 Will regenerate embeddings...')
    sys.exit(1)
" && SKIP_GENERATION=true || SKIP_GENERATION=false
else
    SKIP_GENERATION=false
fi

# Generate embeddings if needed
if [ "$SKIP_GENERATION" = false ]; then
    echo "🏗️ Generating GTE-base-en-v1.5 embeddings for amazon dataset..."
    echo "📚 This will process all documents and generate LLM-based embeddings"
    echo ""

    # Run the optimized embedding generation script with GTE-base-en-v1.5 model
    python emb_generate.py \
      --dataset amazon \
      --emb_model alibaba-nlp/gte-base-en-v1.5 \
      --mode doc \
      --batch_size 4 \
      --add_rel \
      --compact

    echo "✅ Qwen2.5-1.5B embedding generation completed!"
else
    echo "✅ Using existing embeddings"
fi

echo ""
echo "🔍 Starting evaluation..."

# Evaluate on human generated queries (original queries)
echo "📊 Evaluating on human generated queries (original)..."
python eval.py \
  --dataset amazon \
  --model VSS \
  --emb_model alibaba-nlp/gte-base-en-v1.5 \
  --split human_generated_eval \
  --batch_size 128

echo "✅ Human generated queries evaluation completed!"

# Evaluate on query variants (by strategy)
echo ""
echo "📊 Evaluating on query variants by strategy..."

cd /home/wlia0047/ar57/wenyu/stark

# Define strategies (including error_aware)
strategies=("character" "embedding" "other" "typo" "wordnet" "dependency" "error_aware")

# OPTIMIZATION: Pre-generate all variant embeddings at once (for traditional strategies)
echo "🚀 OPTIMIZATION: Pre-generating traditional variant query embeddings..."
python emb_generate.py \
  --dataset amazon \
  --emb_model alibaba-nlp/gte-base-en-v1.5 \
  --mode query \
  --batch_size 16 \
  --dataset_root data/stark_variants_dataset

echo "✅ Traditional variant embeddings pre-generated!"

# Evaluate each strategy separately (with smart embedding reuse)
for strategy in "${strategies[@]}"; do
    echo ""
    echo "🔄 Evaluating strategy: $strategy"
    echo "----------------------------------------"

    # Create dataset for this strategy (if not exists)
    if [ "$strategy" = "error_aware" ]; then
        dataset_dir="data/stark_error_aware_dataset"
    else
        dataset_dir="data/stark_${strategy}_variants_dataset"
    fi

    if [ ! -d "$dataset_dir" ]; then
        python create_variants_dataset.py --strategy $strategy
    fi

    # Check if embeddings already exist for this strategy
    if [ "$strategy" = "error_aware" ]; then
        EMB_PATH="emb/amazon/alibaba-nlp/gte-base-en-v1.5/query_no_rel_no_compact_stark_error_aware_dataset/query_emb_dict.pt"
        dataset_root_param="--dataset_root data/stark_error_aware_dataset"
    else
        EMB_PATH="emb/amazon/alibaba-nlp/gte-base-en-v1.5/query_no_rel_no_compact_stark_${strategy}_variants_dataset/query_emb_dict.pt"
        dataset_root_param="--dataset_root data/stark_${strategy}_variants_dataset"
    fi

    if [ -f "$EMB_PATH" ]; then
        echo "✅ Query embeddings already exist for $strategy, skipping generation"
    else
        echo "🏗️ Generating query embeddings for strategy: $strategy"
        python emb_generate.py \
          --dataset amazon \
          --emb_model alibaba-nlp/gte-base-en-v1.5 \
          --mode query \
          --batch_size 4 \
          $dataset_root_param
    fi

    # Evaluate this strategy
    if [ "$strategy" = "error_aware" ]; then
        dataset_root_param="--dataset_root data/stark_error_aware_dataset"
    else
        dataset_root_param="--dataset_root data/stark_${strategy}_variants_dataset"
    fi

    python eval.py \
      --dataset amazon \
      --model VSS \
      --emb_model alibaba-nlp/gte-base-en-v1.5 \
      $dataset_root_param \
      --split variants \
      --strategy $strategy \
      --batch_size 128 \
      --force_rerun

    echo "✅ Strategy $strategy evaluation completed!"
done

echo "✅ All query variant strategies evaluation completed!"

# Create LLMbasedeval directory and copy results
echo "📁 Organizing results in LLMbasedeval directory..."
mkdir -p /home/wlia0047/ar57/wenyu/stark/LLMbasedeval
cp LLMbasedeval/amazon/VSS/alibaba-nlp/gte-base-en-v1.5/eval_metrics_*.json /home/wlia0047/ar57/wenyu/stark/LLMbasedeval/ 2>/dev/null || true

echo "📊 All evaluation metrics saved in: /home/wlia0047/ar57/wenyu/stark/LLMbasedeval/"
echo "🎉 All tasks completed successfully!"
