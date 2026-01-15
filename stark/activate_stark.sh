#!/bin/bash
# Stark Environment Activation Script
# Automatically activates the conda environment for the stark project

STARK_CONDA_ENV="/home/wlia0047/ar57_scratch/wenyu/stark"

echo "🔄 Activating Stark conda environment..."
conda activate "$STARK_CONDA_ENV"

if [ $? -eq 0 ]; then
    echo "✅ Successfully activated: $STARK_CONDA_ENV"
    echo "📍 Current environment: $CONDA_DEFAULT_ENV"
    echo "🐍 Python path: $(which python3)"
else
    echo "❌ Failed to activate conda environment"
    echo "🔍 Please check if conda is installed and the environment exists"
    echo "📋 Available environments:"
    conda info --envs
fi