#!/bin/bash
# EVA configuration for uv-based setup

# Set data path
export ROOT_PATH=${ROOT_PATH:-"/home/ubuntu/gravatar/submodules/EVA_Official/test_data/male-3-casual"}
export OUT_PATH=${ROOT_PATH}/smplifyx

# Activate EVA venv
source /home/ubuntu/gravatar/submodules/EVA_Official/.venv_eva/bin/activate

# CUDA paths
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "EVA environment ready:"
echo "  ROOT_PATH=$ROOT_PATH"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
