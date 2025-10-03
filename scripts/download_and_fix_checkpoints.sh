#!/bin/bash
# Download and fix corrupt checkpoints from EVA tar.gz extraction

set -e

echo "=========================================="
echo "Downloading and Fixing EVA Checkpoints"
echo "=========================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVA_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$EVA_ROOT/.venv_eva_stage2/bin/python"

# 1. MANO models
echo "1. Downloading MANO models..."
mkdir -p "$EVA_ROOT/preprocess/hamer/_DATA/data/mano"
cd "$EVA_ROOT/preprocess/hamer/_DATA/data/mano"
if [ ! -f "MANO_RIGHT.pkl" ]; then
    wget https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_RIGHT.pkl
fi
if [ ! -f "MANO_LEFT.pkl" ]; then
    wget https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_LEFT.pkl
fi
echo "✓ MANO models downloaded"
echo ""

# 2. ViTPose wholebody checkpoint
echo "2. Downloading ViTPose wholebody checkpoint..."
cd "$EVA_ROOT/preprocess/hamer/_DATA/vitpose_ckpts/vitpose+_huge"
if [ -f "wholebody.pth" ]; then
    mv wholebody.pth wholebody_corrupt.pth
fi
wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-h-wholebody.pth -O wholebody.pth
echo "✓ ViTPose checkpoint downloaded"
echo ""

# 3. Detectron2 checkpoint (convert from pickle to PyTorch ZIP format)
echo "3. Downloading and converting Detectron2 checkpoint..."
DETECTRON_DIR="$HOME/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692"
mkdir -p "$DETECTRON_DIR"
cd "$DETECTRON_DIR"

# Download official detectron2 checkpoint
if [ ! -f "model_final_f05665_official.pkl" ]; then
    wget https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl \
        -O model_final_f05665_official.pkl
fi

# Convert to PyTorch format
echo "Converting detectron2 pickle to PyTorch ZIP format..."
$VENV << 'PYEOF'
import pickle
import torch
import io

class TorchUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if isinstance(pid, tuple):
            typename, data_tuple = pid[0], pid[1:]
            if typename == 'storage':
                storage_type, root_key, location, numel, view_metadata = data_tuple
                if storage_type == torch.FloatStorage:
                    storage = torch.FloatStorage(numel)
                elif storage_type == torch.LongStorage:
                    storage = torch.LongStorage(numel)
                else:
                    storage = torch.UntypedStorage(numel)
                return storage
        return None

with open('model_final_f05665_official.pkl', 'rb') as f:
    checkpoint = TorchUnpickler(f).load()

torch.save(checkpoint, 'model_final_f05665.pkl')
print("✓ Converted to PyTorch format")
PYEOF

echo "✓ Detectron2 checkpoint converted"
echo ""

echo "=========================================="
echo "✅ All checkpoints downloaded and fixed!"
echo "=========================================="
echo ""
echo "Fixed checkpoints:"
echo "  - MANO_LEFT.pkl, MANO_RIGHT.pkl (7.3MB)"
echo "  - wholebody.pth (2.55GB) - ViTPose"
echo "  - model_final_f05665.pkl (3.9GB) - Detectron2"
echo ""
