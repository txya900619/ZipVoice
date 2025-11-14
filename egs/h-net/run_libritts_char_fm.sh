#!/bin/bash

# This is an example script for training ZipVoice on LibriTTS dataset.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=2
stop_stage=9

#### Prepare datasets (1)

# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#       echo "Stage 1: Data Preparation for LibriTTS dataset"
#       bash local/prepare_libritts.sh
# fi

### Training ZipVoice (2 - 3)

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Train the ZipVoice model"
      torchrun --nproc-per-node 4 --standalone -m zipvoice.bin.train_hnet_tts_fm \
            --use-fp16 1 \
            --num-iters 100000 \
            --max-duration 160 \
            --max-len 20 \
            --base-lr 5e-4 \
            --valid-by-epoch 1 \
            --model-config conf/hnet_2stage_small.json \
            --tokenizer libritts \
            --token-file data/tokens_libritts.txt \
            --dataset libritts \
            --manifest-dir data/fbank \
            --input-strategy PrecomputedFeaturesNJT \
            --exp-dir exp/hnet_libritts_fm_lr_5e-4_N2 \
            --loss-rt-weight 1e-3 \
            --feat-scale 0.4343
fi

# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#       echo "Stage 3: Average the checkpoints for ZipVoice"
#       python3 -m zipvoice.bin.generate_averaged_model \
#             --epoch 60 \
#             --avg 10 \
#             --model-name zipvoice \
#             --exp-dir exp/zipvoice_libritts
#       # The generated model is exp/zipvoice_libritts/epoch-60-avg-10.pt
# fi

### Inference with PyTorch models (8 - 9)

# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#       echo "Stage 8: Inference of the ZipVoice model"
#       python3 -m zipvoice.bin.infer_zipvoice \
#             --model-name zipvoice \
#             --model-dir exp/zipvoice_libritts \
#             --checkpoint-name epoch-60-avg-10.pt \
#             --tokenizer libritts \
#             --test-list test.tsv \
#             --res-dir results/test_libritts \
#             --num-step 8 \
#             --guidance-scale 1 \
#             --t-shift 0.7
# fi
