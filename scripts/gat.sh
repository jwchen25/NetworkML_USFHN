#!/bin/bash

METHOD="gat"

for i in chemistry mathematics physics computer_science
do
echo "USFHN subset: $i"
python -u train.py \
          --method $METHOD \
          --subset $i \
          --conv_layers 3 \
          --hidden_dim 256 \
          --lr 0.001 \
          --epochs 100 \
          --runs 10 \
          > exp_results/${METHOD}_${i}.log
done
