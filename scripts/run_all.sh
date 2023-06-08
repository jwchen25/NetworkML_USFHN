#!/bin/bash

for i in gcn sage gin gat gatv2 deeper trans
do
echo "Method: $i"
bash scripts/train.sh $i
done
