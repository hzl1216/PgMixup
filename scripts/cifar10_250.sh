#!/usr/bin/env bash
python train.py --epochs=200  --optimizer='sgd'   --n-labeled=250 --gpu=1  --consistency-weight=8.0 --seed=0 --scheduler=log
