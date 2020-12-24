#!/usr/bin/env bash
python train.py --optimizer='sgd'  --epochs=400 --gpu=0 --consistency-weight=2.0

