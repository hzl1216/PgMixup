python train.py --dataset=cifar10 --epochs=150 --warmup-step=15 --optimizer='sgd' --consistency-weight=6.0 --lr=0.03  --scheduler=log --n-labeled=250 --gpu=1