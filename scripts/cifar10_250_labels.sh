python train.py --dataset=cifar10 --epochs=200 --warmup-step=10  --optimizer='sgd' --consistency-weight=6.0 --lr=0.03 --weight-decay=0.0007  --n-labeled=250 --gpu=1 --batch-size=16 --unsup-ratio=20
