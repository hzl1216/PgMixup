python train.py --dataset=svhn --optimizer='sgd' --warmup-step=15 --lr=0.05 --weight-decay=0.0005 --epochs=150 --gpu=1 --batch-size=32 --unsup-ratio=20  --n-labeled=4000