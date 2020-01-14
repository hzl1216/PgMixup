# 4000 2000, 1000, 500
for sup_size in  4000 2000 1000;
do
    python train.py --dataset=cifar10 --optimizer='sgd' --warmup-step=20 --lr=0.03 --weight-decay=0.0005 --epochs=400 --gpu=0  --n-labeled=${sup_size}
    $@
done
# 500 250
for sup_size in  500 250;
do
    python train.py --dataset=cifar10 --epochs=150 --warmup-step=15 --optimizer='sgd' --consistency-weight=6.0 --lr=0.03 --weight-decay=0.0007 --scheduler=exp --n-labeled=${sup_size} --gpu=1 --batch-size=16 --unsup-ratio=20

    $@
done


