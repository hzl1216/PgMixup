# 2000, 1000, 500
for sup_size in  4000 2000 1000 500;
do
    python train.py --dataset=svhn --optimizer='sgd' --warmup-step=15 --lr=0.05 --weight-decay=0.0005 --epochs=150 --gpu=1 --batch-size=32 --unsup-ratio=20  --n-labeled=${sup_size}
    $@
done
#250
python train.py --dataset=svhn --optimizer='sgd' --warmup-step=15 --lr=0.05 --weight-decay=0.0005 --epochs=150 --gpu=2 --n-labeled=250 --batch-size=16 --unsup-ratio=20