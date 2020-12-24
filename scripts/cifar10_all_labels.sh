# 4000 2000, 1000, 500
for sup_size in  4000 2000 1000;
do
    python train.py --optimizer='sgd'  --epochs=400 --gpu=0 --consistency-weight=2.0 --n-labeled=${sup_size}
    $@
done
# 500 250
for sup_size in  500 250;
do
   python train.py --epochs=200  --optimizer='sgd'   --n-labeled=${sup_size} --gpu=1  --consistency-weight=8.0 --seed=0
    $@
done


