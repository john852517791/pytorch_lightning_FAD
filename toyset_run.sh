# model=$1
# tlmodel=$2
# dmmodule=$3
train_log_dir=a_train_log/aasist 
lr=0.01
gpu=0
# loss=$5


model=$(echo "$model" | sed 's/\//./g')
model=$(echo "$model" | sed 's/\.py//g')


if [ -z "$loss" ] || [ "$loss" = "-" ]; then
    loss="CE"
fi

# CUDA_VISIBLE_DEVICES=${gpu} 
# nohup 
line="
python main.py 
--seed 1234
--module_model models.aasist.AASIST
--tl_model models.tl_model
--data_module utils.loadData.toyset_dm
--savedir ${train_log_dir} 
--optim_lr ${lr}
--gpuid ${gpu} 
--batch_size 2
--epochs 3
--no_best_epochs 2
--optim adam
--weight_decay 0.0001
--loss WCE
--scheduler cosAnneal
--truncate 64600
> b_gpu_log/test_${gpu}.log
"
# &
# --usingDA
# --da_prob 0.7
echo ${line}
eval ${line}