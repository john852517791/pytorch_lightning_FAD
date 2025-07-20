# model=$1
# tlmodel=$2
# dmmodule=$3
# lr=$2
gpu=$1
module_model=$2
# loss=$5
# --savedir ${train_log_dir} 
# --optim_lr ${lr}
# --gpuid ${gpu} 

model=$(echo "$model" | sed 's/\//./g')
model=$(echo "$model" | sed 's/\.py//g')


if [ -z "$loss" ] || [ "$loss" = "-" ]; then
    loss="CE"
fi

# CUDA_VISIBLE_DEVICES=${gpu} 
line="
nohup python main_loss.py 
--seed 888
--module_model models.moe_research.${module_model} 
--tl_model models.tl_model_moe
--data_module utils.loadData.asvspoof_data_DA
--savedir a_log/${module_model}/2_4_64
--optim_lr 0.00001
--gpuid ${gpu} 
--batch_size 4
--epochs 50
--no_best_epochs 3
--optim adamw
--weight_decay 0.0001
--loss WCE

--scheduler cosWarmup
--num_warmup_steps 3 

--truncate 64600

--moe_topk 2
--moe_experts 4
--moe_exp_hid 128
--loss_weight 0 

--usingDA
> b_gpu_log/25_test_${gpu}.log
&"

# cosWarmup
# --da_prob 0.7
echo ${line}
eval ${line}