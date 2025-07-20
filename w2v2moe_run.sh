# model=$1
# tlmodel=$2
# dmmodule=$3
train_log_dir=$1
lr=$2
gpu=$3
# loss=$5


model=$(echo "$model" | sed 's/\//./g')
model=$(echo "$model" | sed 's/\.py//g')


if [ -z "$loss" ] || [ "$loss" = "-" ]; then
    loss="CE"
fi

# CUDA_VISIBLE_DEVICES=${gpu} 
line="
nohup python main_loss.py 
--seed 888
--module_model models.moe_research.arch_w2v2.w2v2_aasist 
--tl_model models.tl_model
--data_module utils.loadData.asvspoof_data_DA_still_process
--savedir ${train_log_dir} 
--optim_lr ${lr}
--gpuid ${gpu} 
--batch_size 2
--epochs 50
--no_best_epochs 3
--optim adam
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
> b_gpu_log/test_${gpu}.log
&"

# cosWarmup
# --da_prob 0.7
echo ${line}
eval ${line}