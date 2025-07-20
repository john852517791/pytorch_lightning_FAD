# --module-model models.pure.stable_aasist
ckpt=$1
gpu=$2


# --module-model models.rawformer.stable_base
line=" 
nohup python main.py --inference 
--trained_model ${ckpt}
--batch_size 128
--gpuid ${gpu}
--moe_topk 2
--moe_experts 4
--moe_exp_hid 128
> ${ckpt}/z_infer_1.log
&"
# --truncate 96000

# --colour 2



echo ${line}
eval ${line}