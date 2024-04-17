# --module-model models.pure.stable_aasist
ckpt=$1
gpu=$2


# --module-model models.rawformer.stable_base
line=" 
nohup python main.py --inference 
--trained_model ${ckpt}
--batch_size 100
--gpuid ${gpu}
> ${ckpt}/z_infer.log
&"
# --truncate 96000

# --colour 2



echo ${line}
eval ${line}