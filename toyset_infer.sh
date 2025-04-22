# --module-model models.pure.stable_aasist
ckpt="a_train_log/aasist/version_0"
gpu=0


# --module-model models.rawformer.stable_base
# nohup 
line=" 
python main.py --inference 
--trained_model ${ckpt}
--batch_size 1
--gpuid ${gpu}
> ${ckpt}/z_infer.log
"
# &
# --truncate 96000

# --colour 2



echo ${line}
eval ${line}