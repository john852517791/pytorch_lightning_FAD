scoreFilepath=$1
eval_name=$2

line="python utils/tools/cul_eer.py --pos 2 --scoreFile $scoreFilepath/infer_19.log > $scoreFilepath/eer_19"
# line="python utils/tools/cul_eer19.py --scoreFile $scoreFilepath/infer/infer_19.log "
echo $line
eval $line
line="python utils/tools/cul_eer21.py --scoreFile $scoreFilepath/infer_LA21.log > $scoreFilepath/eer_21"
echo $line
eval $line
line="python utils/tools/cul_eer21df.py --scoreFile $scoreFilepath/infer_DF21.log > $scoreFilepath/eer_df21"
echo $line
eval $line
line="python utils/tools/cul_itw.py --scoreFile $scoreFilepath/infer_ITW.log > $scoreFilepath/eer_itw"
echo $line
eval $line