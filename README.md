# pytorch_lightning_FAD

This is a general framework for fake audio detection using pytorch lightning
the dataset used here is asvspoof2019

# env

python 3.9

```
pip install -r requirement.txt
```

# run sample

first thing first, change the dir in the "utils/loadData/asvspoof_data_DA.py"

the run this line

```
bash run.sh a_train_log/aasist 0.01 6
```

# usage

## 1. data module

if you want to use anthor data input format, please reference file 'utils/loadData/asvspoof_data_DA.py' to write the datamodule.

if you won't change anything in the "models/tl_model.py", please **make sure that the train set return three elements (tensor, label, filename), and the dev/test set return two elements (tensor, filename)**

then change the **"--data_module"** config when you run the "run.sh"

## 2. model

if you want to use another model architecture, add it in to the folder "models".

if you won't change anything in the "models/tl_model.py", please **make sure the model you create return at least two elements** (prediction and hidden state) and change the model class name to "Model"

then change the **"--module_model"** config when you run the "run.sh"

## 3. tl_model

if you want to modify something in the train/eval/test/inference stage (like modification about the loss culculation), create a new file and reference file "models/tl_model.py"

then change the **"--tl_model"** config when you run the "run.sh"

# Generalized Fake Audio Detection via Deep Stable Learning [arxiv](https://arxiv.org/pdf/2406.03237)
reweight leaner is in utils/ideas/
check usage in the tl_model_file (models/tl_model_postft_loss.py) and model file (models/wav2vec/l5_aasist_step_stable.py)
and follow the usage of this framework mentioned above




