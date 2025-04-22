import numpy as np
import soundfile as sf
import torch,os
from torch import Tensor
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from .RawBoost import process_Rawboost_feature
import lightning as L
        
class asvspoof_dataModule(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args
                
                # TODO: change the dir to your own data dir
                # label file
                self.protocols_path = "data/toysets/"
                self.train_protocols_file = self.protocols_path + "sample.txt"
                self.dev_protocols_file = self.protocols_path + "sample.txt"
                # flac file dir
                self.dataset_base_path=self.protocols_path
                self.train_set=self.dataset_base_path
                self.dev_set=self.dataset_base_path
                # test set 
                self.eval_protocols_file_19 = self.protocols_path + "sample.txt"
                self.eval_set_19 = self.dataset_base_path
                self.eval_protocols_file_21 = self.eval_protocols_file_19
                self.eval_set_21 = self.dataset_base_path

                
                self.LA21 = self.protocols_path + "test.txt"
                self.LA21FLAC = self.protocols_path 
                self.LA21TRIAL = self.protocols_path + "trail.txt"

                self.DF21 = self.LA21
                self.DF21FLAC = self.LA21FLAC
                self.DF21TRIAL = self.LA21TRIAL

                self.ITWTXT = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/release_in_the_wild/label.txt"
                self.ITWDIR = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/release_in_the_wild/wav"

                
                
                self.truncate = args.truncate
                self.predict = args.testset # LA21, DF21, ITW

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                d_label_trn,file_train = genSpoof_list(
                    dir_meta=self.train_protocols_file,
                    is_train=True,
                    is_eval=False
                    )
                
                self.asvspoof19_trn_set = Dataset_ASVspoof2019_train(
                    list_IDs=file_train,
                    labels=d_label_trn,
                    base_dir=self.train_set,
                    cut=self.truncate,
                    args= self.args
                    )
   
                _, file_dev = genSpoof_list(
                    dir_meta=self.dev_protocols_file,
                    is_train=False,
                    is_eval=False)
                
                self.asvspoof19_val_set = Dataset_ASVspoof2019_devNeval(
                    list_IDs=file_dev,
                    base_dir=self.dev_set,
                    args= self.args,
                    cut=self.truncate
                    )
   
            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                file_eval = genSpoof_list(
                    dir_meta=self.eval_protocols_file_19,
                    is_train=False,
                    is_eval=True
                    )
                self.asvspoof19_test_set = Dataset_ASVspoof2019_evaltest(
                    list_IDs=file_eval,
                    base_dir=self.eval_set_19,
                    cut=self.truncate
                    )

            if stage == "predict":
                if self.predict == "LA21":
                    file_list=[]
                    with open(self.LA21, 'r') as f:
                        l_meta = f.readlines()
                    for line in l_meta:
                        key= line.strip()
                        file_list.append(key)
                    print(f"no.{(len(file_list))} of eval  trials")
                    self.predict_set = Dataset_ASVspoof2019_evaltest(
                        list_IDs=file_list,
                        base_dir=self.LA21FLAC,
                        cut=self.truncate)
 
                elif self.predict == "DF21":
                    file_list=[]
                    with open(self.DF21, 'r') as f:
                        l_meta = f.readlines()
                    for line in l_meta:
                        key= line.strip()
                        file_list.append(key)
                    print(f"no.{(len(file_list))} of eval  trials")
                    self.predict_set = Dataset_ASVspoof2019_evaltest(
                        list_IDs=file_list,
                        base_dir=self.DF21FLAC,
                        cut=self.truncate)
 
                elif self.predict == "ITW":
                    file_list=[]
                    with open(self.DF21, 'r') as f:
                        l_meta = f.readlines()
                    for line in l_meta:
                        key= line.strip()
                        file_list.append(key)
                    print(f"no.{(len(file_list))} of eval  trials")
                    self.predict_set = Dataset_ASVspoof2019_evaltest(
                        list_IDs=file_list,
                        base_dir=self.DF21FLAC,
                        cut=self.truncate)

                    
                    
                

        def train_dataloader(self):
            return DataLoader(self.asvspoof19_trn_set, batch_size=self.args.batch_size, shuffle=True,drop_last = True,num_workers=4)

        def val_dataloader(self):
            return DataLoader(self.asvspoof19_val_set, batch_size=self.args.batch_size, shuffle=False,drop_last = False,num_workers=4)            

        def test_dataloader(self):                
            datald =  DataLoader(
                self.asvspoof19_test_set,batch_size=self.args.batch_size,
                shuffle=False,num_workers=4
                )
            if "," in self.args.gpuid:
                datald =  DataLoader(
                    self.asvspoof19_test_set,batch_size=self.args.batch_size,
                    shuffle=False,num_workers=4,
                    sampler=DistributedSampler(self.asvspoof19_test_set)
                    )
            return datald

        def predict_dataloader(self):
            predict_loader = DataLoader(
                self.predict_set,
                batch_size= self.args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=4)
            if "," in self.args.gpuid:
                predict_loader = DataLoader(
                    self.predict_set,
                    batch_size= self.args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    sampler=DistributedSampler(self.predict_set),
                    num_workers=4
                    )
            return predict_loader
 
      
      
      

class dataset_itw(Dataset):
    def __init__(self, list_IDs, base_dir,cut = 64600):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(os.path.join(self.base_dir,f"{key}.wav"))
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
    


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir,args,cut = 64600):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.args = args
        self.cut = cut  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir,f"{key}.wav"))

        if self.args.usingDA and (np.random.rand() < self.args.da_prob):
            X=process_Rawboost_feature(X,fs,self.args,self.args.algo)
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        # 1. tensor 2.label 3.filename
        return x_inp, y, key


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir,args=None,cut = 64600):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir,f"{key}.wav"))
        if self.args.usingDA and ("ASVspoof2019_LA_dev" in self.base_dir):
            X=process_Rawboost_feature(X,fs,self.args,self.args.algo)
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        # 1.tensor 2.filename
        return x_inp, key


class Dataset_ASVspoof2019_evaltest(Dataset):
    def __init__(self, list_IDs, base_dir,args=None,cut = 64600):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir,f"{key}.wav"))
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key 
      
      
      
      
      
      
      
      
