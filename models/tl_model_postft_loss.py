from typing import Any
import lightning as L
import torch
import logging,os
from utils.wrapper import loss_wrapper, optim_wrapper,schedule_wrapper   
from utils.tools import cul_eer 
# from models.wav2vec.l5_aasist_step import Model
from utils.ideas.reweight_learner import weight_learner
from utils.ideas.reweight_learner import args as stable_arg



class base_model(L.LightningModule):
    def __init__(self, 
                 model,
                 args,
                 ) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.stable_conf = stable_arg()
        self.args.stable_conf = str(vars(self.stable_conf))
        self.save_hyperparameters(self.args)
        
        self.model_optimizer = optim_wrapper.optimizer_wrap(self.args, self.model).get_optim()
        self.LRScheduler = schedule_wrapper.scheduler_wrap(self.model_optimizer,self.args).get_scheduler()
        # for loss
        self.args.model = model
        self.args.samloss_optim = self.model_optimizer
        self.loss_criterion,self.loss_optimizer,self.minimizor = loss_wrapper.loss_wrap(self.args).get_loss()
        
        
        self.logging_test = None
        self.logging_predict = None
        
        
    def forward(self,x):
        return self.model(x)
    
    # def on_train_epoch_start(self):
    #     if self.args.start_ft != 0:
    #         print(self.current_epoch)
    #         if self.current_epoch >= self.args.start_ft:
    #             self.model.unfreeze_parameters()
    #         else:
    #             self.model.freeze_parameters()
            # self.model.unfreeze_parameters()
            
    
    def training_step(self, batch, batch_idx):
        
        # batch[0] -- tensor
        # batch[1] -- label
        # batch[2] -- filename
        
                
        # model output, better return 2 elements, prediction and any other thing
        output = self.forward(batch[0])
        batch_loss = self.loss_criterion(output[0], batch[1])
        
           # stable
        pre_features = self.model.pre_features
        pre_weight1 = self.model.pre_weight1
        
        loss_weight, pre_features, pre_weight1 = weight_learner(
                output[1], 
                pre_features, 
                pre_weight1, 
                args=self.stable_conf,
                global_epoch = self.current_epoch, iter = batch_idx)
        self.model.pre_features.data.copy_(pre_features)
        self.model.pre_weight1.data.copy_(pre_weight1)
        batch_loss =  batch_loss.view(1, -1).mm(loss_weight).view(1)
        
        
        
        batch_loss = batch_loss.mean()
        self.log_dict({
            "loss": batch_loss,
            },on_step=True, 
                on_epoch=True,prog_bar=True, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        return batch_loss
        
    def validation_step(self,batch):
        # batch[0] -- tensor
        # batch[1] -- label
        # batch[2] -- filename
        
        # model output
        output = self.forward(batch[0])
        
        softmax_pred = torch.nn.functional.softmax(output[0],dim=1)
        
        # log the prediction for cul eer
        with open(os.path.join(self.logger.log_dir,"dev.log"), 'a') as file:
            for i in range(len(softmax_pred)):
                file.write(f"{batch[2][i]} {str(softmax_pred.cpu().numpy()[i][1])}\n")
        
        # batch_loss = self.loss_criterion(data_predict, data_label).mean()
        # # Logging to TensorBoard (if installed) by default
        # self.log("val_loss", batch_loss, batch_size=len(data_in),sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        # culculate the dev eer
        dev_eer = 0.
        dev_tdcf = 0.
        with open(os.path.join(self.logger.log_dir,"dev.log"), 'r') as file:
            lines = file.readlines()

        if len(lines) > 10000:
            if "singfake" in self.args.data_module:
                dev_eer = cul_eer.eeronly(
                    os.path.join(self.logger.log_dir,"dev.log"),
                    "/data8/wangzhiyong/project/fakeAudioDetection/pytorch_lightning_FAD/datasets/sing_fsd/dataset/label/dev.txt",
                ) 
            else:
                dev_eer, dev_tdcf = cul_eer.eerandtdcf(
                    os.path.join(self.logger.log_dir,"dev.log"),
                    "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/datasets/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
                    "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/datasets/asvspoof2019/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
                )
        with open(os.path.join(self.logger.log_dir,"dev.log"), 'w') as file:
            pass
        self.log_dict({
            "dev_eer": (dev_eer),
            "dev_tdcf": dev_tdcf,
            },on_step=False, 
                on_epoch=True,prog_bar=False, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        
    def on_test_start(self):
        # logging.basicConfig(filename=os.path.join(self.logger.log_dir,f"infer_test.log"),level=logging.INFO,format="")
        self.logging_test = logging.getLogger("logging_test")
        self.logging_test.setLevel(logging.INFO)
        hdl=logging.FileHandler(os.path.join(self.logger.log_dir,f"infer_19.log"))
        hdl.setFormatter("")
        self.logging_test.addHandler(hdl)        
        
    def test_step(self, batch,) -> Any:
        # batch[0] -- tensor
        # batch[1] -- filename
        
        # model output
        output = self.forward(batch[0])
        
        data_predict = torch.nn.functional.softmax(output[0],dim=1)
        
        for i in range(len(batch[1])):
            self.logging_test.info(f"{batch[1][i]} {str(data_predict.cpu().numpy()[i][0])} {str(data_predict.cpu().numpy()[i][1])}")
        # return data_info[0],data_predict.cpu().numpy()
        return {'loss': 0, 'y_pred': data_predict}
    
    def on_predict_start(self):
        # logging.basicConfig(filename=os.path.join(self.args.savedir,f"infer_predict.log"),level=logging.INFO,format="")
        self.logging_predict = logging.getLogger(f"logging_predict_{self.args.testset}")
        self.logging_predict.setLevel(logging.INFO)
        hdlx = logging.FileHandler(os.path.join(self.logger.log_dir,f"infer_{self.args.testset}.log"))
        hdlx.setFormatter("")
        self.logging_predict.addHandler(hdlx)
    
    def predict_step(self, batch, batch_idx):
        # batch[0] -- tensor
        # batch[1] -- filename
        
        # model output
        output = self.forward(batch[0])
        
        data_predict = torch.nn.functional.softmax(output[0],dim=1)
         
        # self.logging_predict.info(f"{data_info[0]} {str(data_predict.cpu().numpy()[0][1])} {str(data_predict.cpu().numpy()[0][0])}")
        for i in range(len(batch[1])):
            self.logging_predict.info(f"{batch[1][i]} {str(data_predict.cpu().numpy()[i][1])}")
        # return data_info[0],data_predict.cpu().numpy()
        return 

    def configure_optimizers(self):
        configure = None
        if self.LRScheduler is not None:
            configure = {
                "optimizer":self.model_optimizer,
                'lr_scheduler': self.LRScheduler, 
                'monitor': 'dev_eer'
                }
        else:
            configure = {
                "optimizer":self.model_optimizer,
                }
            
        return configure