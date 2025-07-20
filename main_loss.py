import os,yaml,shutil
from utils.arg_parse import f_args_parsed,set_random_seed
args = f_args_parsed()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
import lightning as L
import importlib
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint,LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
# arguments initialization

### temporal config
# 
# args.stage = 1
# 
# ###


# config gpu

# random seed initialization and gpu seed 
set_random_seed(args.seed, args)

# config the base model containing train eval test and inference funtion
tl_model = importlib.import_module(args.tl_model)

# config the data module containing the train set, dev set and test set
dm_module = importlib.import_module(args.data_module)
asvspoof_dm = dm_module.asvspoof_dataModule(args=args)

if True:
    # ⭐train 
    if not args.inference:
        # import model.py
        prj_model = importlib.import_module(args.module_model)
        
        # model 
        model = prj_model.Model(args)

        # init model, including loss func and optim 
        customed_model_wrapper = tl_model.base_model(
            model=model,
            args=args
            )

        # config logdir
        tb_logger = pl_loggers.TensorBoardLogger(args.savedir,name="")
        
        # model initialization
        trainer = L.Trainer(
            max_epochs=args.epochs,
            strategy='ddp_find_unused_parameters_true',
            log_every_n_steps = 1,
            callbacks=[
                # dev损失无下降就提前停止
                EarlyStopping('loss',patience=args.no_best_epochs,mode="min",verbose=True,log_rank_zero_only=True),
                # 模型按照最低val_loss来保存
                ModelCheckpoint(monitor='loss',
                                save_top_k=1,
                                save_weights_only=True,mode="min",filename='best_model-{epoch:02d}-{dev_eer:.4f}-{loss:.4f}'),
                LearningRateMonitor(logging_interval='epoch',log_weight_decay=True),
                ],
            check_val_every_n_epoch=1,
            logger=tb_logger,
            enable_progress_bar=False
            )
        trainer.fit(
            model=customed_model_wrapper, 
            datamodule=asvspoof_dm
            )
        
        # # test 19 default
        # trainer.test(
        #     model=customed_model_wrapper,
        #     datamodule=asvspoof_dm
        #     )
    