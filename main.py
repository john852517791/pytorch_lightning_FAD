import lightning as L
from utils.arg_parse import f_args_parsed,set_random_seed
import importlib
import os,yaml,shutil
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint,LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
from datetime import datetime
# arguments initialization
args = f_args_parsed()

### temporal config
# 
# args.stage = 1
# 
# ###


# config gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

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
                EarlyStopping('dev_eer',patience=args.no_best_epochs,mode="min",verbose=True,log_rank_zero_only=True),
                # 模型按照最低val_loss来保存
                ModelCheckpoint(monitor='dev_eer',
                                save_top_k=1,
                                save_weights_only=True,mode="min",filename='best_model-{epoch:02d}-{dev_eer:.4f}'),
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
        trainer.test(
            model=customed_model_wrapper,
            datamodule=asvspoof_dm
            )
    else:
        checkpointpath=args.trained_model
        # checkpointpath=trainer.log_dir
        args.savedir = checkpointpath
        
        # gain model
        ymlconf = os.path.join(checkpointpath,"hparams.yaml")
        with open(ymlconf,"r") as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
        infer_m = importlib.import_module(parser1["module_model"])
        test_dm_module = importlib.import_module(parser1["data_module"])
        test_asvspoof_dm = test_dm_module.asvspoof_dataModule(args=args)
            
        infer_model = infer_m.Model(args)
        
        print(parser1)
        
        # print(args.savedir)
        ckpt_files = [file for file in os.listdir(checkpointpath+"/checkpoints/") if file.endswith(".ckpt")]
        # customed_model=model_wrapper.base_model(model=model)
        customed_model=tl_model.base_model.load_from_checkpoint(
            checkpoint_path=os.path.join(f"{checkpointpath}/checkpoints/",ckpt_files[0]),
            model=infer_model,
            args = args,
            strict=False)
        inferer = L.Trainer(logger=pl_loggers.TensorBoardLogger(args.savedir,name=""))
        
        # la19
        inferer.test(
            model=customed_model,
            datamodule=test_asvspoof_dm
            )
        # la21
        inferer.predict(
            model=customed_model,
            datamodule=test_asvspoof_dm
            )
        # df21
        inferer.model.args.testset = "DF21"
        test_asvspoof_dm = test_dm_module.asvspoof_dataModule(args=args)
        inferer.predict(
            model=customed_model,
            datamodule=test_asvspoof_dm
            )
        
        # ITW
        inferer.model.args.testset = "ITW"
        test_asvspoof_dm = test_dm_module.asvspoof_dataModule(args=args)
        inferer.predict(
            model=customed_model,
            datamodule=test_asvspoof_dm
            )

        
        # change the version_0 to infer, and delete useless files
        current_time = datetime.now()
        time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        inferfolder = os.path.join(checkpointpath,f"infer_{time_str}")
        if not os.path.exists(inferfolder):
            os.makedirs(inferfolder)
        folder_a = os.path.join(checkpointpath,"version_0")
        for filename in os.listdir(folder_a):
            if filename.endswith('.log'):  
                original_path = os.path.join(folder_a, filename)
                destination_path = os.path.join(inferfolder, filename)
                shutil.move(original_path, destination_path)
        shutil.rmtree(folder_a)
        
# print(args)