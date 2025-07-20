from models.tl_model import base_model as original_base_model


class base_model(original_base_model):
    def __init__(self, 
                 model,
                 args,
                 ) -> None:
        super().__init__(model,args)
        
    def forward(self,x,train = False):
        return self.model(x,train)
    def training_step(self, batch, batch_idx):
        # batch[0] -- tensor
        # batch[1] -- label
        # batch[2] -- filename
        # model output, better return 2 elements, prediction and any other thing
        output = self.forward(batch[0],train=True)
        batch_loss = self.loss_criterion(output[0], batch[1])
        
        batch_loss = batch_loss.mean() + self.args.loss_weight * output[2]
        self.log_dict({
            "loss": batch_loss,
            },on_step=True, 
                on_epoch=True,prog_bar=True, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        return batch_loss