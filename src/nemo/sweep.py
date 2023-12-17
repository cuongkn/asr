from pytorch_lightning import Trainer
from omegaconf import DictConfig
from ruamel.yaml import YAML
import wandb
from pytorch_lightning.loggers import WandbLogger
import nemo.collections.asr as nemo_asr

from src.data.components.libri import LibriDataset


sweep_config = {
    "method": "random",   # Random search
    "metric": {           # We want to minimize `val_loss`
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        'lr': {
            # log uniform distribution between exp(min) and exp(max)
            "distribution": "log_uniform",
            "min": -11.513,   
            "max": -6.9078
        },
        'epochs': {
            "distribution": "int_uniform",
            "min": 30,
            "max": 50
        },
        # 'dropout': {
        #     "distribution": "uniform",
        #     "min": 0,  
        #     "max": 0.25     
        # }
        
    }
}


def sweep_iteration():
    dev_clean = LibriDataset(option="dev-clean")
    dev_other = LibriDataset(option="dev-other")
    test_clean = LibriDataset(option="test-clean")
    test_other = LibriDataset(option="test-other")
    # load config
    config_path = './src/quartznet/config.yaml'
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)    
        
    # set up W&B logger
    # wandb.init(config=sweep_config)    # required to have access to `wandb.config`
    wandb.init()
    wandb_logger = WandbLogger(log_model='all')  # log final model

    for k,v in params.items(): 
        wandb_logger.experiment.config[k]=v 

    # setup data
    params['model']['train_ds']['manifest_filepath'] = dev_clean.manifest_path
    params['model']['validation_ds']['manifest_filepath'] = test_clean.manifest_path
    
    # setup sweep param
    params['model']['optim']['lr'] = wandb.config.lr
    # params['model']['encoder']['jasper'][-1]['dropout'] = wandb.config.dropout
    
    trainer = Trainer(
        devices=1, 
        accelerator='gpu', 
        enable_checkpointing=True, 
        check_val_every_n_epoch=5, 
        max_epochs=wandb.config.epochs, 
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        logger=wandb_logger)
    
    # setup model - note how we refer to sweep parameters with wandb.config
    model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

    # train
    trainer.fit(model)
    return
