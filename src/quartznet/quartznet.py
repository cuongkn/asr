import argparse
import json
import copy
from omegaconf import open_dict, DictConfig
import torch.nn as nn
from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import WandbLogger
# import wandb
import nemo.collections.asr as nemo_asr
from src.data.components.libri import LibriDataset

from src.data.components.vivos import *
import logging

logging.basicConfig(filename="./logs/quartznet.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)

class QuartzNet():
    def __init__(self, charset_path: str, model_name: str, freeze_encoder:bool = False) -> None:
        super().__init__()
        self.model_name = model_name

        charset_list = self.get_charset(charset_path=charset_path)

        char_model = nemo_asr.models.ASRModel.from_pretrained(model_name)

        if freeze_encoder:
            char_model.encoder.freeze()
            char_model.encoder.apply(self.enable_bn_se)
        else:
            char_model.encoder.unfreeze()

        self.char_model = char_model


    def get_charset(self, charset_path):
        try:
            with open(charset_path, 'rb') as json_file:
                character_list = json.load(json_file)
                return character_list
        except FileNotFoundError:
            print(f"The file {charset_path} does not exist.")
            return [] 
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {charset_path}: {e}")
            return []
        
    def enable_bn_se(self, model):
        if type(model) == nn.BatchNorm1d:
            model.train()
            for param in model.parameters():
                param.requires_grad_(True)

        if 'SqueezeExcite' in type(model).__name__:
            model.train()
            for param in model.parameters():
                param.requires_grad_(True)

if __name__ == "__main__":
    _logger.info("Start ............")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=str, help="'vivos' or 'libri'")
    args = parser.parse_args()

    if args.task == "vivos":
        model = QuartzNet(charset_path = "data/vivos/charset.json", model_name = "stt_en_quartznet15x5")
        quartznet = model.char_model

        cfg = copy.deepcopy(quartznet.cfg)

        train_dataset = VivosDataset(type="train")
        test_dataset = VivosDataset(type="test")

        quartznet.change_vocabulary(new_vocabulary=list(set(get_charset(train_dataset.data).keys())))

        with open_dict(cfg):
            cfg.train_ds.manifest_filepath = train_dataset.manifest_path
            cfg.train_ds.labels = list(set(get_charset(train_dataset.data).keys()))
            cfg.train_ds.batch_size = 8
            cfg.train_ds.num_workers = 0
            cfg.train_ds.pin_memory = True
            cfg.train_ds.trim_silence = True

            cfg.validation_ds.manifest_filepath = test_dataset.manifest_path
            cfg.validation_ds.labels = list(set(get_charset(test_dataset.data).keys()))
            cfg.validation_ds.batch_size = 8
            cfg.validation_ds.num_workers = 0
            cfg.validation_ds.pin_memory = True
            cfg.validation_ds.trim_silence = True
    
        quartznet.setup_training_data(cfg.train_ds)
        quartznet.setup_multiple_validation_data(cfg.validation_ds)

        with open_dict(quartznet.cfg.optim):
            quartznet.cfg.optim.lr = 5e-5
            quartznet.cfg.optim.betas = [0.95, 0.5]  # from paper
            quartznet.cfg.optim.weight_decay = 0.001  # Original weight decay
            quartznet.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
            quartznet.cfg.optim.sched.warmup_ratio = None
            quartznet.cfg.optim.sched.min_lr = 0.0

        quartznet.spec_augmentation = quartznet.from_config_dict(quartznet.cfg.spec_augment)

        #@title Metric
        use_cer = True #@param ["False", "True"] {type:"raw"}
        log_prediction = True #@param ["False", "True"] {type:"raw"}
        
        EPOCHS = 100  # 100 epochs would provide better results, but would take an hour to train

        trainer = Trainer(
            devices=1,
            accelerator='gpu',
            max_epochs=EPOCHS,
            accumulate_grad_batches=1,
            enable_checkpointing=False,
            logger=False,
            log_every_n_steps=1,
            check_val_every_n_epoch=10
        )

        # Setup model with the trainer
        quartznet.set_trainer(trainer)
        quartznet.cfg = quartznet._cfg

        trainer.fit(quartznet)
        quartznet.save_to('quartznet.nemo')
        # wandb.finish()
    elif args.task == "libri":
        dev_clean = LibriDataset(option="dev-clean")
        dev_other = LibriDataset(option="dev-other")
        test_clean = LibriDataset(option="test-clean")
        test_other = LibriDataset(option="test-other")

        quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
        # --- Config Information ---#
        try:
            from ruamel.yaml import YAML
        except ModuleNotFoundError:
            from ruamel_yaml import YAML
        config_path = './src/quartznet/config.yaml'

        yaml = YAML(typ='safe')
        with open(config_path) as f:
            params = yaml.load(f)
        trainer = Trainer(devices=1, accelerator='gpu', max_epochs=100)   
                
        params['model']['train_ds']['manifest_filepath'] = dev_clean.manifest_path
        params['model']['validation_ds']['manifest_filepath'] = test_clean.manifest_path
        first_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)
        trainer.fit(first_asr_model)
    _logger.info("End..............")
