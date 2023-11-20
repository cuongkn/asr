import json
from omegaconf import DictConfig
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr

class QuartzNetLitModule(nemo_asr.models.ASRModel.from_pretrained("stt_en_quartznet15x5", map_location='cpu')):
    def __init__(self, cfg: DictConfig, charset_path: str, model_name: str, freeze_encoder:bool = False, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        charset_list = self.get_charset(charset_path=charset_path)
        self.change_vocabulary(new_vocabulary=charset_list)
        self.cfg.labels = charset_list

        if freeze_encoder:
            self.encoder.freeze()
            self.encoder.apply(self.enable_bn_se)
        else:
            self.encoder.unfreeze()

    def get_charset(self, charset_path):
        try:
            with open(charset_path, 'r', encoding='utf-8') as json_file:
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
