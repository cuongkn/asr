import json
import torch.nn as nn
import nemo.collections.asr as nemo_asr


class QuartzNet():
    def __init__(self, charset_path: str, model_name: str, freeze_encoder:bool = False) -> None:
        super().__init__()
        self.model_name = model_name

        charset_list = self.get_charset(charset_path=charset_path)

        char_model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location='cpu')
        char_model.change_vocabulary(new_vocabulary=list(charset_list))
        char_model.cfg.labels = charset_list

        if freeze_encoder:
            char_model.encoder.freeze()
            char_model.encoder.apply(self.enable_bn_se)
        else:
            char_model.encoder.unfreeze()

        self.char_model = char_model


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
    
