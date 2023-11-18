from typing import Optional
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.vivos import VivosDataset

class VivosDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, pin_memory: bool, trim_silence: bool, normalize_transcripts: bool):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = Optional[Dataset] = None
        self.data_val = Optional[Dataset] = None
        self.data_test = Optional[Dataset] = None

    def setup(self, stage: Optional[str]):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = VivosDataset(data_dir=self.hparams.data_dir, type="train")
            self.data_test = VivosDataset(data_dir=self.hparams.data_dir, type="train")

            train_length = len(self.data_train)
            test_length = len(self.data_test)

            self.data_train, self.data_val = random_split(dataset=self.data_train, lengths=[train_length-test_length, test_length], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_sampler=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_sampler=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_sampler=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

if __name__ == "__main__":
    import rootutils
    from omegaconf import DictConfig
    import hydra

    path = rootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")
    print("root", path, config_path)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="vivos.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        datamodule.setup()

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()

        print("number of batches: ", len(train_loader))

        batch = next(iter(train_loader))
        print(f'type of batch: {type(batch)}') 
        print(f'len of batch: {len(batch)}')  
        print(f'type of each element in batch: {type(batch[0])}') 
        print(f'shape of the first element in batch: {batch[0].shape}') 
        print(f'shape of the second element in batch: {batch[1].shape}') 
        print(f'value of the second element in each batch: {batch[1][0]}')

    main()
