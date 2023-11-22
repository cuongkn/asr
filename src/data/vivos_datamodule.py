from typing import Optional
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.vivos_dataset import VivosDataset
from src.models.utils.utils import TextProcess

class VivosDataModule(LightningDataModule):
    def __init__(self, root: str, batch_size: int, num_workers: int, pin_memory: bool):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
        self.text_process = TextProcess(lang="vi")

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            data_train = VivosDataset(root=self.hparams.root, subset="train")
            data_test = VivosDataset(root=self.hparams.root, subset="test")

            train_length = len(data_train)
            test_length = len(data_test)

            self.data_train, self.data_val = random_split(dataset=data_train, lengths=[train_length-test_length, test_length], generator=torch.Generator().manual_seed(42))
            self.data_test = data_test

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self._collate_fn,
            shuffle=True,
            # persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self._collate_fn,
            shuffle=False,
            # persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self._collate_fn,
            shuffle=False,
            # persistent_workers=True
        )
    
    def tokenize(self, s):
        s = s.lower()
        s = self.text_process.tokenize(s)
        return s

    def _collate_fn(self, batch):
        """
        Take feature and input, transform and then padding it
        """

        specs = [i[0] for i in batch]
        input_lengths = torch.IntTensor([i.size(0) for i in specs])
        trans = [i[1] for i in batch]

        bs = len(specs)

        # batch, time, feature
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)

        trans = [self.text_process.text2int(self.tokenize(s)) for s in trans]
        target_lengths = torch.IntTensor([s.size(0) for s in trans])
        trans = torch.nn.utils.rnn.pad_sequence(trans, batch_first=True).to(
            dtype=torch.int
        )

        # concat sos and eos to transcript
        sos_id = torch.IntTensor([[self.text_process.sos_id]]).repeat(bs, 1)
        eos_id = torch.IntTensor([[self.text_process.eos_id]]).repeat(bs, 1)
        trans = torch.cat((sos_id, trans, eos_id), dim=1).to(dtype=torch.int)

        return specs, input_lengths, trans, target_lengths
    
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

        print("number of batches: ", len(val_loader))

        batch = next(iter(val_loader))

        print(batch[0])
        print(batch[1])
        print(batch[2])
        print(batch[3])


        # print(f'type of batch: {type(batch)}') # list
        # print(f'len of batch: {len(batch)}') # 2 
        # print(f'type of each element in batch: {type(batch[0])}') # Tensor
        # print(f'shape of the first element in batch: {batch[0].shape}') # [16, 1, 64000] = [batch_size, channel, n_samples]
        # print(f'shape of the second element in batch: {batch[1].shape}') # [16] 
        # print(f'value of the second element in each batch: {batch[1][0]}') # 16000 = sampling rate

    main()

