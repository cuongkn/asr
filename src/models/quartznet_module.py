import torch
from lightning import LightningModule

from src.models.components.quartznet import QuartzNet


class QuartzNetLitModule(LightningModule):
    def __init__(
        self,
        quartznet: QuartzNet
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['quartznet'])

        # model
        self.quartznet = quartznet

    def forward(self, x: torch.Tensor):
        return self.soundstream(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_sisnr_best.reset()

    def model_step(self, batch: Any, optimizer_idx: int):
        orig_x = batch[0] # [batch_size, channel, n_samples]
        sr = batch[1][0] # batch[1].shape = [batch_size]

        recon_x = self.forward(orig_x)

        if optimizer_idx == 0:
            total_loss_g = self.criterion_g(orig_x, orig_x, recon_x, sr, self.zero)
            return total_loss_g, orig_x, recon_x

        if optimizer_idx == 1:
            total_loss_d = self.criterion_d(orig_x, recon_x)
            return total_loss_d, orig_x, recon_x

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        loss, orig_x, recon_x = self.model_step(batch, optimizer_idx)

        if optimizer_idx == 0:
            self.train_loss_g(loss)
            self.log("train/loss_g", self.train_loss_g,
                     on_step=False, on_epoch=True, prog_bar=True)

        if optimizer_idx == 1:
            self.train_loss_d(loss)
            self.log("train/loss_d", self.train_loss_d,
                     on_step=False, on_epoch=True, prog_bar=True)

        self.train_sisnr(recon_x, orig_x)
        self.log("train/sisnr", self.train_sisnr,
                 on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": recon_x, "targets": orig_x}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss_g, orig_x, recon_x = self.model_step(batch, optimizer_idx=0)
        self.val_loss_g(loss_g)
        self.log("val/loss_g", self.val_loss_g,
                 on_step=False, on_epoch=True, prog_bar=True)

        loss_d, _, _ = self.model_step(batch, optimizer_idx=1)
        self.val_loss_d(loss_d)
        self.log("val/loss_d", self.val_loss_d,
                 on_step=False, on_epoch=True, prog_bar=True)

        self.val_sisnr(recon_x, orig_x)
        self.log("val/sisnr", self.val_sisnr, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss_g": loss_g, 'loss_d': loss_d}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_sisnr.compute()  # get current val acc
        self.val_sisnr_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/sisnr_best", self.val_sisnr_best.compute(),
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss_g, orig_x, recon_x = self.model_step(batch, optimizer_idx=0)
        self.test_loss_g(loss_g)
        self.log("test/loss_g", self.test_loss_g,
                 on_step=False, on_epoch=True, prog_bar=True)

        loss_d, _, _ = self.model_step(batch, optimizer_idx=1)
        self.test_loss_d(loss_d)
        self.log("test/loss_d", self.test_loss_d,
                 on_step=False, on_epoch=True, prog_bar=True)

        sisnr = self.test_sisnr(recon_x, orig_x)
        self.log("test/sisnr", sisnr.mean(), on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss_g": loss_g, 'loss_d': loss_d}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        _, _, recon_x = self.model_step(batch)
        return recon_x 

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_g = self.hparams.optimizer_g(
            params=self.soundstream.parameters())
        optimizer_d = self.hparams.optimizer_d(params=list(
            self.criterion_d.ms_discriminator.parameters()) + list(self.criterion_d.stft_discriminator.parameters()))

        return [optimizer_g, optimizer_d], []


if __name__ == "__main__":
    # read config file from configs/model/dlib_resnet.yaml
    import rootutils
    from omegaconf import DictConfig, OmegaConf
    import hydra

    # find paths
    path = rootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs")
    output_path = path / "outputs"
    print(f"path: {path}")
    print(f'config path: {config_path}')
    print(f'output path: {output_path}')

    def test_soundstream(cfg):
        soundstream = hydra.utils.instantiate(cfg.model.soundstream)
        output = soundstream(torch.randn(1, 1, 16000))
        print("soundstream output shape", output.shape)

    def test_module(cfg):
        module = hydra.utils.instantiate(cfg.model)
        output = module(torch.randn(1, 1, 16000))
        print("module output shape", output.shape)

    def test_optimizer(cfg):
        optimizer_d = hydra.utils.instantiate(cfg.model.optimizer_d)
        print(f'optimizer_d: {optimizer_d}')

    def test_criterion_g(cfg):
        criterion_g = hydra.utils.instantiate(cfg.model.criterion_g)
        print(f'criterion_g: {criterion_g}')

    def test_training_step_random_input(cfg):
        module = hydra.utils.instantiate(cfg.model)
        torch.manual_seed(7749)
        xs = torch.randn(4, 1, 16000)
        sr = torch.tensor([16000, 16000, 16000, 16000])
        output = module.training_step([xs, sr], 0, 0)
        print("training_step output", output)

    def test_training_step_real_input(cfg):
        module = hydra.utils.instantiate(cfg.model)
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()

        loader = datamodule.test_dataloader()
        print("number of batches: ", len(loader))

        batch = next(iter(loader))
        output = module.training_step(batch, 0, 0)
        print("training_step output", output)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(f'config: \n {OmegaConf.to_yaml(cfg.model, resolve=True)}')
        # test_soundstream(cfg)
        # test_module(cfg)
        # test_optimizer(cfg)
        # test_criterion_g(cfg)

        # test_training_step_random_input(cfg)
        # test_training_step_real_input(cfg)

        model = hydra.utils.instantiate(cfg.model)
        data = hydra.utils.instantiate(cfg.data)
        import lightning as pl
        trainer = pl.Trainer(max_epochs=1, fast_dev_run=False)
        trainer.fit(model=model, datamodule=data)

    main()
