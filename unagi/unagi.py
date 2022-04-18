import logging

import pytorch_lightning as pl

from unagi.data_driver import get_data
from unagi.trainer.trainer import UnagiModule

logger = logging.getLogger(__name__)


def main(config):
    # Create dataloaders
    data = get_data(config.dataflow)

    # set seed
    if (
        "random_seed" in config.dataflow.keys()
        and config.dataflow.random_seed is not None
    ):
        pl.seed_everything(seed=config.dataflow.random_seed)

    if config.model.train:
        unagi_module = UnagiModule(
            config=config,
            dataset=data.dataset,
            train_dataloaders=data.train_dataloaders,
            val_dataloaders=data.val_dataloaders,
            test_dataloaders=data.test_dataloaders,
        )

        if "wandb" in config.keys():
            logger = pl.loggers.WandbLogger(**{**config.wandb, "config": config})

        # Create trainer
        trainer = pl.Trainer(
            **{
                **config.trainer,
                "logger": logger,
                "callbacks": unagi_module.configure_callbacks(),
            }
        )
        trainer.fit(unagi_module)
        trainer.test(ckpt_path="best")
