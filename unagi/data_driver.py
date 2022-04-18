import logging
from types import SimpleNamespace

from unagi.datasets import DATASET_CLASSES

logger = logging.getLogger(__name__)


def get_data(dataflow_config):
    """
    Builds datasets and dataloaders from config file.

    # Inputs
    :param config: (dict) dictionary representation of experiment config file

    # Returns
    :return: SimpleNamespace containing datasets and dataloaders (train, val, test).
        A dataloader is built for every task / dataset type split.
    """

    datasets = list(dataflow_config.dataset.keys())
    assert len(datasets) == 1, "Only one dataset is supported."
    dataset_name = datasets[0]

    dataset = DATASET_CLASSES[dataset_name](
        data_dir=dataflow_config.data_dir,
        x_transforms=dataflow_config.x,
        y_transforms=dataflow_config.y,
        transform_pool=dataflow_config.transforms,
        **dataflow_config.dataset[dataset_name],
    )
    train_dataloaders = dataset.train_dataloader(
        batch_size=dataflow_config.batch_size,
        num_workers=dataflow_config.num_workers,
        drop_last=True,
    )
    val_dataloaders = dataset.val_dataloader(
        batch_size=dataflow_config.batch_size,
        num_workers=dataflow_config.num_workers,
        drop_last=True,
    )
    test_dataloaders = dataset.test_dataloader(
        batch_size=dataflow_config.batch_size,
        num_workers=dataflow_config.num_workers,
        drop_last=True,
    )

    return SimpleNamespace(
        dataset=dataset,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        test_dataloaders=test_dataloaders,
    )
