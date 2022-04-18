import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange
from sklearn.manifold import TSNE


class UnagiCallback:
    def __init__(self):
        self.train_batches = {}
        self.val_batches = {}
        self.test_batches = {}

    def on_train_batch_end(self, outputs, name):
        if name not in self.train_batches.keys():
            self.train_batches[name] = [outputs]
        else:
            self.train_batches[name].append(outputs)

    def on_validation_batch_end(self, outputs, name):
        if name not in self.val_batches.keys():
            self.val_batches[name] = [outputs]
        else:
            self.val_batches[name].append(outputs)

    def on_test_batch_end(self, outputs, name):
        if name not in self.test_batches.keys():
            self.test_batches[name] = [outputs]
        else:
            self.test_batches[name].append(outputs)

    def on_train_epoch_end(self):
        for split_name, batch_list in self.train_batches.items():
            self.train_batches[split_name] = []

    def on_validation_epoch_end(self):
        for split_name, batch_list in self.val_batches.items():
            self.val_batches[split_name] = []

    def on_test_epoch_end(self):
        for split_name, batch_list in self.test_batches.items():
            self.test_batches[split_name] = []


class LogImage(UnagiCallback):
    def __init__(
        self,
        name,
        logging_batch_idx,
        max_images,
        input_names,
        task_name,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.logging_batch_idx = logging_batch_idx
        self.max_images = max_images
        self.task_name = task_name
        self.input_names = input_names

    def _on_batch_end(self, output_batch):
        inputs = output_batch["inputs"]
        prefix = output_batch["prefix"]
        trainer = output_batch["trainer"]
        batch_idx = output_batch["batch_idx"]
        if batch_idx == self.logging_batch_idx:
            for input_name, inp in zip(self.input_names, inputs):
                imgs = [wandb.Image(img) for img in inp][: self.max_images]
                trainer.logger.experiment.log(
                    {
                        f"{prefix}/{self.task_name}_{self.name}_{input_name}": imgs,
                        "trainer/global_step": trainer.global_step,
                    }
                )

    def on_train_batch_end(self, output_batch):
        super().on_train_batch_end(output_batch, output_batch["prefix"])
        self._on_batch_end(output_batch)

    def on_validation_batch_end(self, output_batch):
        super().on_validation_batch_end(output_batch, output_batch["prefix"])
        self._on_batch_end(output_batch)

    def on_test_batch_end(self, output_batch):
        super().on_test_batch_end(output_batch, output_batch["prefix"])
        self._on_batch_end(output_batch)


class LogEmbedding(UnagiCallback):
    def __init__(
        self,
        name,
        logging_batch_idx,
        input_names,
        task_name,
        batch_size,
        eval_batch_size,
        class_names,
        plot_embeddings,
        plot_embeddings_stride,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.class_names = class_names
        self.logging_batch_idx = logging_batch_idx
        self.task_name = task_name
        self.input_names = input_names
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        # self.val_input_embeddings = [[] for _ in range(len(self.input_names))]
        # self.train_input_embeddings = [
        #     [] for _ in range(len(self.input_names))
        # ]
        # self.test_input_embeddings = [[] for _ in range(len(self.input_names))]
        self.val_input_embeddings = {}
        self.test_input_embeddings = {}
        self.train_input_embeddings = {}
        self.plot_embeddings = plot_embeddings
        self.plot_embeddings_stride = plot_embeddings_stride

    def _extract_views(self, batch_size, inp, view=0):
        n_views = math.ceil(inp.shape[0] / batch_size)
        embs = rearrange(inp, "(b v) ... -> b v ...", v=n_views)
        embs = embs[:, view, ...]
        return embs

    def _on_batch_end(self, output_batch, split):
        inputs = output_batch["inputs"]
        prefix = output_batch["prefix"]
        for i, inp in enumerate(inputs):
            if split == "train":
                if prefix not in self.train_input_embeddings:
                    self.train_input_embeddings[prefix] = [
                        [] for _ in range(len(self.input_names))
                    ]

                self.train_input_embeddings[prefix][i].append(
                    self._extract_views(self.batch_size, inp)
                )
            elif split == "test":
                if prefix not in self.test_input_embeddings:
                    self.test_input_embeddings[prefix] = [
                        [] for _ in range(len(self.input_names))
                    ]

                self.test_input_embeddings[prefix][i].append(
                    self._extract_views(self.eval_batch_size, inp)
                )
            else:
                if prefix not in self.val_input_embeddings:
                    self.val_input_embeddings[prefix] = [
                        [] for _ in range(len(self.input_names))
                    ]
                self.val_input_embeddings[prefix][i].append(
                    self._extract_views(self.eval_batch_size, inp)
                )

    def _plot_tsne(self, tsne_results, categories, s=1, figsize=(12, 8)):
        plt.figure(figsize=figsize)

        for indices, label in categories:
            plt.scatter(
                tsne_results[:, 0][indices],
                tsne_results[:, 1][indices],
                label=label,
                s=s,
            )

        plt.legend()
        return plt

    def on_train_batch_end(self, output_batch):
        super().on_train_batch_end(output_batch, output_batch["prefix"])
        self._on_batch_end(output_batch, "train")

    def on_validation_batch_end(self, output_batch):
        super().on_validation_batch_end(output_batch, output_batch["prefix"])
        self._on_batch_end(output_batch, "val")

    def on_test_batch_end(self, output_batch):
        super().on_test_batch_end(output_batch, output_batch["prefix"])
        self._on_batch_end(output_batch, "test")

    def _on_epoch_end(self, split):
        # inputs = []
        if split == "train":
            input_embeddings = self.train_input_embeddings
            batches = self.train_batches
            self.train_input_embeddings = {}
        elif split == "test":
            input_embeddings = self.test_input_embeddings
            batches = self.test_batches
            self.test_input_embeddings = {}
        else:
            input_embeddings = self.val_input_embeddings
            batches = self.val_batches
            self.val_input_embeddings = {}

        for prefix, inps in input_embeddings.items():
            inputs = []
            for i in inps:  # iterates through all inputs
                inputs.append(torch.cat(i, dim=0))
            input_embeddings[prefix] = inputs

        for prefix, inputs in input_embeddings.items():
            print(f"logging {prefix} embeddings to wandb...may be slow")
            trainer = batches[prefix][-1]["trainer"]
            for input_name, inp in zip(self.input_names, inputs):
                if len(inp.shape) == 1:
                    inp = inp.unsqueeze(-1)

                inp = inp.numpy()

                columns = [str(x) for x in range(inp.shape[1])]
                inp = [inp[x] for x in range(inp.shape[0])]
                inp = wandb.Table(columns=columns, data=inp)
                trainer.logger.experiment.log(
                    {
                        f"{prefix}/{self.task_name}_{self.name}_{input_name}_"
                        f"epoch{trainer.current_epoch}": inp
                    }
                )

            if self.plot_embeddings:
                print(f"generating {prefix} TSNE plot...may be slow")
                labels, embs = None, None
                for input_name, inp in zip(self.input_names, inputs):
                    if input_name == "labels":
                        labels = inp.numpy()
                    if input_name not in [
                        "labels",
                        "sample_uid",
                    ]:  # this is hacky
                        embs = inp

                tsne = TSNE(n_iter=300)
                if len(embs) > self.plot_embeddings_stride * 100:
                    embs = embs[:: self.plot_embeddings_stride]
                    labels = labels[:: self.plot_embeddings_stride]
                tsne_results = tsne.fit_transform(embs)

                categories = [
                    (np.argwhere(labels == i).flatten(), self.class_names[i])
                    for i in range(len(self.class_names))
                ]
                plt_mpl = self._plot_tsne(tsne_results, categories, figsize=(6, 6))
                plt_wb = wandb.Image(plt_mpl)
                wandb.log(
                    {
                        f"{prefix}/{self.task_name}_{self.name}_{input_name}"
                        f"_tsne": plt_wb,
                        "trainer/global_step": trainer.global_step,
                    }
                )

    def on_train_epoch_end(self):
        self._on_epoch_end("train")
        super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        self._on_epoch_end("test")
        super().on_test_epoch_end()


"""
def log_image(
    # prefix,
    task_name,
    trainer,
    name,
    input_names,
    # trainer,
    logging_batch_idx,
    # inputs,
    # batch_idx,
    max_images,
    # batch_args,
    **kwargs,
):
    batch_args = kwargs[f"batch_{logging_batch_idx}"]
    inputs = batch_args["inputs"]
    prefix = batch_args["prefix"]
    # name = batch_args["name"]
    # trainer = batch_args["trainer"]
    # task_name = batch_args["task_name"]
    for input_name, inp in zip(input_names, inputs):
        imgs = [wandb.Image(img) for img in inp][:max_images]
        trainer.logger.experiment.log(
            {
                f"{prefix}/{task_name}_{name}_{input_name}": imgs,
                "trainer/global_step": trainer.global_step,
            }
        )


def log_embedding(
    # prefix,
    task_name,
    name,
    input_names,
    trainer,
    # logging_batch_idx,
    # inputs,
    # batch_idx,
    # max_images,
    # batch_args,
    **kwargs,
):
    # batch_args = kwargs[f"batch_{logging_batch_idx}"]
    batch_inputs = [[] for _ in range(len(input_names))]
    for key, item in kwargs.items():
        if "batch_" in key and key != "logging_batch_idx":
            for i, inp in enumerate(item["inputs"]):
                batch_inputs[i].append(inp)
            prefix = item["prefix"]
            # name = item["name"]
            # trainer = item["trainer"]
            # task_name = item["task_name"]
    inputs = []
    for inps in batch_inputs:
        inputs.append(torch.cat(inps, dim=0))

    for input_name, inp in zip(input_names, inputs):
        # imgs = [wandb.Image(img) for img in inp][:max_images]
        columns = [str(x) for x in range(inp.shape[1])]
        inp = [inp[x] for x in range(inp.shape[0])]
        inp = wandb.Table(columns=columns, data=inp)
        trainer.logger.experiment.log(
            {
                f"{prefix}/{task_name}_{name}_{input_name}_"
                f"epoch{trainer.current_epoch}": inp
            }
        )
"""

output_callback_fns = {"log_image": LogImage, "log_embedding": LogEmbedding}
