from typing import List

import hydra
import pytorch_lightning as pl
import torch
from torch.backends import cudnn as cudnn

from unagi.task import create_tasks, instantiate_modules
from unagi.trainer.model import UnagiModel
from unagi.trainer.scheduler import SCHEDULERS


def process_inputs(inputs_list, output_dict, Y_dict, task_flow):
    inputs = []
    for input_node, key in inputs_list:
        if input_node == "_input_":
            inputs.append(output_dict[input_node][key])
        elif not input_node == "_output_":
            inputs.append(output_dict[(input_node, task_flow[input_node]["module"])])
        else:
            inputs.append(Y_dict[key])
    return inputs


class UnagiModule(pl.LightningModule):
    def _set_module_shapes(self, config):
        """
        Automatically set the shapes of modules.
        - Embedding modules: set the input shape to the dataset input shape
        - Encoder / Decoder modules: set the input shape to the output shape
          of the previous module
        - Decoder modules: set the output shape to the dataset output shape
          (using the loss it's linked to)
        """
        # start w/embeddings because their input is directly dataset
        #
        for _, task_config in self.config.tasks.items():

            decoder_to_dataset_key = {}
            for _, loss_config in task_config.losses.items():
                input_modules = []
                # For all input modules to the loss, we keep track of what dataset
                # output those modules map to
                for input_module, input_index in loss_config.inputs:
                    if input_module == "_output_":
                        # "Label" input from the dataset to the loss
                        dataset_key = input_index
                    else:
                        # Input from another module (typically decoder) to the loss
                        input_modules.append(input_module)

                for module in input_modules:
                    decoder_to_dataset_key[module] = dataset_key

            d_output = {}
            for step, step_config in task_config.task_flow.items():
                if step_config.module in config.model.embeddings:
                    # Set the d_input of the embedding module
                    """assert (
                        len(step_config.inputs) == 1
                    ), "Assume only one input to embedding."""
                    assert len(step_config.inputs[0]) == 2
                    dataset_key = step_config.inputs[0][1]
                    # TODO: this will need to be updated if there are multiple datasets
                    # used for training
                    # (Currently, assumes that all datasets have the same
                    # dataset.input_shapes)
                    if dataset_key in self.dataset.input_shapes:
                        config.model.embeddings[
                            step_config.module
                        ].d_input = self.dataset.input_shapes[dataset_key][0]

                    # Carry over the d_model of the embedding to the encoder
                    d_output[step] = config.model.embeddings[step_config.module].d_model
                elif step_config.module in config.model.encoders:
                    # Check the input of the encoder and set the d_model of the
                    # encoder using that
                    input_module = step_config.inputs[0][0]
                    assert (
                        input_module in d_output
                    ), f"Encoder {step_config.module} has no recognizable input."
                    config.model.encoders[step_config.module].d_model = d_output[
                        input_module
                    ]

                    # TODO: Carry over the d_output of the encoder instead to the next
                    # encoder or decoder
                    # (Currently assumes that encoders have
                    #                   d_input = d_output = d_model)
                    d_output[step] = config.model.encoders[step_config.module].d_model
                elif step_config.module in config.model.decoders:
                    # Check the input of the decoder and set the d_input of the decoder
                    # using that
                    # TODO (ASN): THIS IS A HACK -- some decoders take inputs directly
                    # from task_preprocessing layer
                    input_module = step_config.inputs[0][0]
                    dataset_key = step_config.inputs[0][1]
                    # assert input_module in d_output, f"Decoder {step_config.module}
                    # has no recognizable input."
                    # breakpoint()
                    if config.model.decoders[step_config.module].d_input is None:
                        config.model.decoders[step_config.module].d_input = (
                            d_output[input_module]
                            if input_module in d_output
                            else self.dataset.input_shapes[dataset_key][-1]
                        )

                    # Set the d_output of the decoder by looking at the dataset
                    # output key it uses for supervision (through a loss fn)
                    # (Currently, assumes that all datasets have the same
                    #             dataset.output_shapes)
                    if step in decoder_to_dataset_key:
                        dataset_key = decoder_to_dataset_key[step]
                        config.model.decoders[
                            step_config.module
                        ].d_output = self.dataset.output_shapes[dataset_key][0]
                    else:
                        config.model.decoders[
                            step_config.module
                        ].d_output = config.model.decoders[step_config.module].d_input
                        d_output[step] = config.model.decoders[
                            step_config.module
                        ].d_input
        # Make sure all embeddings and decoders are being used
        for emb, emb_config in config.model.embeddings.items():
            assert (
                emb_config.d_input is not None
            ), f"Embedding {emb} has no input shape and is unused."

        """for dec, dec_config in config.model.decoders.items():
            assert (
                dec_config.d_output is not None
            ), f"Decoder {dec} has no output shape and is unused." """

        return config

    def _set_d_input(self):
        # Grab the random state from torch
        rng = torch.get_rng_state()

        if isinstance(self.train_dataloaders, torch.utils.data.DataLoader):
            dl = self.train_dataloaders
        else:
            dl = self.train_dataloaders[0]

        for batch in dl:
            X_dict, _ = batch
            # shape is (B, V, ...) -- discard (B, V)
            X_shapes = {
                k: tuple(v.shape[2:])
                for k, v in X_dict["inputs"].items()
                if k != "is_train"
            }
            break

        self.dataset.input_shapes = X_shapes

        # Set the random state back to torch
        torch.set_rng_state(rng)

    def __init__(
        self,
        config,
        dataset,
        train_dataloaders: List[torch.utils.data.DataLoader],
        val_dataloaders: List[torch.utils.data.DataLoader],
        test_dataloaders: List[torch.utils.data.DataLoader],
    ):
        super(UnagiModule, self).__init__()
        self.config = config
        self.dataset = dataset
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.test_dataloaders = test_dataloaders
        self.val_loader_names = list(self.val_dataloaders.keys()) + list(
            self.test_dataloaders.keys()
        )
        self.test_loader_names = list(self.val_dataloaders.keys()) + list(
            self.test_dataloaders.keys()
        )
        self.test_loader_names = [f"final/{name}" for name in self.test_loader_names]

        # Set the d_input
        self._set_d_input()

        self.task_scheduler = SCHEDULERS[self.config.learner.task_scheduler]()

        # Update the internal d_input / d_model / d_output shapes of all modules
        self.config = self._set_module_shapes(self.config)

        # Construct all the torch modules
        self.module_dict, self.loss_dict = instantiate_modules(
            self.config.model.preprocessors,
            self.config.model.embeddings,
            self.config.model.encoders,
            self.config.model.decoders,
            self.config.model.losses,
        )
        self.tasks = create_tasks(self.config.tasks)
        self.model = UnagiModel(tasks=self.tasks, module_dict=self.module_dict)
        self.checkpoint_scheduler = self.config.learner.checkpoint_scheduler

        # Set cudnn benchmark
        cudnn.benchmark = True

    def setup(self, stage=None):
        pass

    def forward(self, x_dict):
        return self.model(x_dict)

    def configure_callbacks(self):
        checkpoint = pl.callbacks.ModelCheckpoint(**self.checkpoint_scheduler)
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        return [checkpoint, lr_monitor]

    def _shared_step(self, batch, batch_idx, prefix="train"):

        # TODO: make batch return less things in the schedulers
        (
            X_dict,
            Y_dict,
        ) = batch

        # Forward pass through the model(s)
        output_dict = self.forward(X_dict)

        # Compute loss
        loss_per_task = {name: 0.0 for name, task in self.tasks.items()}
        for name, task in self.tasks.items():
            # Run all the losses for this task
            for loss in task.losses.values():
                # Pull out relevant attributes of `loss`
                loss_module_name = loss["module"]  # UID of the loss module
                # loss_node = loss["node"]  # UID of the loss node

                # Grab the actual loss module
                loss_module = self.loss_dict[loss_module_name]

                # Gather all the inputs to the loss module
                loss_inputs = process_inputs(
                    loss["inputs"], output_dict, Y_dict, task.task_flow
                )
                # Calculate the loss, add it to the task loss
                loss_per_task[name] += loss_module(*loss_inputs) * loss["weight"]

        # Add up all the task losses weighted by task_weight
        loss = sum(
            [
                loss_per_task[name] * task.task_weight
                for name, task in self.tasks.items()
            ]
        )

        # Compute metrics
        metrics_per_task = {name: {} for name, task in self.tasks.items()}
        for task_name, task in self.tasks.items():
            # Run all the metrics for this task
            for name, metric in task.metrics.items():
                # Gather all the inputs to the metric module
                metric_inputs = process_inputs(
                    metric["inputs"], output_dict, Y_dict, task.task_flow
                )

                # Calculate the metric, add to the task metric dict
                metrics_per_task[task_name][name] = task.get_metric(
                    metric, *metric_inputs
                )

        # Compute torchmetrics
        for task_name, task in self.tasks.items():
            # Run all the metrics for this task
            for name, torchmetric in task.torchmetrics.items():
                # Gather all the inputs to the metric module
                _ = process_inputs(
                    torchmetric["inputs"], output_dict, Y_dict, task.task_flow
                )

                # TODO: figure out what to do here (call .update on torchmetrics)
                # Calculate the metric, add to the task metric dict
                # metrics_per_task[task.name][metric['node']] =
                #               task.get_metric(metric, *metric_inputs)

        # Run callbacks
        for task_name, task in self.tasks.items():
            for name, callback in task.callbacks.items():
                callback_batch = {}
                if self.current_epoch % callback["log_every_n_epochs"] == 0:
                    callback_batch["inputs"] = [
                        x.clone().detach().cpu()
                        if isinstance(x, torch.Tensor)
                        else torch.Tensor(x)
                        for x in process_inputs(
                            callback["inputs"],
                            output_dict,
                            Y_dict,
                            task.task_flow,
                        )
                    ]
                    callback_batch["trainer"] = self
                    callback_batch["batch_idx"] = batch_idx
                    callback_batch["prefix"] = prefix
                    if prefix == "train":
                        task.all_callbacks[name].on_train_batch_end(callback_batch)
                    elif prefix in self.test_loader_names:
                        task.all_callbacks[name].on_test_batch_end(callback_batch)
                    else:
                        task.all_callbacks[name].on_validation_batch_end(callback_batch)

        # Calculate all metrics and log
        metrics = {
            f"{name}_loss": loss_per_task[name] * task.task_weight
            for name, task in self.tasks.items()
        }
        metrics["loss"] = loss
        metrics.update(
            {
                f"{task}_{metric}": metric_val
                for task, task_metrics in metrics_per_task.items()
                for metric, metric_val in task_metrics.items()
            }
        )

        """task_accuracy = {}
        for task_name, label_name in task_to_label_dict.items():
            if (
                task_name in self.model.loss_funcs
                and self.model.loss_funcs[task_name] is not None
            ):
                preds = F.softmax(
                    output_dict["classifier"][0],
                    dim=1,
                )
                target = Y_dict[label_name]
                accuracy = torchmetrics.functional.accuracy(preds, target)
                task_accuracy[f"accuracy_{task_name}"] = accuracy"""

        # metrics.update(task_accuracy)
        # metrics["batch_size"] = len(batch[0])
        metrics = {f"{prefix}/{k}": v.detach() for k, v in metrics.items()}
        metrics["loss"] = loss
        self.log_dict(
            metrics,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        # return {"loss": loss, "metrics": metrics, "callback_values": callback_values}
        # return metrics
        return metrics

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, self.val_loader_names[dataloader_idx]
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, self.test_loader_names[dataloader_idx]
        )

    def training_epoch_end(self, outputs):
        for task_name, task in self.tasks.items():
            for name, callback in task.callbacks.items():
                if self.current_epoch % callback["log_every_n_epochs"] == 0:
                    task.all_callbacks[name].on_train_epoch_end()
        # return torch.stack([torch.stack(x).sum() for x in outputs]).mean()
        # return outputs

    def validation_epoch_end(self, outputs):
        for task_name, task in self.tasks.items():
            for name, callback in task.callbacks.items():
                if self.current_epoch % callback["log_every_n_epochs"] == 0:
                    task.all_callbacks[name].on_validation_epoch_end()
                    task.all_callbacks[name].on_test_epoch_end()
        # val_loss = torch.stack([torch.stack(x).sum() for x in outputs]).mean()
        # return val_loss
        # return outputs

    def test_epoch_end(self, outputs):
        for task_name, task in self.tasks.items():
            for name, callback in task.callbacks.items():
                if self.current_epoch % callback["log_every_n_epochs"] == 0:
                    task.all_callbacks[name].on_validation_epoch_end()
                    task.all_callbacks[name].on_test_epoch_end()
        # return torch.stack([torch.stack(x).sum() for x in outputs]).mean()
        # return outputs

    def _eval_dataloaders(self):
        val_loaders = (
            list(self.val_dataloaders.values())
            if isinstance(self.val_dataloaders, dict)
            else [self.val_dataloaders]
        )
        test_loaders = (
            list(self.test_dataloaders.values())
            if isinstance(self.test_dataloaders, dict)
            else [self.test_dataloaders]
        )
        return val_loaders + test_loaders

    def train_dataloader(self):
        return self.train_dataloaders

    def val_dataloader(self):
        return self._eval_dataloaders()
        # return self.task_scheduler.get_batches(self.val_dataloaders, self.model)

    def test_dataloader(self):
        return self._eval_dataloaders()
        # return self.task_scheduler.get_batches(self.test_dataloaders, self.model)

    def configure_optimizers(self):
        # Optimizer params
        optimizer_config = self.config.learner.optimizer
        scheduler_config = self.config.learner.scheduler
        modules_to_freeze = self.config.learner.modules_to_freeze

        all_parameters = []
        # All parameters in the model
        for param_name, param_values in self.named_parameters():
            # freeze_param = False
            for module_name in modules_to_freeze:
                if module_name in param_name:
                    # freeze_param = True
                    param_values.requires_grad = False
            # if not freeze_param:
            all_parameters.append(param_values)

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        assert optimizer_config._target_ is not None, "No optimizer target specified"

        # optimizer = hydra.utils.instantiate(optimizer_config, params=params)
        optimizer = hydra.utils.instantiate(
            optimizer_config,
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
        )

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in set(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": params, **hp})

        # Create a lr scheduler
        scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)

        scheduler = {
            "scheduler": scheduler,
            "interval": self.config.learner.interval,
            "monitor": self.config.learner.monitor,
            "name": self.config.learner.name,
        }

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(
                " | ".join(
                    [
                        f"Optimizer group {i}",
                        f"{len(g['params'])} tensors",
                    ]
                    + [f"{k} {v}" for k, v in group_hps.items()]
                )
            )

        return [optimizer], [scheduler]
