from dataclasses import dataclass
from typing import Dict, Union

import torchmetrics as tm

import unagi.trainer.callbacks as C
import unagi.trainer.metrics as M


@dataclass
class UnagiTask:
    name: str
    task_weight: Union[int, float]
    task_flow: Dict
    losses: Dict
    metrics: Dict
    torchmetrics: Dict
    callbacks: Dict
    # weight: float

    def __post_init__(self):
        # use once hydra-fied
        """self.metric_names = [m.module for m in self.metrics]
        self.torchmetric_names = [m.module for m in self.torchmetrics]"""
        self._tracked_torchmetrics = {}
        self.all_callbacks = {}
        for callback_name, callback in self.callbacks.items():
            if callback["module"] in C.output_callback_fns:
                callback["task_name"] = self.name
                callback["name"] = callback_name
                self.all_callbacks[callback_name] = C.output_callback_fns[
                    callback["module"]
                ](**callback)

    def _init_torchmetrics(self, prefix):
        """
        Instantiate torchmetrics.
        """
        self._tracked_torchmetrics[prefix] = {}
        for name, torchmetric in self.torchmetrics.items():
            # TODO: .to('cuda') is a hack to make it work on GPU, generalize
            self._tracked_torchmetrics[prefix][torchmetric["module"]] = getattr(
                tm, name
            )(
                **{
                    **{
                        k: v
                        for k, v in torchmetric
                        if k not in ["node", "module", "inputs"]
                    },
                    "compute_on_step": False,
                }
            ).to(
                "cuda"
            )

        # TODO: deal with num_classes
        # for name in self.torchmetric_names:
        #     if name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1']:
        #         self._tracked_torchmetrics[prefix][name] = getattr(tm, name)(
        #               average='macro', num_classes=self.dataset.d_output,
        #               compute_on_step=False).to('cuda')
        #     elif '@' in name:
        #         k = int(name.split('@')[1])
        #         mname = name.split('@')[0]
        #         self._tracked_torchmetrics[prefix][name] = getattr(tm, mname)(
        #           average='macro', num_classes=self.dataset.d_output,
        #                       compute_on_step=False, top_k=k).to('cuda')
        #     else:
        #         self._tracked_torchmetrics[prefix][name] = getattr(tm, name)(
        #           compute_on_step=False).to('cuda')

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics
        for prefix in all_prefixes:
            for torchmetric in self.torchmetrics:
                try:
                    self._tracked_torchmetrics[prefix][torchmetric["module"]].reset()
                except KeyError:  # metrics don't exist yet
                    pass

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {
            torchmetric["module"]: self._tracked_torchmetrics[prefix][name].compute()
            for name, torchmetric in self.torchmetrics.items()
        }

    def update_torchmetrics(self, x, y, prefix):
        """
        Update torchmetrics with new x, y.
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)

        for torchmetric in self.torchmetrics:
            self._tracked_torchmetrics[prefix][torchmetric["module"]].update(x, y)

    def get_metric(self, metric, *args):
        """
        Metrics are just functions
        output metrics are a function of output and target
        # loss metrics are a function of loss (e.g. perplexity)
        """
        # TODO: handle the loss metrics (perplexity, bpb)
        """if metric.module in M.output_metric_fns:
            return M.output_metric_fns[metric.module](*args)"""
        if metric["module"] in M.output_metric_fns:
            return M.output_metric_fns[metric["module"]](*args)
        return None

    def get_callback(self, callback, **kwargs):
        """
        Callbacks are just arbitrary functions
        Can be used for, e.g., custom logging
        """

        if callback["module"] in C.output_callback_fns:
            return C.output_callback_fns[callback["module"]](**kwargs)
        return None
