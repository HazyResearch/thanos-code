import logging
from typing import Any, Dict, List, Optional, Union

from torch import nn
from torch.nn import ModuleDict

from unagi.task import UnagiTask

logger = logging.getLogger(__name__)


class UnagiModel(nn.Module):
    """A class to build multi-task model.

    Args:
      name: Name of the model, defaults to None.
      tasks: A task or a list of tasks.
    """

    def __init__(
        self,
        tasks: Optional[Union[UnagiTask, List[UnagiTask]]] = None,
        module_dict: ModuleDict = None,
    ) -> None:

        super().__init__()

        # Initiate the model attributes
        self.module_dict = module_dict
        self.task_flows = {name: task.task_flow for name, task in tasks.items()}
        self.task_names = list(self.task_flows.keys())

    def _get_data_from_output_dict(
        self,
        output_dict: Dict[str, Any],
        index: Any,
        task_flow: Dict[str, Any],
    ) -> Any:
        """
        Get output_dict output based on output_idx.

        For the valid index, please check the definition of Action.
        """

        if index[0] != "_input_":
            index_aug = [item for item in index]
            index_aug[0] = (index[0], task_flow[index[0]]["module"])
            index = index_aug

        # Handle any output_dict's item and index is str or int
        if isinstance(index, (str, int)):
            if index in output_dict:
                return output_dict[index]
            else:
                raise ValueError(f"Action {index}'s output is not in the output_dict.")
        # Handle output_dict's item is a list, tuple or dict, and index is (X, Y)
        elif isinstance(output_dict[index[0]], (list, tuple)):
            if isinstance(index[1], int):
                return output_dict[index[0]][index[1]]
            else:
                raise ValueError(
                    f"Action {index[0]} output has {type(output_dict[index[0]])} type, "
                    f"while index has {type(index[1])} not int."
                )
        elif isinstance(output_dict[index[0]], dict):
            if index[1] in output_dict[index[0]]:
                return output_dict[index[0]][index[1]]
            else:
                raise ValueError(
                    f"Action {index[0]}'s output doesn't have attribute {index[1]}."
                )
        # Handle output_dict's item is neither a list or dict, and index is (X, Y)
        elif int(index[1]) == 0:
            return output_dict[index[0]]

        raise ValueError(f"Cannot parse action index {index}.")

    def forward(
        self,
        X_dict: Dict[str, Any],
        # task_names: List[str],
    ) -> Dict[str, Any]:
        """Forward based on input and task flow.

        Note:
          We assume that all shared modules from all tasks are based on the
          same input.

        Args:
          X_dict: The input data
        #   task_names: The task names that needs to forward.

        Returns:
          The output of all forwarded modules
        """
        output_dict = dict(_input_=X_dict)

        # Call forward for each task
        for task_name in self.task_names:
            for node, action in self.task_flows[task_name].items():
                if (node, action["module"]) not in output_dict:
                    if action["inputs"]:
                        input = [
                            self._get_data_from_output_dict(
                                output_dict,
                                _input,
                                self.task_flows[task_name],
                            )
                            for _input in action["inputs"]
                        ]
                        # TODO: this might be important for the multi-gpu case
                        # try:
                        #     action_module_device = (
                        #         self.module_device[action.module]
                        #         if action.module in self.module_device
                        #         else default_device
                        #     )
                        #     input = move_to_device(
                        #         [
                        # self._get_data_from_output_dict(output_dict, _input)
                        #             for _input in action.inputs
                        #         ],
                        #         action_module_device,
                        #     )
                        # except Exception:
                        #     raise ValueError(f"Unrecognized action {action}.")
                        output = self.module_dict[action["module"]].forward(*input)
                    else:
                        # TODO: Handle multiple device with not inputs case
                        output = self.module_dict[action["module"]].forward(output_dict)
                    output_dict[(node, action["module"])] = output

        return output_dict

    # def __repr__(self) -> str:
    #     """Represent the model as a string."""
    #     cls_name = type(self).__name__
    #     return f"{cls_name}"#(name={self.name})"

    # def add_tasks(self, tasks: Union[UnagiTask, List[UnagiTask]]) -> None:
    #     """
    #     Build the MTL network using all tasks.

    #     Args:
    #       tasks: A task or a list of tasks.
    #     """
    #     if not isinstance(tasks, Iterable):
    #         tasks = [tasks]
    #     for task in tasks:
    #         self.add_task(task)

    # def add_task(self, task: UnagiTask) -> None:
    #     """Add a single task into MTL network.

    #     Args:
    #       task: A task to add.
    #     """
    #     if not isinstance(task, UnagiTask):
    #         raise ValueError(f"Unrecognized task type {task}.")

    #     # TODO: move this check taht there are no duplicate tasks somewhere else
    #     # if task.name in self.task_names:
    #     #     raise ValueError(
    #     #         f"Found duplicate task {task.name}, different task should use "
    #     #         f"different task name."
    #     #     )

    #     # # Combine module_dict from all tasks
    #     # for key in task.module_dict.keys():
    #     #     if key in self.module_dict.keys():
    #     #         task.module_dict[key] = self.module_dict[key]
    #     #     else:
    #     #         self.module_dict[key] = task.module_dict[key]

    #     # Collect task name
    #     self.task_names.add(task.name)

    #     # Collect task flow
    #     self.task_flows[task.name] = task.task_flow
