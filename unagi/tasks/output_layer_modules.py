from torch.nn import functional as F


def multiclass_classification(module_name, immediate_output_dict):
    return F.softmax(
        immediate_output_dict[module_name][len(immediate_output_dict[module_name]) - 1],
        dim=1,
    )


def multilabel_classification(module_name, immediate_output_dict):
    return F.sigmoid(
        immediate_output_dict[module_name][len(immediate_output_dict[module_name]) - 1]
    )
