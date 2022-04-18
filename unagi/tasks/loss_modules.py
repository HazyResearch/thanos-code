import torch
from einops import rearrange


def mask_loss(
    loss_fns,
    aug_type,
    n_views,
    module_names,
    task_name,
    intermediate_output_dict,
    Y,
):
    # TODO: MODIFY THIS TO SUPPORT NEW FLOW
    total_loss = 0
    pre_encoding_embs = intermediate_output_dict["pre_encoder"][0]
    decoder_ouputs = intermediate_output_dict["decoder"][0]
    for field_name, field_emb in pre_encoding_embs.items():
        total_loss += loss_fns["masked_loss"](decoder_ouputs[field_name], field_emb)
    return total_loss


def ce_loss(
    loss_fns,
    aug_type,
    n_views,
    module_names,
    task_name,
    intermediate_output_dict,
    Y,
):
    total_loss = 0
    if aug_type is not None and aug_type != "raw":
        Y = intermediate_output_dict[f"{aug_type}_augmentation"][-1]
    #     Y = rearrange(Y, "b v d -> (b v) d", v=n_views)
    for mname in module_names:
        output = intermediate_output_dict[mname][0]
        # output = rearrange(output, "(b v) d -> b v d", v=n_views)
        total_loss += loss_fns[task_name](output, Y)

    return total_loss


def sce_loss(
    loss_fns,
    aug_type,
    n_views,
    module_names,
    task_name,
    intermediate_output_dict,
    Y,
):
    if aug_type is not None and aug_type != "raw":
        Y = intermediate_output_dict[f"{aug_type}_augmentation"][-1]
    if len(Y.size()) == 1:
        label = intermediate_output_dict[module_names][0].new_zeros(
            intermediate_output_dict[module_names][0].size()
        )
        label.scatter_(1, Y.view(Y.size()[0], 1), 1.0)
    else:
        label = Y

    return loss_fns[task_name](intermediate_output_dict[module_names][0], label)


def contrastive_loss(
    loss_fns,
    aug_type,
    n_views,  # number of views passed to contrastive loss function
    module_names,
    task_name,
    intermediate_output_dict,
    Y,
):
    total_loss = 0
    if aug_type is not None and aug_type != "raw":
        Y = intermediate_output_dict[f"{aug_type}_augmentation"][-1]
    # tgts = Y.unsqueeze(1).repeat(1, n_views)

    tgts = rearrange(Y, "(b v) ... -> b v ...", v=n_views)

    assert n_views == len(module_names), "Please properly select views"

    field_embs = [
        intermediate_output_dict[embed_layer_name][0]
        for embed_layer_name in module_names
    ]

    embs = torch.stack(field_embs, dim=1)
    total_loss += loss_fns[task_name](embs, tgts)
    """for embed_layer_name in module_names:

        field_emb = intermediate_output_dict[embed_layer_name][0]

        embs = rearrange(field_emb, "(b v) d -> b v d", b=n_views)
        total_loss += loss_fns[task_name](embs, tgts)"""
    return total_loss


def contrastive_loss_clip(
    loss_fns,
    aug_type,
    n_views,
    module_names,
    task_name,
    intermediate_output_dict,
    Y,
):
    total_loss = 0
    if aug_type is not None and aug_type != "raw":
        Y = intermediate_output_dict[f"{aug_type}_augmentation"][-1]
    tgts = rearrange(Y, "(b v) ... -> b v ...", v=n_views)

    assert n_views == len(module_names), "Please properly select views"

    field_embs = [
        intermediate_output_dict[embed_layer_name][0]
        for embed_layer_name in module_names
    ]

    embs = torch.stack(field_embs, dim=1)
    total_loss += loss_fns[task_name](embs, tgts)
    """tgts = Y.unsqueeze(1).repeat(1, n_views)
    for embed_layer_name in module_names:
        field_emb = intermediate_output_dict[embed_layer_name][0]
        # if len(field_emb.shape) == 3: field_emb = field_emb.squeeze(1)
        embs = rearrange(field_emb, "(b v) d -> b v d", v=n_views)
        total_loss += loss_fns[task_name](embs, tgts)"""

    return total_loss
