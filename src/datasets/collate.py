import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    if len(dataset_items) == 0:
        return {}
    dataset_keys = dataset_items[0].keys()
    batched_items: dict[torch.Tensor] = {}
    for key in dataset_keys:
        if key == "spectrogram" or key == "text_encoded":
            batched_items[f"{key}_length"] = torch.Tensor(
                [item[key].shape[-1] for item in dataset_items]
            ).to(torch.int32)
            batched_items[key] = pad_sequence(
                [item[key].squeeze(dim=0).permute((1, 0)) for item in dataset_items],
                batch_first=True,
            )
        else:
            batched_items[key] = [item[key] for item in dataset_items]
    batched_items["spectrogram"] = batched_items["spectrogram"].permute((0, 2, 1))
    batched_items["text_encoded"] = batched_items["text_encoded"].squeeze(-1)
    return batched_items
