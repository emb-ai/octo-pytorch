import torch
import numpy as np
from .train_utils_pt import _np2pt
from octo.utils.train_utils import process_text

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""
    # TODO redo
    def __init__(
        self,
        rlds_dataset,
        text_processor,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._text_processor = text_processor
        self._is_train = train

    
    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            del sample["dataset_name"]
            sample["task"]["language_instruction"] = np.array([sample["task"]["language_instruction"]])
            # sample["task"]["pad_mask_dict"]['language_instruction'] = np.array(sample["task"]["pad_mask_dict"]['language_instruction'])
            sample = process_text(sample, self._text_processor)
            
            # remove extra dim
            sample["task"]["language_instruction"]['input_ids'] = sample["task"]["language_instruction"]['input_ids'][0]
            sample["task"]["language_instruction"]['attention_mask'] = sample["task"]["language_instruction"]['attention_mask'][0]
            
            # del sample["dataset_name"]
            sample = _np2pt_batch(sample)
            yield sample

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)


def _np2pt_batch(data, device=None):
    if isinstance(data, dict):
        return {key: _np2pt_batch(val, device) for key, val in data.items()}
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 3 and data.dtype == np.uint8:
            data = data.transpose((2, 0, 1)) #HWC -> CHW
        elif len(data.shape) == 4 and data.dtype == np.uint8:
            data = data.transpose((0, 3, 1, 2)) #THWC -> TCHW
    t = torch.tensor(data, device=device)
    return t