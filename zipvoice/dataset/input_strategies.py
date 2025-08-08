from concurrent.futures import Executor
from typing import Dict, Optional, Tuple

import torch
from lhotse import CutSet
from lhotse.dataset.collation import (
    _read_features,
    collate_vectors,
)
from lhotse.dataset.input_strategies import BatchIO, _get_executor
from lhotse.utils import (
    supervision_to_frames,
)

from zipvoice.models.modules.utils import NJT


def collate_features(
    cuts: CutSet,
    executor: Optional[Executor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load features for all the cuts and return them as a batch in a torch tensor.
    The output shape is ``(batch, time, features)``.
    The cuts will be padded with silence if necessary.

    :param cuts: a :class:`CutSet` used to load the features.
    :param pad_direction: where to apply the padding (``right``, ``left``, or ``both``).
    :param executor: an instance of ThreadPoolExecutor or ProcessPoolExecutor; when provided,
        we will use it to read the features concurrently.
    :return: a tuple of tensors ``(features, features_lens)``.
    """
    assert all(cut.has_features for cut in cuts)
    features_lens = torch.tensor([cut.num_frames for cut in cuts], dtype=torch.int)
    features = [None] * len(cuts)
    if executor is None:
        for idx, cut in enumerate(cuts):
            features[idx] = _read_features(cut)
    else:
        for idx, example_features in enumerate(executor.map(_read_features, cuts)):
            features[idx] = example_features
    return NJT(features), features_lens


class PrecomputedFeaturesNJT(BatchIO):
    """
    :class:`InputStrategy` that reads pre-computed features, whose manifests
    are attached to cuts, from disk.

    It automatically pads the feature matrices so that every example has the same number
    of frames as the longest cut in a mini-batch.
    This is needed to put all examples into a single tensor.
    The padding value is a low log-energy, around log(1e-10).

    .. automethod:: __call__
    """

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads the pre-computed features from disk/other storage.
        The returned shape is ``(B, T, F) => (batch_size, num_frames, num_features)``.

        :return: a tensor with collated features, and a tensor of ``num_frames`` of each cut before padding."""
        return collate_features(
            cuts,
            executor=_get_executor(self.num_workers, executor_type=self._executor_type),
        )

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that specifies the start and end bounds for each supervision,
        as a 1-D int tensor, in terms of frames:

        .. code-block::

            {
                "sequence_idx": tensor(shape=(S,)),
                "start_frame": tensor(shape=(S,)),
                "num_frames": tensor(shape=(S,))
            }

        Where ``S`` is the total number of supervisions encountered in the :class:`CutSet`.
        Note that ``S`` might be different than the number of cuts (``B``).
        ``sequence_idx`` means the index of the corresponding feature matrix (or cut) in a batch.
        """
        start_frames, nums_frames = zip(
            *(
                supervision_to_frames(
                    sup, cut.frame_shift, cut.sampling_rate, max_frames=cut.num_frames
                )
                for cut in cuts
                for sup in cut.supervisions
            )
        )
        sequence_idx = [i for i, c in enumerate(cuts) for s in c.supervisions]
        return {
            "sequence_idx": torch.tensor(sequence_idx, dtype=torch.int32),
            "start_frame": torch.tensor(start_frames, dtype=torch.int32),
            "num_frames": torch.tensor(nums_frames, dtype=torch.int32),
        }

    def supervision_masks(
        self, cuts: CutSet, use_alignment_if_exists: Optional[str] = None
    ) -> torch.Tensor:
        """Returns the mask for supervised frames.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return collate_vectors(
            [
                cut.supervisions_feature_mask(
                    use_alignment_if_exists=use_alignment_if_exists
                )
                for cut in cuts
            ]
        )
