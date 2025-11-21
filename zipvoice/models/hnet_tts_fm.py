from collections import defaultdict
from typing import Optional, Tuple

import torch
from hnet_impl import HNetConfig
from hnet_impl.conceptual import BlockBoundaryMixin
from hnet_impl.modeling_hnet import HNet
from torch import Tensor, nested, nn
from torch.nn import functional as F

from zipvoice.models.modules.hnet import FlowMatchingHead
from zipvoice.models.modules.solver import HNetEulerSolver
from zipvoice.models.modules.utils import NJT


def condition_time_mask(
    speech_flat: Tensor, speech_cu: Tensor, mask_percent: Tuple[float, float]
):
    speech_flat_batch_idx = (
        torch.arange(speech_flat.shape[0], device=speech_flat.device).unsqueeze(1)
        > speech_cu[1:] - 1
    ).sum(dim=1)

    batch_size = speech_cu.shape[0] - 1
    non_mask_size_per_batch = (
        (
            1
            - torch.zeros(
                batch_size, dtype=speech_flat.dtype, device=speech_flat.device
            ).uniform_(*mask_percent)
        )
        * speech_cu.diff()  # lens per batch
    ).to(torch.int64)
    mask_starts_per_batch = speech_cu[:-1] + non_mask_size_per_batch

    mask = (
        torch.arange(speech_flat.shape[0], device=speech_flat.device)
        >= mask_starts_per_batch[speech_flat_batch_idx]
    )

    return mask


def concat_flat_with_bos(
    flat1: Tensor, flat1_cu: Tensor, flat2: Tensor, flat2_cu: Tensor, bos: Tensor
):
    """
    Concatenate two flat tensors per batch, and inster bos emb in middle .
    """
    bos_cu = torch.arange(flat1_cu.shape[0], device=flat1_cu.device)
    concated_cu = flat1_cu + flat2_cu + bos_cu
    concated_msl = concated_cu.diff().max().item()

    flat1_batch_idx = (
        torch.arange(flat1.shape[0], device=flat1.device).unsqueeze(1)
        > flat1_cu[1:] - 1
    ).sum(dim=1)

    flat1_dest_idx = (
        torch.arange(flat1.shape[0], device=flat1.device)
        + flat2_cu[flat1_batch_idx]
        + bos_cu[flat1_batch_idx]
    )

    bos_batch_idx = torch.arange(bos_cu.shape[0] - 1, device=bos_cu.device)
    bos_dest_idx = bos_batch_idx + flat1_cu[bos_batch_idx + 1] + flat2_cu[bos_batch_idx]

    flat2_batch_idx = (
        torch.arange(flat2.shape[0], device=flat2.device).unsqueeze(1)
        > flat2_cu[1:] - 1
    ).sum(dim=1)
    flat2_dest_idx = (
        torch.arange(flat2.shape[0], device=flat2.device)
        + flat1_cu[flat2_batch_idx + 1]
        + bos_cu[flat2_batch_idx + 1]
    )

    inverse_perm = torch.cat([flat1_dest_idx, bos_dest_idx, flat2_dest_idx], dim=0)
    perm = inverse_perm.argsort()

    perm_expanded = perm.unsqueeze(1).expand(-1, flat1.shape[1])

    concated_flat = torch.cat(
        [flat1, bos.expand(flat1_cu.shape[0] - 1, -1), flat2], dim=0
    )
    concated_flat = torch.gather(concated_flat, 0, perm_expanded)

    inverse_perm = perm.argsort()

    return concated_flat, concated_cu, concated_msl, inverse_perm


def get_input_mel(mels: Tensor, mels_cu: Tensor):
    input_mel_mask = torch.ones(mels.shape[0], dtype=torch.bool, device=mels.device)
    input_mel_mask[mels_cu[1:] - 1] = False
    input_mel_cu = mels_cu - torch.arange(mels_cu.shape[0], device=mels_cu.device)
    return mels[input_mel_mask], input_mel_cu


def get_target_mel(mels: Tensor, mels_cu: Tensor):
    target_mel_mask = torch.ones(mels.shape[0], dtype=torch.bool, device=mels.device)
    target_mel_mask[mels_cu[:-1]] = False
    target_mel_cu = mels_cu - torch.arange(mels_cu.shape[0], device=mels_cu.device)
    return mels[target_mel_mask], target_mel_cu


class MelPreNet(nn.Module):
    def __init__(self, mel_d, hidden_d, output_d):
        super().__init__()
        self.input_linear = nn.Linear(mel_d, hidden_d)
        self.relu = nn.ReLU()
        self.hidden_linear = nn.Linear(hidden_d, hidden_d)
        self.output_linear = nn.Linear(hidden_d, output_d)
        self.dropout = nn.Dropout(0.5)

    def forward(self, mels: Tensor, dropout_mask: Tensor = None):
        x = self.input_linear(mels)
        x = self.relu(x)
        if dropout_mask is not None:
            # Apply dropout to entire tensor, then select based on mask (non-inplace)
            x_full_dropped = self.dropout(x)
            x = torch.where(dropout_mask.unsqueeze(-1), x_full_dropped, x)
        else:
            x = F.dropout(x, p=0.5, training=True)
        x = self.hidden_linear(x)
        x = self.relu(x)
        if dropout_mask is not None:
            # Apply dropout to entire tensor, then select based on mask (non-inplace)
            x_full_dropped = self.dropout(x)
            x = torch.where(dropout_mask.unsqueeze(-1), x_full_dropped, x)
        else:
            x = F.dropout(x, p=0.5, training=True)
        return self.output_linear(x)


class HNetTTS(BlockBoundaryMixin, nn.Module):
    def __init__(self, c: HNetConfig):
        super().__init__()
        self.c, v, d = c, c.vocab_size, c.d_model[0]
        self.embeddings = nn.Embedding(v, d)
        self.backbone = HNet(c, stage_idx=0)

        # maybe don't need this
        self.mel_prenet = MelPreNet(100, 512, d)
        self.lm_head = nn.Linear(d, 100)
        self.fm_head = FlowMatchingHead(100, d)

        self.mel_bos = nn.Parameter(torch.randn(1, d))
        nn.init.normal_(self.mel_bos, mean=0, std=d**-0.5)
        self.stop_head = nn.Linear(d, 1)
        self.dropout = nn.Dropout(0.1)

        self.solver = HNetEulerSolver(self, func_name="forward_fm_decoder")

    def forward_fm_decoder(
        self,
        t: Tensor,
        xt: Tensor,
        condition: Tensor,
        guidance_scale: Optional[Tensor] = None,
    ) -> Tensor:
        # placeholder guidance_scale for future implementation
        return self.fm_head(xt, t, condition)

    def forward(self, iids: Tensor, mels: Tensor, noise: Tensor, t: Tensor):
        assert iids.is_nested and iids.ndim == 2
        text_condition = self.embeddings(iids.values())
        text_condition_cu = iids.offsets()

        input_mels, input_mel_cu = get_input_mel(mels.values(), mels.offsets())
        target_mels, target_mel_cu = get_target_mel(mels.values(), mels.offsets())

        speech_condition_mask = condition_time_mask(
            speech_flat=input_mels,
            speech_cu=input_mel_cu,
            mask_percent=(0.7, 1.0),
        )  # speech condition is False, non-speech condition is True

        input_mels = self.mel_prenet(input_mels)

        x_flat, cu_s, msl, inverse_perm = concat_flat_with_bos(
            text_condition, text_condition_cu, input_mels, input_mel_cu, self.mel_bos
        )  # inverse_perm sort: text, bos, input_mels

        x_flat, extra = self.backbone(x_flat, cu_s, msl)

        pred_features = x_flat[inverse_perm[-target_mels.shape[0] :]]

        stop_logits = self.stop_head(pred_features)
        stop_labels = torch.zeros_like(stop_logits)
        stop_labels[target_mel_cu[1:] - 1] = 1
        loss_bce = F.binary_cross_entropy_with_logits(
            stop_logits,
            stop_labels,
            pos_weight=torch.Tensor([100]).to(stop_logits.device),
        )

        t, _ = get_target_mel(t, mels.offsets())
        noise, _ = get_target_mel(noise, mels.offsets())

        # need condition_drop_ratio?
        xt = target_mels * t + noise * (1 - t)
        ut = target_mels - noise
        vt = self.forward_fm_decoder(t, xt, pred_features)
        loss_fm = torch.mean((vt - ut) ** 2)

        lm_logits = self.lm_head(pred_features)
        cond_loss_l1 = F.l1_loss(lm_logits, target_mels)
        cond_loss_l2 = F.mse_loss(lm_logits, target_mels)
        loss_cond = cond_loss_l1 + cond_loss_l2

        # loss_bce may need *0.01
        loss = loss_fm + loss_bce * 0.01 + loss_cond * 0.1

        return (
            loss,
            loss_fm,
            loss_bce,
            loss_cond,
            extra,
        )

    @torch.inference_mode()
    @torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    def sample(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features: Tensor,
        stop_threshold: float = 0.5,
        max_length: int = 10 * 94,
        min_length: int = 0,
        num_step: int = 10,
        guidance_scale: float = 0.0,
        t_shift: float = 1.0,
    ):
        concated_tokens = NJT(
            [
                torch.cat((prompt_token, token), dim=0)
                for prompt_token, token in zip(prompt_tokens.unbind(), tokens.unbind())
            ]
        )

        text_condition = self.embeddings(concated_tokens)

        mels_inputs = self.mel_prenet(prompt_features)
        mels_inputs = NJT(
            [
                torch.cat((t_i, self.mel_bos, m_i), dim=0)
                for t_i, m_i in zip(text_condition.unbind(), mels_inputs.unbind())
            ]
        )

        batch_size = tokens.shape[0]
        results = [None] * batch_size
        index_map = list(range(batch_size))
        steps = 0

        mel_outputs = None

        while index_map and steps < max_length:
            print(steps)

            cu_s, msl = mels_inputs.offsets(), mels_inputs._get_max_seqlen()
            x_flat = mels_inputs.values()
            x_flat, _ = self.backbone(x_flat, cu_s, msl)

            x = nested.nested_tensor_from_jagged(
                values=x_flat, offsets=cu_s, max_seqlen=msl
            )
            x_last = NJT([x_i[-1].unsqueeze(0) for x_i in x.unbind()])

            stop_logits = self.stop_head(x_last)
            stop_probs = F.sigmoid(stop_logits)

            x0 = torch.randn(
                x_last.values().shape[0],
                100,
                device=x_last.device,
            )

            x1 = self.solver.sample(
                x=x0,
                condition=x_last.values(),
                num_step=num_step,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
            ).unsqueeze(1)

            if mel_outputs is None:
                mel_outputs = NJT(x1.unbind())
            else:
                mel_outputs = NJT(
                    [
                        torch.cat((m_i, x_i), dim=0)
                        for m_i, x_i in zip(mel_outputs.unbind(), x1)
                    ]
                )

            print(stop_probs[0])
            will_stop = []
            for pos, s_i in enumerate(stop_probs.unbind()):
                will_stop.append(
                    bool(s_i > stop_threshold)
                    and mel_outputs[pos].shape[0] >= min_length
                )

            for pos, stop_flag in enumerate(will_stop):
                if stop_flag:
                    results[index_map[pos]] = mel_outputs[pos]

            survivors = [
                pos for pos, stop_flag in enumerate(will_stop) if not stop_flag
            ]
            if not survivors:
                break

            x1 = self.mel_prenet(x1)

            mels_inputs = NJT(
                [torch.cat((mels_inputs[pos], x1[pos]), dim=0) for pos in survivors]
            )

            index_map = [index_map[pos] for pos in survivors]
            steps += 1

        if any(r is None for r in results):
            for pos, original_idx in enumerate(index_map):
                results[original_idx] = mel_outputs[pos]

        return NJT(results)

    def split_params_by_hierachy(self) -> list[list[nn.Parameter]]:
        # for each param, count the number of times ".main_network" appears in it.
        d = defaultdict(list)
        for n, p in self.named_parameters():
            d[n.count("main_network")].append(p)
        # special-case innermost hnet which has redundant .main_network
        max_depth = max(d.keys())
        assert 1 == len(d[max_depth - 1]), (
            f"expected single .pad_dimension at {max_depth - 1}"
        )
        d[max_depth - 1] += d.pop(max_depth)

        return [d[k] for k in range(len(d))]
