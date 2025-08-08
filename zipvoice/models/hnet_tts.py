# Copyright    2024    Xiaomi Corp.        (authors:  Wei Kang
#                                                     Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from zipvoice.models.config.hnet import HNetConfig
from zipvoice.models.modules.hnet import HNet, Isotropic
from zipvoice.models.modules.utils import NJT
from zipvoice.utils.common import (
    condition_time_mask,
    make_pad_mask,
)


class HNetTTS(nn.Module):
    def __init__(self, c: HNetConfig):
        super().__init__()
        self.c, v, d = c, c.vocab_size, c.d_model[0]
        self.embeddings = nn.Embedding(v, d)
        self.backbone = HNet(c, stage_idx=0)
        self.mel_input_linear = nn.Linear(100, d)
        self.output_linear = nn.Linear(d, 100)

    def forward(self, iids: Tensor, mels: Tensor, mels_lens: Tensor):
        assert iids.is_nested and iids.ndim == 2
        x = self.embeddings(iids)
        mels_input = self.mel_input_linear(mels)

        # concat NJTs
        unbond = [x.unbind(), mels_input.unbind()]
        concated_unbond = [
            torch.cat((x_i, m_i[:-1, :]), dim=0) for x_i, m_i in zip(*unbond)
        ]
        x = NJT(concated_unbond)

        x, *others = self.backbone(x)

        output = self.output_linear(x)

        # get pred mels
        unbond = [output.unbind(), mels.unbind()]
        for output_i, m_i in zip(*unbond):
            if m_i.size(0) >= output_i.size(0):
                print(f"output_i: {output_i.size()}, m_i: {m_i.size()}")
        pred_mels = [
            output_i[-m_i.size(0) :, :].contiguous() for output_i, m_i in zip(*unbond)
        ]

        mse_loss = torch.nn.functional.mse_loss(
            NJT(pred_mels).values(), mels.values(), reduction="sum"
        )
        mse_loss /= mels_lens.sum().item()

        return (mse_loss, *others)

    def flops(self, slen: int, msl: int):
        all_d, v = self.c.d_model, self.c.vocab_size

        # llm
        emb = 2 * slen * v * all_d[0]
        lmh = 2 * slen * v * all_d[0]
        # outer hnets: residual & routing mod have 3 dxd linears
        aux = 2 * slen * sum(3 * d * d for d in all_d[:-1])
        # isotropics
        iso = sum(
            slen * m.flops_per_token(msl)
            for m in self.modules()
            if isinstance(m, Isotropic)
        )

        # NOTE: we do not account for heavy scalar costs (including dechunk layer) here
        return emb + lmh + aux + iso

    @staticmethod
    def random_data(msl: int, *, s_min=256, s_max=1024):
        # samples = [
        #     torch.randint(256,(l,), dtype=torch.int, device='cuda')
        #     for l in torch.randint(s_min,s_max,(bsz,))
        # ]
        import random

        samples, total = [], 0
        while True:
            i = random.randint(s_min, s_max)
            t = torch.randint(256, (i,), dtype=torch.int, device="cuda")
            if i + total > msl:
                iids = NJT([s[:-1] for s in samples])
                lbls = NJT([s[1:] for s in samples]).long()
                yield (iids, lbls)
                samples, total = [t], i
            else:
                samples.append(t)
                total += i


class ZipVoice(nn.Module):
    """The ZipVoice model."""

    def __init__(
        self,
    ):
        """
        Initialize the model with specified configuration parameters.

        Args:
            fm_decoder_downsampling_factor: List of downsampling factors for each layer
                in the flow-matching decoder.
            fm_decoder_num_layers: List of the number of layers for each block in the
                flow-matching decoder.
            fm_decoder_cnn_module_kernel: List of kernel sizes for CNN modules in the
                flow-matching decoder.
            fm_decoder_feedforward_dim: Dimension of the feedforward network in the
                flow-matching decoder.
            fm_decoder_num_heads: Number of attention heads in the flow-matching
                decoder.
            fm_decoder_dim: Hidden dimension of the flow-matching decoder.
            text_encoder_num_layers: Number of layers in the text encoder.
            text_encoder_feedforward_dim: Dimension of the feedforward network in the
                text encoder.
            text_encoder_cnn_module_kernel: Kernel size for the CNN module in the
                text encoder.
            text_encoder_num_heads: Number of attention heads in the text encoder.
            text_encoder_dim: Hidden dimension of the text encoder.
            time_embed_dim: Dimension of the time embedding.
            text_embed_dim: Dimension of the text embedding.
            query_head_dim: Dimension of the query attention head.
            value_head_dim: Dimension of the value attention head.
            pos_head_dim: Dimension of the position attention head.
            pos_dim: Dimension of the positional encoding.
            feat_dim: Dimension of the acoustic features.
            vocab_size: Size of the vocabulary.
            pad_id: ID used for padding tokens.
        """
        super().__init__()

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        condition_drop_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass of the model for training.
        Args:
            tokens: a list of list of token ids.
            features: the acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: the length of each acoustic feature sequence, shape (batch,).
            noise: the intitial noise, with the shape (batch, seq_len, feat_dim).
            t: the time step, with the shape (batch, 1, 1).
            condition_drop_ratio: the ratio of dropped text condition.
        Returns:
            fm_loss: the flow-matching loss.
        """

        (
            text_condition,
            padding_mask,
        ) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        if condition_drop_ratio > 0.0:
            drop_mask = (
                torch.rand(text_condition.size(0), 1, 1).to(text_condition.device)
                > condition_drop_ratio
            )
            text_condition = text_condition * drop_mask

        xt = features * t + noise * (1 - t)
        ut = features - noise  # (B, T, F)

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        return fm_loss

    def sample(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features: torch.Tensor,
        prompt_features_lens: torch.Tensor,
        features_lens: Optional[torch.Tensor] = None,
        speed: float = 1.0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate acoustic features, given text tokens, prompts feature
            and prompt transcription's text tokens.
        Args:
            tokens: a list of list of text tokens.
            prompt_tokens: a list of list of prompt tokens.
            prompt_features: the prompt feature with the shape
                (batch_size, seq_len, feat_dim).
            prompt_features_lens: the length of each prompt feature,
                with the shape (batch_size,).
            features_lens: the length of the predicted eature, with the
                shape (batch_size,). It is used only when duration is "real".
            duration: "real" or "predict". If "real", the predicted
                feature length is given by features_lens.
            num_step: the number of steps to use in the ODE solver.
            guidance_scale: the guidance scale for classifier-free guidance.
        """

        assert duration in ["real", "predict"]

        if duration == "predict":
            (
                text_condition,
                padding_mask,
            ) = self.forward_text_inference_ratio_duration(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
            )
        else:
            assert features_lens is not None
            text_condition, padding_mask = self.forward_text_inference_gt_duration(
                tokens=tokens,
                features_lens=features_lens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
            )
        batch_size, num_frames, _ = text_condition.shape

        speech_condition = torch.nn.functional.pad(
            prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
        )  # (B, T, F)

        # False means speech condition positions.
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        x0 = torch.randn(
            batch_size,
            num_frames,
            prompt_features.size(-1),
            device=text_condition.device,
        )

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )
        x1_wo_prompt_lens = (~padding_mask).sum(-1) - prompt_features_lens
        x1_prompt = torch.zeros(
            x1.size(0), prompt_features_lens.max(), x1.size(2), device=x1.device
        )
        x1_wo_prompt = torch.zeros(
            x1.size(0), x1_wo_prompt_lens.max(), x1.size(2), device=x1.device
        )
        for i in range(x1.size(0)):
            x1_wo_prompt[i, : x1_wo_prompt_lens[i], :] = x1[
                i,
                prompt_features_lens[i] : prompt_features_lens[i]
                + x1_wo_prompt_lens[i],
            ]
            x1_prompt[i, : prompt_features_lens[i], :] = x1[
                i, : prompt_features_lens[i]
            ]

        return x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_features_lens

    def sample_intermediate(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        speech_condition_mask: torch.Tensor,
        t_start: float,
        t_end: float,
        num_step: int = 1,
        guidance_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate acoustic features in intermediate timesteps.
        Args:
            tokens: List of list of token ids.
            features: The acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: The length of each acoustic feature sequence,
                with the shape (batch,).
            noise: The initial noise, with the shape (batch, seq_len, feat_dim).
            speech_condition_mask: The mask for speech condition, True means
                non-condition positions, with the shape (batch, seq_len).
            t_start: The start timestep.
            t_end: The end timestep.
            num_step: The number of steps for sampling.
            guidance_scale: The scale for classifier-free guidance inference,
                with the shape (batch, 1, 1).
        """
        (
            text_condition,
            padding_mask,
        ) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        x_t_end = self.solver.sample(
            x=noise,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_start=t_start,
            t_end=t_end,
        )
        x_t_end_lens = (~padding_mask).sum(-1)
        return x_t_end, x_t_end_lens
