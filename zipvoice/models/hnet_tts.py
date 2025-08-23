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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from zipvoice.models.config.hnet import HNetConfig
from zipvoice.models.modules.hnet import HNet, Isotropic
from zipvoice.models.modules.utils import NJT


class HNetTTS(nn.Module):
    def __init__(self, c: HNetConfig):
        super().__init__()
        self.c, v, mel_d, text_d = c, c.vocab_size, c.d_model[0], c.d_model[-1]
        self.embeddings = nn.Embedding(v, text_d)
        self.backbone = HNet(c, stage_idx=0)

        # maybe don't need this
        self.mel_input_linear = nn.Linear(100, mel_d)
        self.mel_output_linear = nn.Linear(mel_d, 100)

        self.mel_bos = nn.Parameter(torch.randn(1, mel_d))
        self.stop_head = nn.Linear(mel_d, 1)
        self.init_weights()

    def forward(self, iids: Tensor, mels: Tensor):
        assert iids.is_nested and iids.ndim == 2
        text_condition = self.embeddings(iids)
        mels_input = self.mel_input_linear(mels)

        mels_input = NJT(
            [
                torch.cat((self.mel_bos, m_i[:-1, :]), dim=0)
                for m_i in mels_input.unbind()
            ]
        )

        x, *others = self.backbone(mels_input, text_condition)
        stop_logits = self.stop_head(x)
        stop_labels = torch.zeros_like(stop_logits.values())
        stop_labels[stop_logits.offsets()[1:] - 1] = 1
        stop_loss = F.binary_cross_entropy_with_logits(
            stop_logits.values(),
            stop_labels,
            pos_weight=torch.Tensor([100]).to(stop_logits.values().device),  # or 1000
        )

        # x = NJT([x_i[:-1] for x_i in x.unbind()])
        x = self.mel_output_linear(x)

        l1_loss = torch.nn.functional.l1_loss(x.values(), mels.values())
        mse_loss = torch.nn.functional.mse_loss(x.values(), mels.values())

        return (x, l1_loss + mse_loss, *others, stop_loss)

    def sample(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features: Tensor,
        mel_lens: Tensor,
    ):
        concated_tokens = [
            torch.cat((prompt_token, token), dim=0)
            for prompt_token, token in zip(prompt_tokens.unbind(), tokens.unbind())
        ]
        text_condition = self.embeddings(concated_tokens)
        mels_input = self.mel_input_linear(prompt_features)
        mels_input = NJT(
            [torch.cat((self.mel_bos, m_i), dim=0) for m_i in mels_input.unbind()]
        )
        for _ in range(mel_lens.max()):
            x, *others = self.backbone(mels_input, text_condition)
            x = self.mel_output_linear(x)
            mel_input = [
                torch.cat((m_i, x_i[-1]), dim=0)
                for x_i, m_i in zip(x.unbind(), mels_input.unbind())
            ]
            mels_input = NJT(mel_input)

        pred_mel = [
            m_i[:-mel_len] for m_i, mel_len in zip(mels_input.unbind(), mel_lens)
        ]
        return NJT(pred_mel)

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

    def init_weights(self, initializer_range: float = 0.02) -> None:
        """
        Initializes the weights of the model.
        """

        nn.init.normal_(self.mel_input_linear.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(
            self.mel_output_linear.weight, mean=0.0, std=initializer_range / 3
        )
        nn.init.zeros_(self.mel_output_linear.bias)
        # embeddings are initialized differently from linears
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        self.backbone._init_weights(initializer_range)
