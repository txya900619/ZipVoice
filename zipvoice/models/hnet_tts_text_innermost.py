from collections import defaultdict

import torch
from hnet_impl import HNetConfig
from hnet_impl.conceptual import BlockBoundaryMixin
from hnet_impl.modeling_hnet import HNet
from torch import Tensor, nested, nn
from torch.nn import functional as F

from zipvoice.models.modules.utils import NJT


class MelPreNet(nn.Module):
    def __init__(self, mel_d, hidden_d, output_d):
        super().__init__()
        self.input_linear = nn.Linear(mel_d, hidden_d)
        self.relu = nn.ReLU()
        self.dropout_rate = 0.5
        self.hidden_linear = nn.Linear(hidden_d, hidden_d)
        self.output_linear = nn.Linear(hidden_d, output_d)

    def forward(self, mels: Tensor):
        x = self.input_linear(mels)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.hidden_linear(x)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=True)
        return self.output_linear(x)

    def init_weights(self, initializer_range: float = 0.02) -> None:
        nn.init.normal_(self.input_linear.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.hidden_linear.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.output_linear.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.input_linear.bias)
        nn.init.zeros_(self.hidden_linear.bias)
        nn.init.zeros_(self.output_linear.bias)


def reparameterize(mu, logvar, temp=1.0):
    """
    :param mu: (Tensor) Mean of the latent Gaussian
    :param logvar: (Tensor) Standard deviation of the latent Gaussian
    :return:
    """
    std = torch.exp(0.5 * logvar) * temp
    eps = torch.randn_like(std)
    return eps * std + mu


class MelPostNet(nn.Module):
    def __init__(self, input_d, hidden_d, mel_d, dropout_rate=0.5):
        super().__init__()
        self.mel_d = mel_d

        self.before_outs_and_logvar_l = nn.Linear(input_d, mel_d * 2)
        self.vae_decoder = nn.Sequential(
            nn.Linear(mel_d, hidden_d, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_d, hidden_d, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_d, mel_d, bias=False),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: Tensor):
        x = self.before_outs_and_logvar_l(x)
        mu = x[..., : self.mel_d]
        logvar = x[..., self.mel_d :]

        reparameterized_outs = reparameterize(mu, logvar)
        vae_decoder_outs = reparameterized_outs + self.vae_decoder(reparameterized_outs)

        return vae_decoder_outs, mu, logvar

    def init_weights(self, initializer_range: float = 0.02) -> None:
        nn.init.normal_(
            self.before_outs_and_logvar_l.weight, mean=0.0, std=initializer_range
        )
        nn.init.zeros_(self.before_outs_and_logvar_l.bias)
        for layer in self.vae_decoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=initializer_range)


class HNetTTS(BlockBoundaryMixin, nn.Module):
    def __init__(self, c: HNetConfig):
        super().__init__()
        self.c, v, d, text_d = c, c.vocab_size, c.d_model[0], c.d_model[-1]
        self.embeddings = nn.Embedding(v, text_d)
        self.backbone = HNet(c, stage_idx=0)

        # maybe don't need this
        self.mel_prenet = MelPreNet(100, 256, d)
        self.mel_postnet = MelPostNet(d, 256, 100)

        self.mel_bos = nn.Parameter(torch.randn(1, d))
        nn.init.normal_(self.mel_bos, mean=0, std=d**-0.5)
        self.stop_head = nn.Linear(d, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, iids: Tensor, mels: Tensor):
        assert iids.is_nested and iids.ndim == 2
        text_condition = self.embeddings(iids)
        mels_input = self.mel_prenet(mels)

        mels_input = self.dropout(mels_input)

        mels_input = nested.as_nested_tensor(
            [torch.cat((self.mel_bos, m_i[:-1]), dim=0) for m_i in mels_input.unbind()],
            layout=torch.jagged,
        )

        # flatten njt
        cu_s, msl = mels_input.offsets(), mels_input._get_max_seqlen()
        x_flat = mels_input.values()

        x_flat, extra = self.backbone(
            x_flat, cu_s, msl, text_condition.values(), text_condition.offsets()
        )

        stop_logits = self.stop_head(x_flat)
        stop_labels = torch.zeros_like(stop_logits)
        stop_labels[cu_s[1:] - 1] = 1
        loss_bce = F.binary_cross_entropy_with_logits(
            stop_logits,
            stop_labels,
            pos_weight=torch.Tensor([100]).to(stop_logits.device),
            reduction="sum",
        )

        vae_decoder_outs, mu, logvar = self.mel_postnet(x_flat)

        spec_flux_for_loss = F.l1_loss(
            vae_decoder_outs[1:],
            mels.values()[:-1],
            reduction="sum",
        )

        loss_l1 = 2 * F.l1_loss(vae_decoder_outs, mels.values(), reduction="sum")
        loss_l2 = 2 * F.mse_loss(vae_decoder_outs, mels.values(), reduction="sum")

        loss_logvar = (-(1 + logvar - (mu - mels.values()).pow(2) - logvar.exp())).sum()

        # loss_bce may need *0.01
        loss = (
            loss_l1 + loss_l2 + 1e-1 * loss_logvar - 1.0 * spec_flux_for_loss + loss_bce
        )

        vae_decoder_outs = nested.nested_tensor_from_jagged(
            values=vae_decoder_outs, offsets=cu_s, max_seqlen=msl
        )

        return (
            vae_decoder_outs,
            loss,
            loss_l1,
            loss_l2,
            loss_logvar,
            loss_bce,
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
            [torch.cat((self.mel_bos, m_i), dim=0) for m_i in mels_inputs.unbind()]
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
            x_flat, _ = self.backbone(
                x_flat, cu_s, msl, text_condition.values(), text_condition.offsets()
            )

            x = nested.nested_tensor_from_jagged(
                values=x_flat, offsets=cu_s, max_seqlen=msl
            )
            x_last = NJT([x_i[-1].unsqueeze(0) for x_i in x.unbind()])

            stop_logits = self.stop_head(x_last)
            stop_probs = F.sigmoid(stop_logits)

            x_last, mu, logvar = self.mel_postnet(x_last)
            if mel_outputs is None:
                mel_outputs = NJT(x_last.unbind())
            else:
                mel_outputs = NJT(
                    [
                        torch.cat((m_i, x_i), dim=0)
                        for m_i, x_i in zip(mel_outputs.unbind(), x_last.unbind())
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

            x_last = self.mel_prenet(x_last)

            mels_inputs = NJT(
                [torch.cat((mels_inputs[pos], x_last[pos]), dim=0) for pos in survivors]
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
