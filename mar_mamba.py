import math

import torch
import torch.nn.functional as F
from torch import nn

from base_model import SequenceModel


class LearnableMFWT(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        scale = math.sqrt(2.0)
        self.levels = levels
        self.analysis_low = nn.Parameter(torch.tensor([1.0 / scale, 1.0 / scale]))
        self.analysis_high = nn.Parameter(torch.tensor([-1.0 / scale, 1.0 / scale]))
        self.synthesis_low = nn.Parameter(torch.tensor([1.0 / scale, 1.0 / scale]))
        self.synthesis_high = nn.Parameter(torch.tensor([1.0 / scale, -1.0 / scale]))

    def _filter(self, x, kernel, dilation):
        batch, stocks, steps, channels = x.shape
        sequence = x.permute(0, 1, 3, 2).reshape(batch * stocks, channels, steps)
        padding = dilation * (kernel.numel() - 1)
        sequence = F.pad(sequence, (padding, 0), mode="circular")
        weight = kernel.flip(0).view(1, 1, -1).repeat(channels, 1, 1)
        output = F.conv1d(sequence, weight, dilation=dilation, groups=channels)
        return output.reshape(batch, stocks, channels, steps).permute(0, 1, 3, 2)

    def analysis(self, x):
        approximation = x
        details = []
        for level in range(self.levels):
            dilation = 2 ** level
            details.append(self._filter(approximation, self.analysis_high, dilation))
            approximation = self._filter(approximation, self.analysis_low, dilation)
        return torch.stack(details + [approximation], dim=2)

    def synthesis(self, bands):
        approximation = bands[:, :, -1]
        for level in range(self.levels - 1, -1, -1):
            dilation = 2 ** level
            low = self._filter(approximation, self.synthesis_low, dilation)
            high = self._filter(bands[:, :, level], self.synthesis_high, dilation)
            approximation = 0.5 * (low + high)
        return approximation

    def forward(self, x):
        return self.analysis(x)


class MarketContextEncoder(nn.Module):
    def __init__(self, market_dim, d_model, context_dim, dropout):
        super().__init__()
        self.context_dim = context_dim
        self.embedding = nn.Linear(market_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, context_dim, bias=False)
        self.key = nn.Linear(d_model, context_dim, bias=False)
        self.value = nn.Linear(d_model, context_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, market):
        hidden = self.norm(self.embedding(market))
        query = self.query(hidden[:, -1:])
        key = self.key(hidden)
        value = self.value(hidden)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.context_dim)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(weights, value).squeeze(1)


class CrossBandRelayAttention(nn.Module):
    def __init__(
        self,
        num_stocks,
        d_model,
        market_dim,
        context_dim=32,
        wavelet_levels=3,
        relay_ratio=0.125,
        num_heads=4,
        dropout=0.3,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_stocks < 1:
            raise ValueError("num_stocks must be positive")
        if not 0 < relay_ratio <= 1:
            raise ValueError("relay_ratio must be in (0, 1]")
        self.num_stocks = num_stocks
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wavelet_levels = wavelet_levels
        self.num_bands = wavelet_levels + 1
        self.num_tokens = num_stocks * self.num_bands
        self.num_relays = max(1, int(math.floor(relay_ratio * num_stocks)))
        self.mfwt = LearnableMFWT(wavelet_levels)
        self.market_encoder = MarketContextEncoder(market_dim, d_model, context_dim, dropout)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gamma_q = nn.Linear(context_dim, d_model)
        self.beta_q = nn.Linear(context_dim, d_model)
        self.gamma_k = nn.Linear(context_dim, d_model)
        self.beta_k = nn.Linear(context_dim, d_model)
        self.key_projection = nn.Parameter(torch.empty(self.num_relays, self.num_tokens))
        self.value_projection = nn.Parameter(torch.empty(self.num_relays, self.num_tokens))
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.key_projection)
        nn.init.xavier_uniform_(self.value_projection)
        for layer in (self.gamma_q, self.beta_q, self.gamma_k, self.beta_k):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _projection_weights(self, logits, token_mask):
        expanded = logits.unsqueeze(0).expand(token_mask.shape[0], -1, -1)
        expanded = expanded.masked_fill(~token_mask.unsqueeze(1), torch.finfo(expanded.dtype).min)
        return torch.softmax(expanded, dim=-1)

    def forward(self, x, market, stock_mask=None, return_attention=False):
        batch, stocks, steps, channels = x.shape
        if stocks != self.num_stocks:
            raise ValueError(f"expected {self.num_stocks} stocks, received {stocks}")
        if stock_mask is None:
            stock_mask = torch.ones(batch, stocks, dtype=torch.bool, device=x.device)
        if torch.any(stock_mask.sum(dim=1) == 0):
            raise ValueError("each sample must contain at least one valid stock")
        bands = self.mfwt.analysis(x)
        tokens = bands.reshape(batch, self.num_tokens, steps, channels)
        token_mask = stock_mask.unsqueeze(-1).expand(-1, -1, self.num_bands).reshape(batch, self.num_tokens)
        tokens = tokens * token_mask[:, :, None, None]
        context = self.market_encoder(market)
        query = self.query(tokens)
        key = self.key(tokens)
        value = self.value(tokens)
        conditioned_query = query * (1.0 + self.gamma_q(context)[:, None, None]) + self.beta_q(context)[:, None, None]
        conditioned_key = key * (1.0 + self.gamma_k(context)[:, None, None]) + self.beta_k(context)[:, None, None]
        key_weights = self._projection_weights(self.key_projection, token_mask)
        value_weights = self._projection_weights(self.value_projection, token_mask)
        relay_key = torch.einsum("brl,bltd->brtd", key_weights, conditioned_key)
        relay_value = torch.einsum("brl,bltd->brtd", value_weights, value)
        query_heads = conditioned_query.reshape(batch, self.num_tokens, steps, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        key_heads = relay_key.reshape(batch, self.num_relays, steps, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        value_heads = relay_value.reshape(batch, self.num_relays, steps, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        logits = torch.einsum("bthld,bthrd->bthlr", query_heads, key_heads) / math.sqrt(self.head_dim)
        attention = self.attention_dropout(torch.softmax(logits, dim=-1))
        attended = torch.einsum("bthlr,bthrd->bthld", attention, value_heads)
        attended = attended.permute(0, 3, 1, 2, 4).reshape(batch, self.num_tokens, steps, channels)
        attended = self.output_dropout(self.output(attended))
        attended = attended * token_mask[:, :, None, None]
        reconstructed = self.mfwt.synthesis(attended.reshape(batch, stocks, self.num_bands, steps, channels))
        reconstructed = reconstructed * stock_mask[:, :, None, None]
        if not return_attention:
            return reconstructed
        maps = {
            "token_to_relay": attention.mean(dim=2),
            "key_pool": key_weights,
            "value_pool": value_weights,
        }
        return reconstructed, maps


class SelectiveStateSpace(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))
        self.parameter_projection = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.delta_projection = nn.Linear(self.dt_rank, d_model)
        base = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(torch.log(base).repeat(d_model, 1))
        self.skip = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        batch, stocks, steps, channels = x.shape
        sequence = x.reshape(batch * stocks, steps, channels)
        parameters = self.parameter_projection(sequence)
        delta_raw, b_param, c_param = torch.split(parameters, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.delta_projection(delta_raw))
        transition = -torch.exp(self.a_log).to(dtype=x.dtype)
        state = x.new_zeros(batch * stocks, channels, self.d_state)
        outputs = []
        for step in range(steps):
            delta_step = delta[:, step].unsqueeze(-1)
            transition_step = torch.exp(delta_step * transition.unsqueeze(0))
            input_step = sequence[:, step]
            state = transition_step * state + delta_step * b_param[:, step].unsqueeze(1) * input_step.unsqueeze(-1)
            output_step = torch.sum(state * c_param[:, step].unsqueeze(1), dim=-1) + self.skip * input_step
            outputs.append(output_step)
        output = torch.stack(outputs, dim=1)
        return output.reshape(batch, stocks, steps, channels)


class RIMambaBlock(nn.Module):
    def __init__(
        self,
        num_stocks,
        d_model,
        market_dim,
        context_dim=32,
        wavelet_levels=3,
        relay_ratio=0.125,
        num_heads=4,
        d_state=16,
        dt_rank=None,
        ffn_ratio=2,
        dropout=0.3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, 2 * d_model)
        self.cbra = CrossBandRelayAttention(
            num_stocks=num_stocks,
            d_model=d_model,
            market_dim=market_dim,
            context_dim=context_dim,
            wavelet_levels=wavelet_levels,
            relay_ratio=relay_ratio,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ssm = SelectiveStateSpace(d_model, d_state=d_state, dt_rank=dt_rank)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, market, stock_mask=None, return_attention=False):
        residual = x
        left, right = self.input_projection(self.input_norm(x)).chunk(2, dim=-1)
        if return_attention:
            spatial, maps = self.cbra(right, market, stock_mask, True)
        else:
            spatial = self.cbra(right, market, stock_mask, False)
            maps = None
        propagated = self.ssm(F.silu(spatial))
        output = self.output_projection(self.output_norm(propagated * F.silu(left)))
        hidden = residual + self.output_dropout(output)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        if stock_mask is not None:
            hidden = hidden * stock_mask[:, :, None, None]
        if return_attention:
            return hidden, maps
        return hidden


class MARMamba(nn.Module):
    def __init__(
        self,
        d_feat,
        num_stocks,
        market_dim,
        d_model=128,
        wavelet_levels=3,
        relay_ratio=0.125,
        market_context_dim=32,
        dropout=0.3,
        num_layers=1,
        num_heads=4,
        d_state=16,
        dt_rank=None,
        ffn_ratio=2,
        market_feature_start=None,
        market_feature_end=None,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.num_stocks = num_stocks
        self.market_dim = market_dim
        self.market_feature_start = d_feat if market_feature_start is None else market_feature_start
        self.market_feature_end = self.market_feature_start + market_dim if market_feature_end is None else market_feature_end
        if self.market_feature_end - self.market_feature_start != market_dim:
            raise ValueError("market feature range must match market_dim")
        self.embedding = nn.Linear(d_feat, d_model)
        self.blocks = nn.ModuleList(
            [
                RIMambaBlock(
                    num_stocks=num_stocks,
                    d_model=d_model,
                    market_dim=market_dim,
                    context_dim=market_context_dim,
                    wavelet_levels=wavelet_levels,
                    relay_ratio=relay_ratio,
                    num_heads=num_heads,
                    d_state=d_state,
                    dt_rank=dt_rank,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.prediction_head = nn.Linear(d_model, 1)

    def _prepare_inputs(self, x, market, stock_mask):
        unbatched = x.dim() == 3
        if unbatched:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError("stock input must have shape [N,T,F] or [B,N,T,F]")
        if x.shape[1] != self.num_stocks:
            raise ValueError(f"expected {self.num_stocks} stocks, received {x.shape[1]}")
        if stock_mask is None:
            stock_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif stock_mask.dim() == 1:
            stock_mask = stock_mask.unsqueeze(0)
        stock_mask = stock_mask.to(device=x.device, dtype=torch.bool)
        if market is None:
            if x.shape[-1] < self.market_feature_end:
                raise ValueError("market input is required when combined features do not contain the configured market range")
            market_tokens = x[..., self.market_feature_start:self.market_feature_end]
            weights = stock_mask[:, :, None, None].to(dtype=x.dtype)
            market = (market_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        elif market.dim() == 2:
            market = market.unsqueeze(0)
        market = market.to(device=x.device, dtype=x.dtype)
        stock = x[..., :self.d_feat]
        return stock, market, stock_mask, unbatched

    def forward(self, x, market=None, stock_mask=None, return_attention=False):
        stock, market, stock_mask, unbatched = self._prepare_inputs(x, market, stock_mask)
        hidden = self.embedding(stock)
        hidden = hidden * stock_mask[:, :, None, None]
        maps = []
        for block in self.blocks:
            if return_attention:
                hidden, block_maps = block(hidden, market, stock_mask, True)
                maps.append(block_maps)
            else:
                hidden = block(hidden, market, stock_mask, False)
        scores = self.prediction_head(self.final_norm(hidden[:, :, -1])).squeeze(-1)
        scores = scores.masked_fill(~stock_mask, 0.0)
        if unbatched:
            scores = scores.squeeze(0)
            if return_attention:
                maps = [{name: value.squeeze(0) for name, value in item.items()} for item in maps]
        if return_attention:
            return scores, maps
        return scores


class MARMambaModel(SequenceModel):
    def __init__(
        self,
        d_feat,
        num_stocks,
        market_dim,
        d_model=128,
        wavelet_levels=3,
        relay_ratio=0.125,
        market_context_dim=32,
        dropout=0.3,
        num_layers=1,
        num_heads=4,
        d_state=16,
        dt_rank=None,
        ffn_ratio=2,
        market_feature_start=None,
        market_feature_end=None,
        lambda_mse=0.1,
        **kwargs,
    ):
        self.lambda_mse = lambda_mse
        super().__init__(**kwargs)
        self.model = MARMamba(
            d_feat=d_feat,
            num_stocks=num_stocks,
            market_dim=market_dim,
            d_model=d_model,
            wavelet_levels=wavelet_levels,
            relay_ratio=relay_ratio,
            market_context_dim=market_context_dim,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            d_state=d_state,
            dt_rank=dt_rank,
            ffn_ratio=ffn_ratio,
            market_feature_start=market_feature_start,
            market_feature_end=market_feature_end,
        )
        self.init_model()

    def loss_fn(self, prediction, label):
        mask = torch.isfinite(label) & torch.isfinite(prediction)
        prediction = prediction[mask]
        label = label[mask]
        if prediction.numel() < 2:
            return F.mse_loss(prediction, label)
        prediction_centered = prediction - prediction.mean()
        label_centered = label - label.mean()
        denominator = torch.sqrt(torch.sum(prediction_centered.square()) * torch.sum(label_centered.square())).clamp_min(1e-8)
        correlation = torch.sum(prediction_centered * label_centered) / denominator
        return -correlation + self.lambda_mse * F.mse_loss(prediction, label)


MAR_Mamba = MARMamba
MAR_MambaModel = MARMambaModel


__all__ = [
    "LearnableMFWT",
    "MarketContextEncoder",
    "CrossBandRelayAttention",
    "SelectiveStateSpace",
    "RIMambaBlock",
    "MARMamba",
    "MARMambaModel",
    "MAR_Mamba",
    "MAR_MambaModel",
]
