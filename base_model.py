import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler


def calc_ic(prediction, label):
    frame = pd.DataFrame({"prediction": prediction, "label": label}).dropna()
    if len(frame) < 2:
        return np.nan, np.nan
    return frame["prediction"].corr(frame["label"]), frame["prediction"].corr(frame["label"], method="spearman")


def zscore_valid(x, mask):
    output = torch.full_like(x, torch.nan)
    values = x[mask]
    scale = values.std(unbiased=True).clamp_min(1e-8)
    output[mask] = (values - values.mean()) / scale
    return output


def valid_label_mask(label, drop_fraction=0.025):
    mask = torch.isfinite(label)
    indices = torch.where(mask)[0]
    if indices.numel() < 4 or drop_fraction <= 0:
        return mask
    trim = int(drop_fraction * indices.numel())
    if trim == 0 or 2 * trim >= indices.numel():
        return mask
    order = torch.argsort(label[indices])
    keep = indices[order[trim:-trim]]
    output = torch.zeros_like(mask)
    output[keep] = True
    return output


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.daily_count = pd.Series(index=data_source.get_index(), dtype=float).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        order = np.arange(len(self.daily_count))
        if self.shuffle:
            np.random.shuffle(order)
        for index in order:
            yield np.arange(self.daily_index[index], self.daily_index[index] + self.daily_count[index])

    def __len__(self):
        return len(self.daily_count)


class SequenceModel:
    def __init__(
        self,
        n_epochs,
        lr,
        weight_decay=0.0,
        GPU=None,
        seed=None,
        train_stop_loss_thred=None,
        save_path="model",
        save_prefix="",
    ):
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(f"cuda:{GPU}" if GPU is not None and torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.save_path = Path(save_path)
        self.save_prefix = save_prefix
        self.fitted = -1
        self.model = None
        self.train_optimizer = None
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")
        self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.to(self.device)

    def loss_fn(self, prediction, label):
        mask = torch.isfinite(label) & torch.isfinite(prediction)
        return torch.mean((prediction[mask] - label[mask]).square())

    def _prepare_day(self, data, training):
        data = torch.squeeze(data, dim=0)
        feature = data[:, :, :-1].to(self.device).float()
        label = data[:, -1, -1].to(self.device).float()
        mask = valid_label_mask(label, 0.025 if training else 0.0)
        normalized = zscore_valid(label, mask)
        return feature, normalized, mask

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        for data in data_loader:
            feature, label, mask = self._prepare_day(data, True)
            prediction = self.model(feature)
            loss = self.loss_fn(prediction[mask], label[mask])
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in data_loader:
                feature, label, mask = self._prepare_day(data, False)
                prediction = self.model(feature)
                losses.append(self.loss_fn(prediction[mask], label[mask]).item())
        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        return DataLoader(data, sampler=sampler, drop_last=drop_last)

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = "loaded"

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_score = -np.inf
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid is not None:
                _, metrics = self.predict(dl_valid)
                score = metrics["IC"]
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_param = copy.deepcopy(self.model.state_dict())
            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                break
        if best_param is None:
            best_param = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_param)
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self.save_path / f"{self.save_prefix}_{self.seed}.pkl"
        torch.save(best_param, path)
        return path

    def predict(self, dl_test):
        if self.fitted == -1:
            raise ValueError("model is not fitted")
        loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        predictions = []
        daily_ic = []
        daily_rank_ic = []
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                feature, label, _ = self._prepare_day(data, False)
                prediction = self.model(feature).detach().cpu().numpy()
                raw_label = label.detach().cpu().numpy()
                predictions.append(prediction.ravel())
                ic, rank_ic = calc_ic(prediction, raw_label)
                daily_ic.append(ic)
                daily_rank_ic.append(rank_ic)
        series = pd.Series(np.concatenate(predictions), index=dl_test.get_index())
        ic_values = np.asarray(daily_ic, dtype=float)
        rank_values = np.asarray(daily_rank_ic, dtype=float)
        metrics = {
            "IC": np.nanmean(ic_values),
            "ICIR": np.nanmean(ic_values) / np.nanstd(ic_values),
            "RIC": np.nanmean(rank_values),
            "RICIR": np.nanmean(rank_values) / np.nanstd(rank_values),
        }
        return series, metrics
