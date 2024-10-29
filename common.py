import pandas as pd
from functools import cache
from torch import tensor, long
from torch.utils.data import Dataset
from PIL import Image


class FairDataset(Dataset):
    def __init__(self, txt_path, transformation_function=None, with_predicted=False):
        f = open(txt_path, "r")
        imgs = []
        self.with_predicted = with_predicted
        if self.with_predicted:
            for line in f:
                line = line.rstrip()
                img_path, expected, predicted, gender, race = line.split()
                imgs.append((img_path, expected, predicted, gender, race))


        else:
            for line in f:
                line = line.rstrip()
                img_path, expected, gender, race = line.split()
                imgs.append((img_path, expected, gender, race))

        self.imgs = imgs
        self.transformation_function = transformation_function

    def __getitem__(self, index):
        img_path, label, *other = self.imgs[index]
        img = Image.open(img_path).convert("RGB")

        if self.transformation_function is not None:
            img = self.transformation_function(img)

        label = tensor(int(label), dtype=long)
        if self.with_predicted:
            predicted, gender, race = other
            return img_path, label, predicted, gender, race
        else:
            gender, race = other
            return img, label, race

    def __len__(self):
        return len(self.imgs)

    def attributes(self, index):
        return self.imgs[index]


class DataFrame:
    def __init__(self, predicted, expected, groups):
        self.data = pd.DataFrame(
            {"Predicted": predicted, "Expected": expected, "Group": groups}
        )

    def groups(self):
        return self.data["Group"].unique()

    def _group_data(self, group):
        if group:
            return self.data[self.data["Group"] == group]
        return self.data

    @cache
    def tp(self, group=None):
        group_data = self._group_data(group)
        tp = ((group_data["Expected"] == 1) & (group_data["Predicted"] == 1)).sum()
        return tp

    @cache
    def tn(self, group=None):
        group_data = self._group_data(group)
        tn = ((group_data["Expected"] == 0) & (group_data["Predicted"] == 0)).sum()
        return tn

    @cache
    def fp(self, group=None):
        group_data = self._group_data(group)
        fp = ((group_data["Expected"] == 0) & (group_data["Predicted"] == 1)).sum()
        return fp

    @cache
    def fn(self, group=None):
        group_data = self._group_data(group)
        fn = ((group_data["Expected"] == 1) & (group_data["Predicted"] == 0)).sum()
        return fn

    @cache
    def accuracy(self, group=None):
        return (self.tp(group) + self.tn(group)) / (
            self.tp(group) + self.tn(group) + self.fp(group) + self.fn(group)
        )

    @cache
    def confusion_matrix(self, group=None):
        return {
            "TP": self.tp(group),
            "TN": self.tn(group),
            "FP": self.fp(group),
            "FN": self.fn(group),
        }

    def confusion_matrix_per_group(self):
        groups = self.data["Group"].unique()
        data = []
        for group in groups:
            metrics = self.confusion_matrix(group)
            metrics["Group"] = group
            data.append(metrics)
        return pd.DataFrame(data)


def _compute_parity(parity_dict, one):
    if one:
        max_parity = max(parity_dict.values())
        min_parity = min(parity_dict.values())
        if max_parity != 0:
            parity_ratio = min_parity / max_parity
        else:
            parity_ratio = 0
        return parity_ratio

    parity_ratios = {}
    groups = list(parity_dict.keys())
    for group in groups:
        parity = parity_dict[group]
        parity_ratios[group] = parity
    return parity_ratios


def display_parities(parities, text="INSERT `text`"):
    values = [(group, ratio) for group, ratio in parities.items()]
    for group, ratio in sorted(values, key=lambda o: o[0]):
        print(f"{text} ({group}): {ratio:.2f}")
