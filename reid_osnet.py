"""Descripteurs d'apparence OSNet x0.25 pour la ReID de BoT-SORT.

Réécriture de l'architecture OSNet (Zhou et al., ICCV 2019) d'après torchreid
(KaiyangZhou/deep-person-reid, licence MIT), avec les mêmes noms de couches
pour charger ses poids tels quels. Aucune dépendance à boxmot (AGPL-3.0).

Poids : `osnet_x0_25_msmt17.pt` du zoo de torchreid (entraîné sur MSMT17,
des piétons) — c'est celui avec lequel BoT-SORT a été mesuré sur UAVDT.

    enc = EncodeurOSNet("osnet_x0_25_msmt17.pt", device="cuda")
    feats = enc(image_bgr, boites_xyxy)   # (N, 512) float32, normes à 1
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Entrée du réseau (hauteur, largeur) et normalisation ImageNet, comme à
# l'entraînement torchreid.
TAILLE_ENTREE = (256, 128)
MOYENNE = (0.485, 0.456, 0.406)
ECART_TYPE = (0.229, 0.224, 0.225)


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv1x1(ConvLayer):
    def __init__(self, cin, cout, stride=1, groups=1):
        super().__init__(cin, cout, 1, stride=stride, groups=groups)


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 puis 3x3 en profondeur (depthwise)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False,
                               groups=cout)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))), inplace=True)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1)
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(g), inplace=True)))
        return x * g


class OSBlock(nn.Module):
    """Bloc omni-échelle : quatre flux de 1 à 4 LightConv3x3, porte partagée."""

    def __init__(self, cin, cout, bottleneck_reduction=4):
        super().__init__()
        mid = cout // bottleneck_reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
              + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        identite = x if self.downsample is None else self.downsample(x)
        return F.relu(self.conv3(x2) + identite)


class OSNet(nn.Module):
    def __init__(self, layers=(2, 2, 2), channels=(16, 64, 96, 128),
                 feature_dim=512, num_classes=1041):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._couche(layers[0], channels[0], channels[1], True)
        self.conv3 = self._couche(layers[1], channels[1], channels[2], True)
        self.conv4 = self._couche(layers[2], channels[2], channels[3], False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim),
                                nn.BatchNorm1d(feature_dim),
                                nn.ReLU(inplace=True))
        # Inutile à l'inférence, gardé pour charger le point de contrôle strict.
        self.classifier = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _couche(n, cin, cout, reduire):
        blocs = [OSBlock(cin, cout)] + [OSBlock(cout, cout)
                                        for _ in range(n - 1)]
        if reduire:
            blocs.append(nn.Sequential(Conv1x1(cout, cout),
                                       nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*blocs)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def charger_osnet(chemin: str, device="cpu") -> OSNet:
    etat = torch.load(chemin, map_location="cpu", weights_only=False)
    etat = etat.get("state_dict", etat)
    etat = {k.removeprefix("module."): v for k, v in etat.items()}
    modele = OSNet(num_classes=etat["classifier.weight"].shape[0])
    modele.load_state_dict(etat, strict=True)
    return modele.eval().to(device)


class EncodeurOSNet:
    """Découpe les boîtes dans l'image, renvoie des descripteurs normés."""

    def __init__(self, poids: str, device="cuda", lot=256, half=True):
        self.device = torch.device(device)
        self.modele = charger_osnet(poids, self.device)
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.modele.half()
        self.lot = lot
        dt = torch.float16 if self.half else torch.float32
        self.moyenne = torch.tensor(MOYENNE, device=self.device,
                                    dtype=dt).view(1, 3, 1, 1)
        self.ecart = torch.tensor(ECART_TYPE, device=self.device,
                                  dtype=dt).view(1, 3, 1, 1)

    @property
    def dim(self) -> int:
        return self.modele.fc[0].out_features

    def decoupes(self, image_bgr: np.ndarray, boites: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 RGB, chaque boîte redimensionnée à l'entrée."""
        import cv2
        h, w = image_bgr.shape[:2]
        th, tw = TAILLE_ENTREE
        sortie = np.empty((len(boites), th, tw, 3), np.uint8)
        for k, (x1, y1, x2, y2) in enumerate(np.asarray(boites, float)):
            x1 = int(np.clip(np.floor(x1), 0, w - 1))
            y1 = int(np.clip(np.floor(y1), 0, h - 1))
            x2 = int(np.clip(np.ceil(x2), x1 + 1, w))
            y2 = int(np.clip(np.ceil(y2), y1 + 1, h))
            crop = cv2.resize(image_bgr[y1:y2, x1:x2], (tw, th),
                              interpolation=cv2.INTER_LINEAR)
            sortie[k] = crop[:, :, ::-1]
        return sortie

    @torch.inference_mode()
    def descripteurs(self, decoupes_rgb: np.ndarray) -> np.ndarray:
        if len(decoupes_rgb) == 0:
            return np.zeros((0, self.dim), np.float32)
        sorties = []
        for debut in range(0, len(decoupes_rgb), self.lot):
            x = torch.from_numpy(decoupes_rgb[debut:debut + self.lot]).to(
                self.device, non_blocking=True)
            x = x.permute(0, 3, 1, 2).to(self.moyenne.dtype).div_(255)
            x = (x - self.moyenne) / self.ecart
            f = self.modele(x).float()
            sorties.append(F.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(sorties)

    def __call__(self, image_bgr, boites):
        return self.descripteurs(self.decoupes(image_bgr, boites))
