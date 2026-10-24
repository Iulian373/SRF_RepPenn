import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

def keypoints_to_heatmaps(kps, output_size=(64,64), sigma=1.5, selected_indices=None):
    F, K, _ = kps.shape
    H, W = output_size
    if selected_indices is None:
        selected_indices = list(range(K))
    Ksel = len(selected_indices)
    heatmaps = np.zeros((F, Ksel, H, W), dtype=np.float32)

    xs = np.arange(W)
    ys = np.arange(H)
    xv, yv = np.meshgrid(xs, ys)

    for f in range(F):
        for ki, kidx in enumerate(selected_indices):
            x, y, vis = kps[f, kidx]
            if vis < 0.1 or x<=0 or y<=0:
                continue
            cx = x
            cy = y
            g = np.exp(-((xv - cx)**2 + (yv - cy)**2) / (2 * sigma * sigma))
            heatmaps[f, ki] = g
        # heatmaps[f] /= (heatmaps[f].max(axis=(1,2), keepdims=True) + 1e-6)
    return heatmaps  # (F, Ksel, H, W)

def scale_keypoints_to_size(kps, frame_size, target_size=(64,64)):
    Wf, Hf = frame_size
    Wt, Ht = target_size
    scale_x = Wt / float(Wf)
    scale_y = Ht / float(Hf)
    kps_scaled = kps.copy()
    kps_scaled[:, :, 0] = kps[:, :, 0] * scale_x
    kps_scaled[:, :, 1] = kps[:, :, 1] * scale_y
    return kps_scaled

# def heatmaps_to_Pmap(heatmaps):
#     F, K, H, W = heatmaps.shape
#     J = heatmaps.transpose(2,3,0,1)  # (H,W,F,K)
#
#     A = J.mean(0)   # (W,F,K)
#     B = J.mean(1)   # (H,F,K)
#     P = np.concatenate([B,A],axis=0)  # (H+W, F, K)
#
#     Pn = 255*(P - P.min())/(P.max()-P.min()+1e-6)
#
#     feat = np.zeros((224,224,K), np.uint8)
#     for k in range(K):
#         feat[:,:,k] = cv2.resize(Pn[:,:,k].astype(np.uint8), (224,224))
#
#     return torch.from_numpy(feat.transpose(2,0,1)).float() / 255.0

def heatmaps_to_processed_feature(heatmaps):
    # (F, K, H, W) to J[h,w,f,k]
    J = np.transpose(heatmaps, (2,3,0,1))  # (H,W,F,K)
    A = np.mean(J, axis=0)  # (W, F, K)
    B = np.mean(J, axis=1)  # (H, F, K)
    P = np.concatenate([B, A], axis=0)  # (H+W, F, K)
    pmin, pmax = P.min(), P.max()
    if pmax - pmin < 1e-6:
        Pn = np.zeros_like(P)
    else:
        Pn = 255.0 * (P - pmin) / (pmax - pmin)
    Kfeat = Pn.shape[2]
    feat_maps = np.zeros((224,224,Kfeat), dtype=np.uint8)
    for k in range(Kfeat):
        mat = Pn[:,:,k]  # shape ((H+W), F)
        mat_norm = mat.astype(np.uint8)
        resized = cv2.resize(mat_norm, (224,224), interpolation=cv2.INTER_LINEAR)
        feat_maps[:,:,k] = resized
    return feat_maps

class MultiTaskModel(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )

        self.recog_net = models.resnet34(weights='IMAGENET1K_V1')
        self.recog_net.fc = nn.Linear(self.recog_net.fc.in_features, num_classes)

        self.count_net = models.resnet18(weights='IMAGENET1K_V1')
        self.count_net.fc = nn.Linear(self.count_net.fc.in_features, 1)

    def forward(self, x):
        x = self.shared(x)
        cls_out = self.recog_net(x)
        cnt_out = self.count_net(x).squeeze()
        return cls_out, cnt_out