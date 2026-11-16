#!/usr/bin/env python3
"""
1D-CNN ECG Anomaly Detector Generator (make_model.py)
Exports a self-contained ONNX model (ecg_anomaly.onnx) with IR version 8 / opset 13
trained to accurately classify Normal Sinus Rhythm (PQRST) vs. Ventricular Arrhythmia (PVCs).
"""

import os
import onnx
import torch
import torch.nn as nn
import numpy as np


class ECGAnomaly1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    model = ECGAnomaly1DCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    t = np.linspace(0, 1, 250)
    X = []
    Y = []
    for _ in range(60):
        c = np.random.uniform(-0.05, 0.05)
        # Normal Sinus (Class 0)
        p = 0.15 * np.exp(-((t - (0.20 + c)) ** 2) / (2 * 0.025 ** 2))
        q = -0.15 * np.exp(-((t - (0.35 + c)) ** 2) / (2 * 0.010 ** 2))
        r = 1.25 * np.exp(-((t - (0.40 + c)) ** 2) / (2 * 0.012 ** 2))
        s = -0.25 * np.exp(-((t - (0.45 + c)) ** 2) / (2 * 0.012 ** 2))
        tw = 0.30 * np.exp(-((t - (0.65 + c)) ** 2) / (2 * 0.040 ** 2))
        noise = np.random.normal(0, 0.02, 250)
        X.append(p + q + r + s + tw + noise)
        Y.append(0)

        # Ventricular Arrhythmia (Class 1)
        pvc1 = 1.75 * np.exp(-((t - (0.22 + c)) ** 2) / (2 * 0.035 ** 2))
        pvc2 = -1.50 * np.exp(-((t - (0.42 + c)) ** 2) / (2 * 0.040 ** 2))
        pvc3 = 1.85 * np.exp(-((t - (0.62 + c)) ** 2) / (2 * 0.035 ** 2))
        pvc4 = -1.35 * np.exp(-((t - (0.82 + c)) ** 2) / (2 * 0.040 ** 2))
        X.append(pvc1 + pvc2 + pvc3 + pvc4 + noise)
        Y.append(1)

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.long)

    model.train()
    for epoch in range(35):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, Y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecg_anomaly.onnx")
    dummy_input = torch.randn(1, 1, 250)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["ecg_input"],
        output_names=["logits"],
        dynamic_axes={"ecg_input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=13,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx_model.ir_version = 8
    for opset in onnx_model.opset_import:
        if opset.domain == "" and opset.version > 13:
            opset.version = 13
    onnx.save_model(onnx_model, output_path, save_as_external_data=False)
    if os.path.exists(output_path + ".data"):
        os.remove(output_path + ".data")

    print(
        f"[OK] Exported trained {output_path} (IR version={onnx_model.ir_version}) as a self-contained model."
    )
