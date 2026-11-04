#!/usr/bin/env python3
"""
1D-CNN ECG Anomaly Detector Generator (make_model.py)
Exports a self-contained ONNX model (ecg_anomaly.onnx) for bedside inference.
"""

import os
import onnx
import torch
import torch.nn as nn


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
    model = ECGAnomaly1DCNN().eval()
    dummy_input = torch.randn(1, 1, 250)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecg_anomaly.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["ecg_input"],
        output_names=["logits"],
        dynamic_axes={"ecg_input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=18,
    )

    onnx_model = onnx.load(output_path)
    onnx.save_model(onnx_model, output_path, save_as_external_data=False)
    if os.path.exists(output_path + ".data"):
        os.remove(output_path + ".data")

    print(f"[OK] Exported {output_path} successfully as a self-contained model.")
