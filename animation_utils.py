"""
Animation Utilities for Training Metrics

Provides helper to create animated graphs (GIF/MP4) of training/validation loss and R^2 over epochs
using matplotlib.animation. Designed to be resilient and optional.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _extract_series(training_history: List[Dict[str, Any]]):
    epochs = [int(row.get("epoch", idx)) for idx, row in enumerate(training_history)]
    train_loss = [float(row.get("train_loss", np.nan)) for row in training_history]
    val_loss = [float(row.get("val_loss", np.nan)) for row in training_history]
    r2 = [float(row.get("r2_score", np.nan)) for row in training_history]
    return np.array(epochs), np.array(train_loss), np.array(val_loss), np.array(r2)


def create_training_animation(training_history: List[Dict[str, Any]],
                              output_path: str,
                              fps: int = 8,
                              dpi: int = 120,
                              max_frames: Optional[int] = None) -> Optional[str]:
    """
    Create and save an animated GIF that shows train/val loss and R^2 evolving over epochs.

    Args:
        training_history: List of epoch dictionaries from PerformanceTracker
        output_path: Target path for the .gif file
        fps: Frames per second for the animation
        dpi: Resolution
        max_frames: If provided, down-sample to at most this many frames

    Returns:
        The path to the saved GIF or None if creation failed
    """
    try:
        if not training_history or len(training_history) < 3:
            return None

        epochs, train_loss, val_loss, r2 = _extract_series(training_history)
        n = len(epochs)

        if max_frames and n > max_frames:
            # Uniformly sample frames
            idx = np.linspace(0, n - 1, max_frames).astype(int)
            epochs, train_loss, val_loss, r2 = epochs[idx], train_loss[idx], val_loss[idx], r2[idx]
            n = len(epochs)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
        fig.suptitle("Training Progress Animation")

        # Loss axes
        ax1.set_title("Loss over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        (train_line,) = ax1.plot([], [], label="Train Loss", color="#1f77b4")
        (val_line,) = ax1.plot([], [], label="Val Loss", color="#ff7f0e")
        ax1.legend(loc="upper right")

        # R2 axes
        ax2.set_title("R² over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("R²")
        ax2.set_ylim(-1.0, 1.0)
        ax2.grid(True, alpha=0.3)
        (r2_line,) = ax2.plot([], [], label="R²", color="#2ca02c")
        ax2.legend(loc="lower right")

        # Precompute axis limits for stability
        valid_losses = np.concatenate([np.nan_to_num(train_loss, nan=np.nan), np.nan_to_num(val_loss, nan=np.nan)])
        finite_losses = valid_losses[np.isfinite(valid_losses)]
        if finite_losses.size:
            ymin, ymax = np.min(finite_losses), np.max(finite_losses)
            if ymin == ymax: ymax = ymin + 1.0
            ax1.set_ylim(max(0.0, ymin * 0.9), ymax * 1.1)
        ax1.set_xlim(int(epochs[0]), int(epochs[-1]))
        ax2.set_xlim(int(epochs[0]), int(epochs[-1]))

        def init():
            train_line.set_data([], [])
            val_line.set_data([], [])
            r2_line.set_data([], [])
            return train_line, val_line, r2_line

        def update(frame_idx: int):
            ep = epochs[: frame_idx + 1]
            tr = train_loss[: frame_idx + 1]
            vl = val_loss[: frame_idx + 1]
            r = r2[: frame_idx + 1]
            train_line.set_data(ep, tr)
            val_line.set_data(ep, vl)
            r2_line.set_data(ep, r)
            return train_line, val_line, r2_line

        anim = FuncAnimation(fig, update, init_func=init, frames=n, interval=1000 // fps, blit=True)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close(fig)
        return output_path
    except Exception as e:
        try:
            plt.close('all')
        except Exception:
            pass
        print(f"Warning: Failed to create training animation: {e}")
        return None
