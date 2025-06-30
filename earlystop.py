import torch
import numpy as np
import os
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_fn=None):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False 
        self.score_max = -np.inf
        self.save_fn = save_fn

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[EarlyStopping] Counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True 
        else:
            if self.verbose:
                print(f'[EarlyStopping] AP improved: {self.best_score:.6f} → {score:.6f}')
            self.best_score = score
            self._save_checkpoint(score)
            self.counter = 0

    def _save_checkpoint(self, score):
        if self.verbose:
            print(f'[EarlyStopping] Saving best model (score: {score:.6f})')
        if self.save_fn:
            self.save_fn()
        self.score_max = score


def adjust_learning_rate(optimizer, decay_factor=0.1, min_lr=1e-9):
    stop_training = False
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = param_group['lr']
        new_lr = max(old_lr * decay_factor, min_lr)
        param_group['lr'] = new_lr
        if old_lr > min_lr and new_lr <= min_lr:
            stop_training = True
        print(f"[LR] Adjusting lr: {old_lr:.6e} → {new_lr:.6e}")
    return not stop_training