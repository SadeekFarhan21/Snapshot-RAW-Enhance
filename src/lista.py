"""LISTA — Learned Iterative Shrinkage Thresholding (deep-unfolded ISTA).

Each ISTA iteration is x_{k+1} = soft_threshold( x_k + W_e (y - A x_k), theta_k )
where, originally,  W_e = (1/L) A^T  and  theta_k = lam/L.

LISTA replaces W_e (and optionally the recurrent matrix W_t = I - (1/L)A^T A)
with learnable matrices, and replaces the scalar threshold theta_k with a
learnable per-layer threshold. K layers are unrolled and trained end-to-end
by backpropagating an MSE loss on the recovered signal.

References: Gregor & LeCun, ICML 2010.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def soft_threshold_torch(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - t, min=0.0)


class LISTA(nn.Module):
    """Tied-weight LISTA with K unfolded layers and a learnable threshold per layer.

    Tied weights (W_e and W_t shared across layers, only thresholds vary) cut
    the parameter count and tend to generalize better when the sensing matrix
    is fixed. The forward call returns the recovered signal (B x N).
    """

    def __init__(self, A: torch.Tensor, n_layers: int = 10, lam_init: float = 0.05):
        super().__init__()
        self.n_layers = n_layers
        M, N = A.shape
        # Lipschitz constant for normalized A
        with torch.no_grad():
            L = torch.linalg.matrix_norm(A.T @ A, ord=2).item()
        step = 1.0 / max(L, 1e-8)

        # Initialize at the classical ISTA values
        self.W_e = nn.Parameter(step * A.T.clone())            # N x M
        self.W_t = nn.Parameter(torch.eye(N) - step * (A.T @ A))  # N x N
        self.theta = nn.Parameter(torch.full((n_layers,), step * lam_init))

        self.register_buffer("A", A.clone())

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, M)
        B = y.shape[0]
        N = self.W_e.shape[0]
        x = torch.zeros(B, N, device=y.device, dtype=y.dtype)
        for k in range(self.n_layers):
            x = soft_threshold_torch(self.W_t @ x.T + self.W_e @ y.T, self.theta[k]).T
        return x


def train_lista(
    A_np,
    train_pairs,
    val_pairs,
    n_layers: int = 10,
    n_epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    grad_clip: float = 1.0,
    verbose: bool = True,
):
    """Train LISTA on (y, x) pairs from a fixed sensing matrix A.

    train_pairs / val_pairs are tuples (Y, X) of numpy arrays of shape
    (n_examples, M) and (n_examples, N).

    Tracks the best validation NMSE seen so far and restores those weights at
    the end — necessary because the K-layer recurrence x_{k+1} = soft(W_t x_k
    + W_e y, theta) can have its spectral radius drift above 1 under sustained
    training, blowing up the iterate. Gradient clipping plus best-checkpoint
    selection neutralizes that.
    """
    import copy

    import numpy as np

    A = torch.tensor(A_np, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(train_pairs[0], dtype=torch.float32, device=device)
    X_tr = torch.tensor(train_pairs[1], dtype=torch.float32, device=device)
    Y_va = torch.tensor(val_pairs[0], dtype=torch.float32, device=device)
    X_va = torch.tensor(val_pairs[1], dtype=torch.float32, device=device)

    model = LISTA(A, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_nmse": []}
    n_train = X_tr.shape[0]
    best_nmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            y_b = Y_tr[idx]
            x_b = X_tr[idx]
            x_hat = model(y_b)
            loss = ((x_hat - x_b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            x_hat_va = model(Y_va)
            num = ((x_hat_va - X_va) ** 2).sum(dim=1)
            den = (X_va ** 2).sum(dim=1) + 1e-12
            val_nmse = float((num / den).mean().item())
        history["train_loss"].append(train_loss)
        history["val_nmse"].append(val_nmse)

        if val_nmse < best_nmse:
            best_nmse = val_nmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d}  train_mse={train_loss:.5f}  val_NMSE={val_nmse:.5f}")

    model.load_state_dict(best_state)
    if verbose:
        print(f"  restored best epoch {best_epoch} with val_NMSE={best_nmse:.5f}")
    history["best_epoch"] = best_epoch
    history["best_val_nmse"] = best_nmse
    return model, history
