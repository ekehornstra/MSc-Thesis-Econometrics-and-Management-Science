"""codine.py

CODINE (Copula Density Neural Estimation; Letizia, Novello & Tonello, 2025) for an unconditional bivariate copula.

The copula density is the density ratio between the pseudo-observations and independent uniforms. A network is
trained on a variational lower bound of the KL divergence between the two (generator f(u) = u log u, so that
f*(t) = exp(t - 1)), and its output is the density estimate itself:

- Architecture: fully connected, two hidden layers of 100 neurons, LeakyReLU (negative slope 0.2), dropout 0.1
  after each hidden layer, Softplus output. This matches the official PyTorch implementation of Letizia et al.
  (github.com/tonellolab/CODINE-copula-estimator). Softplus grows only linearly for large inputs, so the output
  cannot explode the way an unclamped exp(T - 1) can.
- Optimiser: Adam (Kingma & Ba, 2014).
- Normalisation: the density is divided by its Monte Carlo mean over uniform points (calibrate_normalization).
- Self-consistency test: the Monte Carlo mean of the normalised density over uniform points should be 1
  (self_consistency_test).
- Samples from the fitted density come from a grid sampler (sample_from_codine).

The second half of the module is the conditional version, C-CODINE: CODINE extended to condition on a continuous
covariate w, so that the estimated copula density is c(u | w) instead of c(u). The network takes w as an additional
input (three ways of injecting it, see ConditionalCopulaNet) and has the same width, depth, activations, dropout and
Softplus output as the unconditional network. The f-divergence bound is estimated per batch by pairing each real
(u, w) with an independent uniform u that carries the same w, so that the objective targets the conditional density.
The normalising constant is a function Z(w), estimated by Monte Carlo on a grid of w values and interpolated
(calibrate_normalization_conditional). It also holds the evaluation helpers (self-consistency test, sampler,
Kendall's tau curve) and a Student-t copula simulator with covariate-varying correlation for the synthetic recovery
checks.

References:
Dumoulin, V., Perez, E., Schucher, N., Strub, F., de Vries, H., Courville, A., & Bengio, Y. (2018).
    Feature-wise transformations. Distill, 3(7), e11.
Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
Letizia, N. A., Novello, N., & Tonello, A. M. (2025). Copula density neural estimation. IEEE Transactions on
    Neural Networks and Learning Systems, 36(10), 19452-19459.
Masters, D., & Luschi, C. (2018). Revisiting small batch training for deep neural networks. arXiv:1804.07612.
Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general
    conditioning layer. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import kendalltau
from tqdm.auto import tqdm

from copula_families import QUAD_N, _spline_basis_w, expectations_pair
from data_split import train_test_split_chrono
from run_support import get_logger

log = get_logger()

# Numerical floor inside log(): the Softplus output is positive but can underflow to zero early in training.
LOG_EPS = 1e-6


# ── network ──────────────────────────────────────────────────────────────────
class CopulaNet(nn.Module):
    """Density network of CODINE: two hidden layers of 100 units with a Softplus output.

    Each hidden layer is followed by LeakyReLU (negative slope 0.2) and dropout. The dropout of the official
    implementation is kept: that code draws a fresh synthetic batch at every step, whereas this network is fitted to
    one finite dataset, so dropout also guards against overfitting there, next to early stopping on a held-out
    validation block (train_codine_early_stop).

    Input: d-dimensional pseudo-observations u in [0, 1]^d.
    Output: D_theta(u) = f*(T_theta(u)), the unnormalised copula density, directly. The Softplus makes D positive
    everywhere.
    """

    def __init__(self, d: int = 2, hidden: int = 100, dropout: float = 0.1):
        """Construct the network.

        Parameters:
        d (int): Input dimensionality. Default 2.
        hidden (int): Hidden layer width. Default 100.
        dropout (float): Dropout probability applied after each hidden layer. Default 0.1.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the network output D_theta(u) for a batch.

        Parameters:
        u (torch.Tensor): (batch, d) input pseudo-observations.

        Returns:
        torch.Tensor: (batch,) positive outputs.
        """
        return self.net(u).squeeze(-1)


# ── objective ────────────────────────────────────────────────────────────────
def codine_loss(net: CopulaNet,
                u_copula: torch.Tensor,
                u_uniform: torch.Tensor) -> torch.Tensor:
    """Negative variational lower bound of the KL divergence, written in density space.

    The bound is J_f(T) = E_{u~c}[T(u)] - E_{u~pi}[f*(T(u))] (Letizia et al., 2025). With D = f*(T) = exp(T - 1), which
    is what the network outputs, it becomes E_{u~c}[log D(u)] - E_{u~pi}[D(u)] + 1. The constant is dropped, and the
    negative is returned so that gradient descent maximises the bound. This is the combination of the binary
    cross-entropy and Wasserstein terms of the official implementation.

    Parameters:
    net (CopulaNet): The network being trained.
    u_copula (torch.Tensor): Batch of pseudo-observations from the data.
    u_uniform (torch.Tensor): Batch of independent uniform samples.

    Returns:
    torch.Tensor: Scalar loss to minimise.
    """
    d_copula  = net(u_copula)
    d_uniform = net(u_uniform)
    loss = -torch.log(d_copula + LOG_EPS).mean() + d_uniform.mean()
    return loss

# ── validation CVPE ──────────────────────────────────────────────────────────
def _codine_cvpe(net: CopulaNet, u1_val: np.ndarray, u2_val: np.ndarray,
                  grid_size: int = QUAD_N) -> float:
    """CVPE of the network's density on a held-out split.

    Computed like the CVPE of the other models (Gauss-Legendre integration of the density with
    copula_families.expectations_pair), so that early stopping selects for the quantity the models are compared on.
    The ratio of the two integrals renormalises the density locally, so the value does not depend on a global scale
    factor and the network does not need to be calibrated first.

    Parameters:
    net (CopulaNet): Network being evaluated.
    u1_val (np.ndarray): First variable's held-out pseudo-observations.
    u2_val (np.ndarray): Second variable's held-out pseudo-observations.
    grid_size (int): Number of Gauss-Legendre nodes. Default copula_families.QUAD_N.

    Returns:
    float: Mean over points of (u1 - E[u1 | u2])^2 + (u2 - E[u2 | u1])^2, the same scale as
        model_evaluation.cvpe_only, so the early-stopping threshold of 1e-6 means the same for all networks.
    """
    dens = lambda pts, _w: eval_density(net, pts)
    e1, e2 = expectations_pair(dens, np.asarray(u1_val, float), np.asarray(u2_val, float), n=grid_size)
    return float(np.mean((np.asarray(u1_val) - e1) ** 2 + (np.asarray(u2_val) - e2) ** 2))


# ── training ─────────────────────────────────────────────────────────────────
def train_codine(u1, u2, n_epochs=2000, batch_size=32, lr=1e-3,
                 seed=42, init_net=None,
                 val_u1=None, val_u2=None, patience=None,
                 checkpoint_every=100, cvpe_grid_size=QUAD_N, show_progress=True, calib_seed=0,
                 weight_decay=0.0, grad_clip_norm=None):
    """Train a CopulaNet on pseudo-observations with the CODINE objective, optionally with early stopping.

    Training runs in epochs: the indices are shuffled once per epoch and the network is updated over minibatches of
    pseudo-observations, each paired with a batch of independent uniforms. If validation data and a patience are
    given, the validation CVPE (_codine_cvpe) is evaluated every checkpoint_every epochs, the best checkpoint is
    kept, and training stops after patience checkpoints without an improvement of more than 1e-6. The CVPE does not
    depend on the density scale, so the network is calibrated (calibrate_normalization) only once, at the end.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations.
    u2 (np.ndarray): Second variable's pseudo-observations.
    n_epochs (int): Maximum number of epochs. Default 2000.
    batch_size (int): Minibatch size. Default 32, the upper end of the range m = 2 to 32 in which Masters and
        Luschi (2018) consistently find the best performance.
    lr (float): Adam learning rate. Default 1e-3.
    seed (int): Torch and numpy seed. Default 42.
    init_net (CopulaNet): Warm-start network, deep-copied and fine-tuned at lr * 0.1. If None, a new network is
        trained from scratch. Optional.
    val_u1 (np.ndarray): First variable's validation pseudo-observations. Optional.
    val_u2 (np.ndarray): Second variable's validation pseudo-observations. Optional.
    patience (int): Number of consecutive non-improving checkpoints tolerated. Early stopping needs val_u1, val_u2
        and patience; with any of them missing, all n_epochs are run. Default None.
    checkpoint_every (int): Epochs between validation checks. Default 100.
    cvpe_grid_size (int): Number of Gauss-Legendre nodes of the validation CVPE. Default copula_families.QUAD_N.
    show_progress (bool): Show the tqdm epoch bar. Default True.
    calib_seed (int): Seed of the Monte Carlo normalisation at the end of training. Default 0.
    weight_decay (float): Adam weight decay. Default 0.0, the setting of the official implementation.
    grad_clip_norm (float): Maximum L2 norm of the gradient per step. If None, no clipping. Optional.

    Returns:
    tuple: (net, losses, val_cvpe, best_epoch) with early stopping, where val_cvpe maps checkpoint epoch to
        validation CVPE and best_epoch is the epoch of the kept checkpoint; otherwise (net, losses). The returned
        net is normalisation-calibrated with 100,000 draws.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(u1)
    U = torch.tensor(np.column_stack([u1, u2]), dtype=torch.float32)

    use_early_stop = (val_u1 is not None and val_u2 is not None
                       and patience is not None)
    if use_early_stop:
        checkpoints = {}
        val_cvpe = {}
        best_val_cvpe, best_epoch, epochs_since_improve = float('inf'), None, 0

    if init_net is not None:
        net = copy.deepcopy(init_net)          # warm start
        actual_lr = lr * 0.1                   # lower learning rate for fine-tuning
    else:
        net = CopulaNet(d=2, hidden=100)
        actual_lr = lr

    optimizer = optim.Adam(net.parameters(), lr=actual_lr,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    net.train()
    losses = []

    pbar = tqdm(range(1, n_epochs + 1), desc='Training CODINE', disable=not show_progress)
    for epoch in pbar:
        idx = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, n, batch_size):
            batch_idx   = idx[start:start + batch_size]
            u_batch     = U[batch_idx]
            u_uni_batch = torch.rand_like(u_batch)
            optimizer.zero_grad()
            loss = codine_loss(net, u_batch, u_uni_batch)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        losses.append(epoch_loss / n_batches)

        if use_early_stop and (epoch % checkpoint_every == 0 or epoch == n_epochs):
            cvpe_here = _codine_cvpe(net, val_u1, val_u2, grid_size=cvpe_grid_size)
            val_cvpe[epoch] = cvpe_here
            if cvpe_here < best_val_cvpe - 1e-6:
                best_val_cvpe, best_epoch, epochs_since_improve = cvpe_here, epoch, 0
                # Only the best weights are kept.
                checkpoints.clear()
                checkpoints[epoch] = copy.deepcopy(net.state_dict())
            else:
                epochs_since_improve += 1
            pbar.set_postfix(loss=f"{losses[-1]:.4f}", val_cvpe=f"{cvpe_here:.5f}")
            if epochs_since_improve >= patience:
                if show_progress:
                    log.info(f"[early_stop] no improvement for {patience} checkpoints "
                          f"({patience * checkpoint_every} epochs), stopping at "
                          f"epoch {epoch} (best={best_epoch}, val_CVPE={best_val_cvpe:.5f})")
                net.train()
                break
            net.train()

    if use_early_stop:
        if best_epoch is None:
            raise RuntimeError("early stopping found no finite validation CVPE at any checkpoint "
                               "(best_epoch is None): the run diverged or the density collapsed")
        net.load_state_dict(checkpoints[best_epoch])

    net.eval()
    calibrate_normalization(net, seed=calib_seed)  # n_mc = 100,000, with and without early stopping

    if use_early_stop:
        return net, losses, val_cvpe, best_epoch
    return net, losses


def train_codine_early_stop(u1, u2, b=None, val_frac=0.25, max_epochs=8000,
                             checkpoint_every=100, patience=5, seed=42,
                             split_mode='chronological', regime_mask=None,
                             cvpe_grid_size=QUAD_N, verbose=True, **train_kwargs):
    """Train a CopulaNet with early stopping on the CVPE of a held-out validation block.

    The same procedure as train_conditional_codine_early_stop (below) and
    train_conditional_copula_flow_early_stop (conditional_copula_flow.py), so that all neural models are trained
    alike. The validation block is carved out of the training series by split_mode:

    - 'chronological': purge-gap chronological split (data_split.train_test_split_chrono) of a contiguous block.
    - 'regime_chronological': u1 and u2 must be the full training-portion series and regime_mask marks the regime
      members. The full timeline is split chronologically with a purge gap first and then intersected with
      regime_mask, so that the purge separates real time neighbours and not neighbours in the masked array.
    - 'random': i.i.d. random validation split without purge gap, for synthetic data without temporal structure.
      Not suitable for real data: observations close in time can land on opposite sides of the boundary, which
      makes the validation CVPE look better than out-of-sample performance.

    Parameters:
    u1 (np.ndarray): First variable's training-portion pseudo-observations (never the held-out test set). For
        'regime_chronological', the full training-portion series before the regime mask is applied.
    u2 (np.ndarray): Second variable's training-portion pseudo-observations, same restrictions as u1.
    b (float): Purge gap length. Required for 'chronological' and 'regime_chronological' (pass b=1 for i.i.d.
        synthetic data), ignored for 'random'. Optional.
    val_frac (float): Fraction reserved for validation. Default 0.25.
    max_epochs (int): Maximum number of epochs. Default 8000.
    checkpoint_every (int): Epochs between validation checks. Default 100.
    patience (int): Number of consecutive non-improving checkpoints tolerated. Default 5.
    seed (int): Training and split seed. Default 42.
    split_mode (str): 'chronological', 'regime_chronological' or 'random'. Default 'chronological'.
    regime_mask (np.ndarray): Boolean mask of regime members, same length as u1. Required for
        'regime_chronological'. Optional.
    cvpe_grid_size (int): Number of Gauss-Legendre nodes of the validation CVPE. Default copula_families.QUAD_N.
    verbose (bool): Log a one-line summary of the stopping epoch. Default True.
    **train_kwargs: Forwarded to train_codine (batch_size, lr, init_net, weight_decay, grad_clip_norm, calib_seed).

    Returns:
    dict: Keys net (best checkpoint, normalisation-calibrated), losses (training loss up to the stopping epoch),
        val_cvpe (checkpoint epoch to validation CVPE), best_epoch, best_val_cvpe, sub_train_mask and val_mask
        (aligned with the series actually trained on, which for 'regime_chronological' is u1[regime_mask]).
    """
    if split_mode not in ('chronological', 'regime_chronological', 'random'):
        raise ValueError(
            f"split_mode must be 'chronological', 'regime_chronological' "
            f"or 'random', got {split_mode!r}")

    if split_mode == 'regime_chronological':
        if regime_mask is None:
            raise ValueError("split_mode='regime_chronological' requires regime_mask.")
        if b is None:
            raise ValueError(
                "b is required when split_mode='regime_chronological' "
                "(pass b=1 explicitly for i.i.d. synthetic data with no "
                "temporal structure -- though 'random' is simpler for "
                "that case).")
        n_full = len(u1)
        sub_train_full, val_full = train_test_split_chrono(
            n_full, test_frac=val_frac, purge=int(round(b)), one_sided=True)
        sub_train_mask = (sub_train_full & regime_mask)[regime_mask]
        val_mask = (val_full & regime_mask)[regime_mask]
        u1, u2 = u1[regime_mask], u2[regime_mask]
        n = len(u1)
    elif split_mode == 'chronological':
        if b is None:
            raise ValueError(
                "b is required when split_mode='chronological' (pass b=1 "
                "explicitly for i.i.d. synthetic data with no temporal "
                "structure). For regime-masked, non-contiguous data, use "
                "split_mode='regime_chronological' instead."
            )
        n = len(u1)
        sub_train_mask, val_mask = train_test_split_chrono(n, test_frac=val_frac, purge=int(round(b)), one_sided=True)
    else:
        n = len(u1)
        rng = np.random.default_rng(seed)
        val_mask = np.zeros(n, dtype=bool)
        val_idx = rng.choice(n, size=int(round(val_frac * n)), replace=False)
        val_mask[val_idx] = True
        sub_train_mask = ~val_mask

    u1_sub, u2_sub = u1[sub_train_mask], u2[sub_train_mask]
    u1_val, u2_val = u1[val_mask], u2[val_mask]

    net, losses, val_cvpe, best_epoch = train_codine(
        u1_sub, u2_sub, n_epochs=max_epochs, seed=seed,
        val_u1=u1_val, val_u2=u2_val, patience=patience,
        checkpoint_every=checkpoint_every,
        cvpe_grid_size=cvpe_grid_size, show_progress=not verbose,
        **train_kwargs)

    if best_epoch is None:
        raise RuntimeError(
            "early stopping found no finite validation CVPE at any checkpoint (best_epoch is None): the "
            "validation CVPE was NaN/inf, the run diverged or the density collapsed. Nothing is returned "
            "rather than a network picked without a criterion.")
    best_val_cvpe = val_cvpe[best_epoch]

    if verbose:
        log.info(f"[early_stop] best epoch={best_epoch}  val_CVPE={best_val_cvpe:.5f}  "
              f"(stopped at {max(val_cvpe.keys())}/{max_epochs})")

    return {
        'net': net, 'losses': losses, 'val_cvpe': val_cvpe,
        'best_epoch': best_epoch, 'best_val_cvpe': best_val_cvpe,
        'sub_train_mask': sub_train_mask, 'val_mask': val_mask,
    }


# ── density, normalisation and self-consistency ──────────────────────────────
def eval_density(net: CopulaNet,
                 u: np.ndarray) -> np.ndarray:
    """Evaluate the estimated copula density at points u.

    The estimate is D_theta(u) divided by net._norm_const (see calibrate_normalization), so that its mean over
    uniform points is 1. A network that has not been calibrated is divided by 1.

    Parameters:
    net (CopulaNet): A trained (and normalisation-calibrated) network.
    u (np.ndarray): (n, 2) points in [0, 1]^2.

    Returns:
    np.ndarray: (n,) density values.
    """
    net.eval()
    with torch.no_grad():
        u_t  = torch.tensor(u, dtype=torch.float32)
        c_hat = net(u_t).numpy()
    norm_const = getattr(net, '_norm_const', 1.0)
    return c_hat / norm_const

def calibrate_normalization(net: CopulaNet, n_mc: int = 100_000, seed: int = 0) -> float:
    """Estimate the mean of the network output over uniform points and store it as the normalising constant.

    The constant is stored as net._norm_const, which eval_density divides out. The network output is used directly
    (not eval_density), because the constant is what is being computed.

    Parameters:
    net (CopulaNet): The network to calibrate (modified in place).
    n_mc (int): Number of uniform Monte Carlo points. Default 100000.
    seed (int): Simulation seed. Default 0.

    Returns:
    float: The normalising constant, also stored on net.
    """
    net.eval()
    rng = np.random.default_rng(seed)
    u_mc = rng.uniform(0, 1, size=(n_mc, 2)).astype(np.float32)
    with torch.no_grad():
        u_t = torch.tensor(u_mc, dtype=torch.float32)
        c_hat = net(u_t).numpy()
    net._norm_const = float(c_hat.mean())
    return net._norm_const


def self_consistency_test(net: CopulaNet, n_mc: int = 10_000, n_seeds: int = 10,
                          seed: int = 0, tol: float = 0.01) -> dict:
    """Check that the normalised density of a calibrated CODINE network integrates to 1 over the unit square.

    The Monte Carlo mean of the normalised density over uniform points is computed for n_seeds independent samples of
    n_mc points. Use seed >= 1: seed 0 with 100,000 points is the calibration sample, and the check should be
    independent of it. Pass or fail is decided on the effect size |mean - 1| <= tol, as in
    self_consistency_test_conditional. The standard error and the z-score are diagnostics only.

    Parameters:
    net (CopulaNet): A calibrated network.
    n_mc (int): Number of uniform points per Monte Carlo integral. Default 10000.
    n_seeds (int): Number of independent integrals. Default 10.
    seed (int): First seed; the integrals use seed, seed + 1, ..., seed + n_seeds - 1. Default 0.
    tol (float): Tolerance on |mean - 1|. Default 0.01.

    Returns:
    dict: Keys integral (mean over seeds), se, z, deviation (|integral - 1|), tol and passed.
    """
    integrals = []
    for s in range(seed, seed + n_seeds):
        rng = np.random.default_rng(s)
        u_mc = rng.uniform(0, 1, size=(n_mc, 2)).astype(np.float32)
        integrals.append(float(eval_density(net, u_mc).mean()))
    integrals = np.array(integrals)
    mean_integral = float(integrals.mean())
    se = float(integrals.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else np.nan
    z = float((mean_integral - 1.0) / se) if se and se > 0 else np.inf
    deviation = abs(mean_integral - 1.0)
    return {"integral": mean_integral, "se": se, "z": z, "deviation": deviation,
            "tol": tol, "passed": bool(deviation <= tol)}


# ── sampling and plotting ────────────────────────────────────────────────────
def sample_from_codine(net: CopulaNet,
                        n_samples: int = 50_000,
                        grid_size: int = 200,
                        seed: int = 1) -> np.ndarray:
    """Draw samples from the estimated copula density with a grid sampler.

    The density is evaluated on a grid_size x grid_size grid, normalised to a discrete probability mass, and grid
    cells are drawn in proportion to their mass. Uniform jitter of one cell width is added and the result is clipped
    to [0.001, 0.999].

    Parameters:
    net (CopulaNet): A trained (and normalisation-calibrated) network.
    n_samples (int): Number of samples. Default 50000.
    grid_size (int): Number of grid points per dimension. Default 200.
    seed (int): Seed of a local numpy generator; numpy's global generator is not touched. Default 1.

    Returns:
    np.ndarray: (n_samples, 2) array of sampled pairs.
    """
    rng = np.random.default_rng(seed)
    g   = np.linspace(0.005, 0.995, grid_size)
    uu, vv = np.meshgrid(g, g)
    pts = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float32)

    density = eval_density(net, pts)
    probs   = density / density.sum()

    # Sample cell indices
    cell_idx = rng.choice(len(probs), size=n_samples, p=probs)
    cell_u   = pts[cell_idx, 0]
    cell_v   = pts[cell_idx, 1]

    # Uniform jitter within each cell
    step  = g[1] - g[0]
    jitter_u = rng.uniform(-step / 2, step / 2, n_samples)
    jitter_v = rng.uniform(-step / 2, step / 2, n_samples)

    samples = np.column_stack([
        np.clip(cell_u + jitter_u, 0.001, 0.999),
        np.clip(cell_v + jitter_v, 0.001, 0.999),
    ])
    return samples


def plot_training_loss(losses: list, ax=None, conditional: bool = False) -> None:
    """Plot the negative variational lower bound over the training epochs.

    Parameters:
    losses (list): Per-epoch mean training losses, as returned by train_codine or train_conditional_codine.
    ax (matplotlib.axes.Axes): Axes to plot on. If None, a new figure is created. Optional.
    conditional (bool): True for the losses of a conditional network (changes the title only). Default False.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses, lw=1, color='grey')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("−J_f  (loss)")
    ax.set_title("Conditional CODINE training loss" if conditional else "CODINE training loss")


# ══════════════════════════════════════════════════════════════════════════════
# Conditional CODINE (C-CODINE)
# ══════════════════════════════════════════════════════════════════════════════
# ── network ──────────────────────────────────────────────────────────────────
class ConditionalCopulaNet(nn.Module):
    """Density network D_theta(u, w) of C-CODINE, with the covariate w injected in one of three ways.

    The width, depth, LeakyReLU, dropout and Softplus output are those of the unconditional CopulaNet.
    The conditioning modes are the same as in the C-CopFlow couplings (conditional_copula_flow.py):

    - 'w_only': the standardised w is concatenated with u (d + 1 inputs). Concatenation is equivalent to a
      w-dependent bias on the first layer (Dumoulin et al., 2018).
    - 'spline_basis': w is replaced by a fixed cubic truncated-power spline expansion (copula_families._spline_basis_w)
      before concatenation with u.
    - 'film': u alone feeds the main network. A small linear map from w gives a gain and a shift that rescale and
      shift the activations of the first hidden layer (FiLM; Perez et al., 2018), with gain = 1 + gamma_raw, so a unit
      can be damped, amplified or negated. The map is initialised at zero, so the modulation starts as the identity.

    Input: (u, w) with u in [0, 1]^d and w a real number, standardised internally.
    Output: D_theta(u, w), the unnormalised copula density, directly (Softplus output, so D > 0).
    """

    def __init__(self, d: int = 2, hidden: int = 100,
                 conditioning_mode: str = 'w_only', spline_knots: torch.Tensor = None,
                 dropout: float = 0.1):
        """Build the network and initialise the standardisation and calibration state.

        Parameters:
        d (int): Dimensionality of u. Default 2.
        hidden (int): Hidden layer width. Default 100.
        conditioning_mode (str): 'w_only', 'spline_basis' or 'film'. Default 'w_only'.
        spline_knots (torch.Tensor): Fixed knot positions for _spline_basis_w, registered as a buffer (part of
            state_dict). Required if conditioning_mode is 'spline_basis'. Optional otherwise.
        dropout (float): Dropout rate after each hidden layer, for all three modes. In 'film' it is implemented as
            explicit nn.Dropout modules, because that forward pass is written out by hand. Default 0.1.
        """
        super().__init__()
        if conditioning_mode not in ('w_only', 'spline_basis', 'film'):
            raise ValueError(f"conditioning_mode must be 'w_only', 'spline_basis', "
                              f"or 'film', got {conditioning_mode!r}")
        self.d = d
        self.conditioning_mode = conditioning_mode

        if conditioning_mode == 'spline_basis':
            if spline_knots is None:
                raise ValueError("conditioning_mode='spline_basis' requires spline_knots")
            self.register_buffer('spline_knots', spline_knots)

        if conditioning_mode == 'film':
            self.linear1 = nn.Linear(d, hidden)
            self.linear2 = nn.Linear(hidden, hidden)
            self.linear3 = nn.Linear(hidden, 1)
            self.film_net = nn.Linear(1, 2 * hidden)
            nn.init.zeros_(self.film_net.weight)
            nn.init.zeros_(self.film_net.bias)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        else:
            in_dim = (d + 4 + len(spline_knots)) if conditioning_mode == 'spline_basis' else d + 1
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
                nn.Softplus(),
            )

        # set by train_conditional_codine
        self._w_mean = 0.0
        self._w_std = 1.0
        # set by calibrate_normalization_conditional
        self._w_grid = None
        self._norm_grid = None

    def forward(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute the network output D_theta(u, w) for a batch.

        Parameters:
        u (torch.Tensor): (batch, d) pseudo-observations.
        w (torch.Tensor): (batch,) or (batch, 1) covariate values on the raw scale (standardised internally with
            net._w_mean and net._w_std).

        Returns:
        torch.Tensor: (batch,) positive outputs.
        """
        w_std = (w - self._w_mean) / self._w_std
        if w_std.dim() == 2:
            w_std = w_std.squeeze(-1)

        if self.conditioning_mode == 'film':
            h = F.leaky_relu(self.linear1(u), negative_slope=0.2)
            h = self.dropout1(h)
            gamma_raw, beta = self.film_net(w_std.unsqueeze(-1)).chunk(2, dim=-1)
            gamma = 1.0 + gamma_raw   # unbounded: can damp, amplify or negate a unit
            h = gamma * h + beta
            h = F.leaky_relu(self.linear2(h), negative_slope=0.2)
            h = self.dropout2(h)
            return F.softplus(self.linear3(h)).squeeze(-1)
        else:
            if self.conditioning_mode == 'spline_basis':
                w_feat = _spline_basis_w(w_std, self.spline_knots)
            else:
                w_feat = w_std.unsqueeze(-1)
            x = torch.cat([u, w_feat], dim=-1)
            return self.net(x).squeeze(-1)


# ── objective ────────────────────────────────────────────────────────────────
def conditional_codine_loss(net: ConditionalCopulaNet,
                             u_copula: torch.Tensor,
                             w_copula: torch.Tensor,
                             u_uniform: torch.Tensor) -> torch.Tensor:
    """Negative variational lower bound of the KL divergence, averaged over w from the data batch, in density space.

    J_f = E_w [ E_{u~c(u|w)}[log D(u, w)] - E_{u~Unif}[D(u, w)] ] + 1, the conditional version of codine_loss.
    Each real pseudo-observation (u_copula[i], w_copula[i]) is paired with an independent uniform draw u_uniform[i]
    that shares w_copula[i]. This makes the bound target the conditional density and not a w-marginalised average.
    The constant is dropped and the negative is returned, so gradient descent maximises the bound.

    Parameters:
    net (ConditionalCopulaNet): The network being trained.
    u_copula (torch.Tensor): Batch of pseudo-observations from the data.
    w_copula (torch.Tensor): Covariate values paired with u_copula.
    u_uniform (torch.Tensor): Batch of independent uniform samples, one per row of u_copula.

    Returns:
    torch.Tensor: Scalar loss to minimise.
    """
    d_copula = net(u_copula, w_copula)
    d_uniform = net(u_uniform, w_copula)   # same w as the paired real sample
    loss = -torch.log(d_copula + LOG_EPS).mean() + d_uniform.mean()
    return loss


# ── marginal-uniformity penalty (Aich & Aich, 2025, eq. 1) ───────────────────────────────────────────────
def margin_uniformity_penalty(density_fn, w, n_grid=20, normalise=True):
    """Penalty on the departure of both margins of a conditional copula density from uniformity.

    For every covariate value in w, the density is evaluated on an n_grid x n_grid grid of randomly shifted midpoints in the unit square (independent shifts along each axis, so the density-grid
    average is an unbiased estimate of its integral). The margins are the grid averages over the other
    coordinate. The penalty is the mean over w and over the grid of the squared deviation of the margin from 1, summed over
    the two margins: the sample version of sum_i int_0^1 (int c(u) du_{-i} - 1)^2 du_i of Aich and Aich (2025, eq. 1), applied at each w.

    Aich, A., & Aich, A. B. (2025). Deep Copula Classifier: Theory, consistency, and empirical evaluation (arXiv:2505.22997v3).

    Parameters:
    density_fn (callable): density_fn(u, w) -> tensor of densities, with u of shape (m, 2) and w of shape (m,), differentiable in the
        network parameters.
    w (torch.Tensor): Covariate values, shape (nw,), on the raw scale.
    n_grid (int): Grid points per coordinate. Default 20.
    normalise (bool): Divide each density by its grid average over the square, so that the penalty does not depend on the scale of an
        unnormalised density (C-CODINE). Default True.

    Returns:
    torch.Tensor: Scalar penalty.
    """
    nw = int(w.shape[0])
    a = (torch.arange(n_grid, dtype=w.dtype, device=w.device) + torch.rand(1, dtype=w.dtype, device=w.device)) / n_grid
    b = (torch.arange(n_grid, dtype=w.dtype, device=w.device) + torch.rand(1, dtype=w.dtype, device=w.device)) / n_grid
    A, B = torch.meshgrid(a, b, indexing="ij")
    pts = torch.stack([A.reshape(-1), B.reshape(-1)], dim=-1)
    u = pts.unsqueeze(0).expand(nw, -1, -1).reshape(-1, 2)
    ww = w.unsqueeze(1).expand(nw, n_grid * n_grid).reshape(-1)
    d = density_fn(u, ww).reshape(nw, n_grid, n_grid)
    if normalise:
        d = d / d.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return ((d.mean(dim=2) - 1.0) ** 2).mean() + ((d.mean(dim=1) - 1.0) ** 2).mean()


# ── training ─────────────────────────────────────────────────────────────────
def train_conditional_codine(u1, u2, w, n_epochs=3000, batch_size=32, lr=1e-3,
                              seed=42, init_net=None, grad_clip_norm=5.0, weight_decay=0.0,
                              conditioning_mode='w_only', checkpoint_every=50,
                              val_u1=None, val_u2=None, val_w=None,
                              patience=None, grid_n=QUAD_N, calib_n_mc_search=10_000,
                              show_progress=True, position=None,
                              margin_penalty=0.0, margin_n_w=8, margin_grid=20, margin_every=1):
    """Train a ConditionalCopulaNet on the conditional CODINE objective, optionally with early stopping.

    Training runs in epochs over shuffled minibatches. The default n_epochs is higher than for the unconditional
    network, because the network has to learn a family of densities indexed by w. Without validation data, a copy
    of the weights is stored every checkpoint_every epochs, which shows where the held-out CVPE bottoms out in one run.
    With validation data and a patience, the validation CVPE (model_evaluation.cvpe_only) is computed at every
    checkpoint after a cheap calibration of Z(w) (calib_n_mc_search draws), only the best weights are kept, and
    training stops after patience checkpoints without an improvement of more than 1e-6.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    w (np.ndarray): Covariate, shape (n,), any real scale (standardised internally).
    n_epochs (int): Maximum number of epochs. Default 3000.
    batch_size (int): Minibatch size, as for the unconditional network (see train_codine). Default 32.
    lr (float): Adam learning rate. Default 1e-3.
    seed (int): Torch and numpy seed. Default 42.
    init_net (ConditionalCopulaNet): Warm-start network, deep-copied and fine-tuned at lr * 0.1. If None, a new
        network is trained from scratch. Optional.
    grad_clip_norm (float): Maximum L2 norm of the gradient per step. If None, no clipping. Default 5.0.
    weight_decay (float): Adam weight decay. Default 0.0.
    conditioning_mode (str): 'w_only', 'spline_basis' or 'film', see ConditionalCopulaNet. Default 'w_only'.
    checkpoint_every (int): Epochs between checkpoints (stored weights, or validation checks). Default 50.
    val_u1 (np.ndarray): First variable's validation pseudo-observations. Optional.
    val_u2 (np.ndarray): Second variable's validation pseudo-observations. Optional.
    val_w (np.ndarray): Validation covariate values. Optional.
    patience (int): Number of consecutive non-improving checkpoints tolerated. Early stopping needs val_u1, val_u2,
        val_w and patience; with any of them missing, all n_epochs are run. Default None.
    grid_n (int): Number of Gauss-Legendre nodes of the validation CVPE. Default copula_families.QUAD_N.
    calib_n_mc_search (int): Monte Carlo draws per grid point for the in-loop calibration of Z(w), used to rank
        checkpoints only (train_conditional_codine_early_stop recalibrates the winner at full precision). Default 10000.
    show_progress (bool): Show the tqdm epoch bar. Pass False when this runs in a ProcessPoolExecutor, because
        bars of concurrent processes interleave. Default True.
    position (int): Row of the tqdm bar, so that concurrent workers can draw to separate lines (best effort,
        not every Jupyter frontend redraws in place). Default None.
    margin_every (int): The penalty is added on every margin_every-th minibatch with its weight multiplied by margin_every (same expected
        strength, a fraction of the cost). Default 1.
    margin_penalty (float): Weight lambda of margin_uniformity_penalty, added to the loss of every minibatch. 0 turns it off (the
        objective of Letizia et al., 2025). Default 0.0.
    margin_n_w (int): Number of covariate values of the minibatch on which the penalty is evaluated. Default 8.
    margin_grid (int): Grid points per coordinate of the penalty. Default 20.

    Returns:
    tuple: (net, losses, checkpoints, val_cvpe, best_epoch). losses is the per-epoch mean training loss without the
        margin penalty. checkpoints maps epoch to state_dict (only the best
        weights with early stopping). val_cvpe maps epoch to validation CVPE and is empty, and best_epoch is None,
        if no validation data were given.
    """
    use_early_stop = (val_u1 is not None and val_u2 is not None
                       and val_w is not None and patience is not None)
    if use_early_stop:
        from model_evaluation import cvpe_only

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(u1)
    U = torch.tensor(np.column_stack([u1, u2]), dtype=torch.float32)
    W = torch.tensor(np.asarray(w, dtype=np.float32))

    if init_net is not None:
        net = copy.deepcopy(init_net)
        actual_lr = lr * 0.1
    else:
        w_mean = float(W.mean())
        w_std_scalar = float(W.std()) if float(W.std()) > 1e-8 else 1.0
        spline_knots = None
        if conditioning_mode == 'spline_basis':
            w_std_train = (W - w_mean) / w_std_scalar
            spline_knots = torch.quantile(w_std_train, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
        net = ConditionalCopulaNet(d=2, hidden=100, conditioning_mode=conditioning_mode,
                                    spline_knots=spline_knots)
        net._w_mean = w_mean
        net._w_std = w_std_scalar
        actual_lr = lr

    optimizer = optim.Adam(net.parameters(), lr=actual_lr,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    net.train()
    losses = []
    checkpoints = {}
    val_cvpe = {}
    best_epoch = None
    best_val_cvpe = float('inf')
    epochs_since_improve = 0
    w_grid_calib = (calibration_w_grid(w, val_w) if use_early_stop else None)

    pbar = tqdm(range(1, n_epochs + 1), desc='Training conditional CODINE',
                disable=not show_progress, position=position, leave=False)
    batch_step = 0
    for epoch in pbar:
        idx = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            u_batch = U[batch_idx]
            w_batch = W[batch_idx]
            u_uni_batch = torch.rand_like(u_batch)
            optimizer.zero_grad()
            loss = conditional_codine_loss(net, u_batch, w_batch, u_uni_batch)
            base_loss = loss.item()
            if margin_penalty > 0 and batch_step % margin_every == 0:
                loss = loss + margin_every * margin_penalty * margin_uniformity_penalty(net, w_batch[:margin_n_w], margin_grid, normalise=True)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_loss += base_loss
            n_batches += 1
            batch_step += 1
        losses.append(epoch_loss / n_batches)
        if epoch % max(1, n_epochs // 20) == 0:
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")

        if epoch % checkpoint_every == 0 or epoch == n_epochs:
            net.eval()
            # With early stopping only the current best weights are kept (every checkpoint's CVPE stays in
            # val_cvpe); without it, the weights of every checkpoint are stored.
            if not use_early_stop:
                checkpoints[epoch] = copy.deepcopy(net.state_dict())

            if use_early_stop:
                # Cheap calibration (calib_n_mc_search), only to rank checkpoints; the winner is
                # recalibrated at full precision in train_conditional_codine_early_stop.
                calibrate_normalization_conditional(net, w_grid_calib,
                                                      n_mc=calib_n_mc_search, seed=0)
                with torch.no_grad():
                    density_fn = lambda pts, wv: eval_conditional_density(net, pts, wv)
                    cvpe_here = cvpe_only(density_fn, val_u1, val_u2, val_w, grid_n=grid_n)
                val_cvpe[epoch] = cvpe_here
                if cvpe_here < best_val_cvpe - 1e-6:
                    best_val_cvpe, best_epoch, epochs_since_improve = cvpe_here, epoch, 0
                    checkpoints.clear()
                    checkpoints[epoch] = copy.deepcopy(net.state_dict())
                else:
                    epochs_since_improve += 1
                pbar.set_postfix(loss=f"{losses[-1]:.4f}", val_cvpe=f"{cvpe_here:.5f}")
                if epochs_since_improve >= patience:
                    if show_progress:
                        log.info(f"[early_stop] no improvement for {patience} checkpoints "
                              f"({patience * checkpoint_every} epochs), stopping at "
                              f"epoch {epoch} (best={best_epoch}, "
                              f"val_CVPE={best_val_cvpe:.5f})")
                    net.train()
                    break

            net.train()

    net.eval()
    return net, losses, checkpoints, val_cvpe, best_epoch


def train_conditional_codine_early_stop(u1, u2, w, b, val_frac=0.25, max_epochs=8000,
                                         checkpoint_every=100, seed=42, grid_n=QUAD_N,
                                         calib_n_mc_search=10_000, calib_n_mc_final=60_000,
                                         patience=5, verbose=True, calib_seed=0, **train_kwargs):
    """Train a ConditionalCopulaNet with early stopping on the CVPE of a held-out validation block.

    The validation block is carved out of u1, u2 and w with the purge-gap chronological split
    (data_split.train_test_split_chrono, purge = b), so it respects the same autocorrelation-driven purge as the
    rest of the pipeline. Never call this with test data mixed in. The checkpoint is chosen by CVPE
    (model_evaluation.cvpe_only), the quantity the models are compared on, and not by the training objective. After
    training, the best checkpoint is reloaded (training may run patience checkpoints past it) and Z(w) is recalibrated
    for it with calib_n_mc_final draws.

    torch.set_num_threads(1) is set at the start: the network is small enough that threading overhead exceeds the
    matrix-multiplication cost, and it makes the function safe to submit to a ProcessPoolExecutor.

    Parameters:
    u1 (np.ndarray): First variable's training-portion pseudo-observations only.
    u2 (np.ndarray): Second variable's training-portion pseudo-observations only.
    w (np.ndarray): Training-portion covariate values only.
    b (float): Purge gap length (pass b=1 for i.i.d. synthetic data).
    val_frac (float): Fraction reserved for validation. Default 0.25.
    max_epochs (int): Maximum number of epochs. Default 8000.
    checkpoint_every (int): Epochs between validation checks. Default 100.
    seed (int): Training seed. Default 42.
    grid_n (int): Number of Gauss-Legendre nodes of the validation CVPE. Default copula_families.QUAD_N.
    calib_n_mc_search (int): Monte Carlo draws per grid point of the in-loop calibration. Default 10000.
    calib_n_mc_final (int): Monte Carlo draws per grid point of the final calibration of the best checkpoint.
        Default 60000.
    patience (int): Number of consecutive non-improving checkpoints tolerated. Default 5.
    verbose (bool): Log a one-line summary when finished. Default True.
    calib_seed (int): Seed of the final calibration. Default 0.
    **train_kwargs: Forwarded to train_conditional_codine (batch_size, lr, weight_decay, conditioning_mode,
        grad_clip_norm, show_progress; show_progress defaults to False here).

    Returns:
    dict: Keys net (best checkpoint, calibrated), losses (training loss up to the stopping epoch), checkpoints (the
        best weights), val_cvpe (checkpoint epoch to validation CVPE), best_epoch, best_val_cvpe, sub_train_mask and
        val_mask (aligned with the u1, u2, w passed in).
    """
    torch.set_num_threads(1)
    train_kwargs.setdefault('show_progress', False)

    n = len(u1)
    sub_train_mask, val_mask = train_test_split_chrono(n, test_frac=val_frac, purge=int(round(b)), one_sided=True)
    u1_sub, u2_sub, w_sub = u1[sub_train_mask], u2[sub_train_mask], w[sub_train_mask]
    u1_val, u2_val, w_val = u1[val_mask], u2[val_mask], w[val_mask]

    net, losses, checkpoints, val_cvpe, best_epoch = train_conditional_codine(
        u1_sub, u2_sub, w_sub, n_epochs=max_epochs, seed=seed,
        checkpoint_every=checkpoint_every,
        val_u1=u1_val, val_u2=u2_val, val_w=w_val,
        patience=patience, grid_n=grid_n, calib_n_mc_search=calib_n_mc_search,
        **train_kwargs)

    if best_epoch is None:
        raise RuntimeError(
            "early stopping found no finite validation CVPE at any checkpoint (best_epoch is None): the "
            "validation CVPE was NaN/inf, the run diverged or the density collapsed. Nothing is returned "
            "rather than a network picked without a criterion.")
    best_val_cvpe = val_cvpe[best_epoch]

    conditioning_mode = train_kwargs.get('conditioning_mode', 'w_only')
    spline_knots = getattr(net, 'spline_knots', None)
    w_grid_calib = calibration_w_grid(w_sub, w_val)

    # Reload the best checkpoint (training may have run patience checkpoints past it) and recalibrate it.
    best_net = ConditionalCopulaNet(d=2, hidden=100, conditioning_mode=conditioning_mode,
                                     spline_knots=spline_knots)
    best_net._w_mean, best_net._w_std = net._w_mean, net._w_std
    best_net.load_state_dict(checkpoints[best_epoch])
    calibrate_normalization_conditional(best_net, w_grid_calib, n_mc=calib_n_mc_final, seed=calib_seed)
    best_net.eval()

    if verbose:
        log.info(f"[early_stop] best epoch={best_epoch}  val_CVPE={best_val_cvpe:.5f}  "
              f"(stopped at {max(val_cvpe.keys())}/{max_epochs}, "
              f"sub_train n={sub_train_mask.sum()}, val n={val_mask.sum()})")

    return {
        'net': best_net, 'losses': losses, 'checkpoints': checkpoints,
        'val_cvpe': val_cvpe, 'best_epoch': best_epoch, 'best_val_cvpe': best_val_cvpe,
        'sub_train_mask': sub_train_mask, 'val_mask': val_mask,
    }


# ── conditional normalisation Z(w) ───────────────────────────────────────────
def calibration_w_grid(*w_arrays, n_points: int = 25, pad: float = 0.10) -> np.ndarray:
    """Grid of w values on which Z(w) is calibrated.

    The grid spans the range of all training-side w values (the sub-train rows and the early-stopping validation
    rows), widened by pad x range on each side, with n_points equally spaced points. Z(w) is interpolated with
    np.interp, which holds the edge value outside the grid, so a grid that covers only the sub-train range would
    evaluate validation rows beyond it with a clamped Z. Only training-side w values are used (no outcomes, no
    held-out rows).

    Parameters:
    *w_arrays (np.ndarray): Training-side covariate arrays; None entries are skipped.
    n_points (int): Number of grid points. Default 25.
    pad (float): Widening on each side, as a share of the range. Default 0.10.

    Returns:
    np.ndarray: (n_points,) grid.
    """
    ws = np.concatenate([np.asarray(a, dtype=float).ravel() for a in w_arrays if a is not None])
    lo, hi = float(ws.min()), float(ws.max())
    span = hi - lo
    return np.linspace(lo - pad * span, hi + pad * span, n_points)


def calibrate_normalization_conditional(net: ConditionalCopulaNet,
                                         w_grid: np.ndarray,
                                         n_mc: int = 60_000,
                                         seed: int = 0) -> dict:
    """Estimate the normalising function Z(w) on a grid of w values, for interpolation in the density evaluation.

    Z(w) = E_{u~Unif}[D(u, w)], the Monte Carlo mean of the raw network output over uniform points. Dividing by the
    interpolated Z(w) makes the integral of c(u | w) over u close to 1 for any w in range. This replaces the single
    constant of the unconditional network with a function stored as a lookup grid: net._w_grid (sorted ascending) and
    net._norm_grid, which eval_conditional_density uses automatically.

    Parameters:
    net (ConditionalCopulaNet): The network to calibrate (modified in place).
    w_grid (np.ndarray): Covariate values to calibrate at. They should span, and slightly exceed, the range of w
        seen in training (see calibration_w_grid). The order does not matter.
    n_mc (int): Number of uniform Monte Carlo points per grid value. More points reduce the noise of Z(w). A bias
        right at the edges of the w range can remain regardless of n_mc, where the data are sparse. Default 60000.
    seed (int): Simulation seed. Default 0.

    Returns:
    dict: Keys w_grid and norm_grid, the values also stored on net.
    """
    net.eval()
    w_grid = np.sort(np.asarray(w_grid, dtype=np.float32))
    norm_vals = np.empty_like(w_grid)

    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for i, wv in enumerate(w_grid):
            u_mc = rng.uniform(0, 1, size=(n_mc, 2)).astype(np.float32)
            u_t = torch.tensor(u_mc, dtype=torch.float32)
            w_t = torch.full((n_mc,), float(wv), dtype=torch.float32)
            c_hat = net(u_t, w_t).numpy()
            norm_vals[i] = c_hat.mean()

    net._w_grid = w_grid
    net._norm_grid = norm_vals
    return {"w_grid": w_grid, "norm_grid": net._norm_grid}


def _norm_const_at(net: ConditionalCopulaNet, w: np.ndarray) -> np.ndarray:
    """Interpolate the calibrated Z(w) at arbitrary w, holding the edge value outside the calibration range.

    Parameters:
    net (ConditionalCopulaNet): A network calibrated with calibrate_normalization_conditional.
    w (np.ndarray): Covariate value(s) to interpolate at.

    Returns:
    np.ndarray: Interpolated Z(w).
    """
    if net._w_grid is None:
        raise RuntimeError(
            "Call calibrate_normalization_conditional(net, w_grid) before "
            "evaluating densities."
        )
    return np.interp(w, net._w_grid, net._norm_grid)


# ── density evaluation and self-consistency ──────────────────────────────────
def eval_conditional_density(net: ConditionalCopulaNet,
                              u: np.ndarray,
                              w: np.ndarray) -> np.ndarray:
    """Evaluate c(u | w) = D_theta(u, w) / Z(w) at points u, each paired with a covariate value.

    Requires calibrate_normalization_conditional to have been run first.

    Parameters:
    net (ConditionalCopulaNet): A trained and calibrated network.
    u (np.ndarray): (n, 2) points in [0, 1]^2.
    w (np.ndarray or float): Covariate value(s), shape (n,) or one float used for all points.

    Returns:
    np.ndarray: (n,) density values.
    """
    net.eval()
    u = np.asarray(u, dtype=np.float32)
    n = u.shape[0]
    w = np.full(n, w, dtype=np.float32) if np.isscalar(w) else np.asarray(w, dtype=np.float32)

    with torch.no_grad():
        u_t = torch.tensor(u, dtype=torch.float32)
        w_t = torch.tensor(w, dtype=torch.float32)
        c_hat = net(u_t, w_t).numpy()

    norm_const = _norm_const_at(net, w)
    return c_hat / norm_const


def self_consistency_test_conditional(net: ConditionalCopulaNet,
                                       w_values: np.ndarray,
                                       n_mc: int = 10_000,
                                       n_seeds: int = 10,
                                       seed: int = 0,
                                       tol: float = 0.01) -> dict:
    """Check that the normalised conditional density integrates to 1 over u at each w in w_values.

    For each w, the Monte Carlo mean of c(u | w) over uniform points is computed for n_seeds independent samples of
    n_mc points. w_values should include values between the calibration grid points, to test the interpolation
    (the grid points themselves would pass trivially). Pass or fail per w is decided on the effect size
    |E[c | w] - 1| <= tol, as in self_consistency_test; the standard error and z-score are diagnostics only.

    A failure that disappears when calibrate_normalization_conditional is rerun with a larger n_mc is calibration
    noise. A failure that persists reflects an undertrained network at that w, most often at the edges of the training
    range where data are sparse.

    Parameters:
    net (ConditionalCopulaNet): A trained and calibrated network.
    w_values (np.ndarray): Covariate values to test at.
    n_mc (int): Number of uniform points per integral. Default 10000.
    n_seeds (int): Number of independent integrals per w. Default 10.
    seed (int): First seed; the integrals use seed, seed + 1, ..., seed + n_seeds - 1. Default 0.
    tol (float): Tolerance on |E[c | w] - 1|. Default 0.01.

    Returns:
    dict: One entry per w (key float(w)) with keys integral, se, z, deviation, tol and passed, plus the overall flag
        'all_passed'.
    """
    results = {}
    all_passed = True
    for wv in w_values:
        integrals = []
        for s in range(seed, seed + n_seeds):
            rng = np.random.default_rng(s)
            u_mc = rng.uniform(0, 1, size=(n_mc, 2)).astype(np.float32)
            c_vals = eval_conditional_density(net, u_mc, float(wv))
            integrals.append(float(c_vals.mean()))
        integrals = np.array(integrals)
        mean_integral = float(integrals.mean())
        se = float(integrals.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else np.nan
        z = float((mean_integral - 1.0) / se) if se > 0 else np.inf
        deviation = abs(mean_integral - 1.0)
        passed = deviation <= tol
        all_passed = all_passed and passed
        results[float(wv)] = {
            "integral": mean_integral, "se": se, "z": z,
            "deviation": deviation, "tol": tol, "passed": passed,
        }
        log.info(f"  w={wv:+.3f}:  E[ĉ|w] = {mean_integral:.4f}  (SE={se:.4f}, z={z:.2f} "
              f"[diagnostic only])  |deviation|={deviation:.4f} {'≤' if passed else '>'} "
              f"tol={tol}  {'PASS ✓' if passed else 'FAIL ✗'}")

    results["all_passed"] = all_passed
    return results


# ── sampling and statistics at a fixed w ─────────────────────────────────────
def sample_from_conditional_codine(net: ConditionalCopulaNet,
                                    w_value: float,
                                    n_samples: int = 20_000,
                                    grid_size: int = 200,
                                    seed: int = 1) -> np.ndarray:
    """Draw samples from c(. | w_value) with a grid sampler.

    The same strategy as sample_from_codine, with the density evaluated at the fixed w_value: grid cells are
    drawn in proportion to their mass, with uniform jitter within the cell.

    Parameters:
    net (ConditionalCopulaNet): A trained and calibrated network.
    w_value (float): The covariate value to condition on.
    n_samples (int): Number of samples. Default 20000.
    grid_size (int): Number of grid points per dimension. Default 200.
    seed (int): Seed of a local numpy generator; numpy's global generator is not touched. Default 1.

    Returns:
    np.ndarray: (n_samples, 2) array of sampled pairs.
    """
    rng = np.random.default_rng(seed)
    g = np.linspace(0.005, 0.995, grid_size)
    uu, vv = np.meshgrid(g, g)
    pts = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float32)

    density = eval_conditional_density(net, pts, w_value)
    density = np.clip(density, 0, None)
    probs = density / density.sum()

    cell_idx = rng.choice(len(probs), size=n_samples, p=probs)
    cell_u = pts[cell_idx, 0]
    cell_v = pts[cell_idx, 1]

    step = g[1] - g[0]
    jitter_u = rng.uniform(-step / 2, step / 2, n_samples)
    jitter_v = rng.uniform(-step / 2, step / 2, n_samples)

    samples = np.column_stack([
        np.clip(cell_u + jitter_u, 0.001, 0.999),
        np.clip(cell_v + jitter_v, 0.001, 0.999),
    ])
    return samples


def conditional_kendall_tau(net: ConditionalCopulaNet,
                             w_value: float,
                             n_samples: int = 20_000,
                             seed: int = 1) -> float:
    """Kendall's tau at a fixed w, estimated from samples drawn from c(. | w).

    Parameters:
    net (ConditionalCopulaNet): A trained and calibrated network.
    w_value (float): The covariate value to condition on.
    n_samples (int): Number of samples. Default 20000.
    seed (int): Sampling seed. Default 1.

    Returns:
    float: Estimated Kendall's tau at w_value.
    """
    samples = sample_from_conditional_codine(net, w_value, n_samples=n_samples, seed=seed)
    tau, _ = kendalltau(samples[:, 0], samples[:, 1])
    return float(tau)


def conditional_calibration_curve(net: ConditionalCopulaNet,
                                   w_grid: np.ndarray,
                                   n_samples: int = 20_000,
                                   seed: int = 1) -> np.ndarray:
    """Neural analogue of the Acar et al. calibration curve: Kendall's tau-hat(w) on a grid of w values.

    Parameters:
    net (ConditionalCopulaNet): A trained and calibrated network.
    w_grid (np.ndarray): Covariate values to evaluate tau-hat at.
    n_samples (int): Samples per w value, passed to conditional_kendall_tau. Default 20000.
    seed (int): Sampling seed. Default 1.

    Returns:
    np.ndarray: tau-hat at each point of w_grid.
    """
    taus = np.array([
        conditional_kendall_tau(net, float(wv), n_samples=n_samples, seed=seed)
        for wv in w_grid
    ])
    return taus


# ── synthetic data, plotting, sampler factory ────────────────────────────────
def simulate_conditional_student_t_copula(n: int,
                                           w: np.ndarray,
                                           rho_fn,
                                           df: float = 4.0,
                                           seed: int = 0):
    """Simulate pseudo-observations from a bivariate Student-t copula whose correlation varies with w.

    The correlation is rho(w) = rho_fn(w) in (-1, 1). Kendall's tau of an elliptical copula is (2 / pi) arcsin(rho),
    independent of the degrees of freedom (Demarta & McNeil, 2005, eq. 9), which gives the true curve to compare a
    fit against.

    Parameters:
    n (int): Number of samples.
    w (np.ndarray): Covariate values, length n.
    rho_fn (callable): Function mapping w to the correlation rho(w).
    df (float): Degrees of freedom. Default 4.0.
    seed (int): Simulation seed. Default 0.

    Returns:
    tuple: (u1, u2, rho, tau_true), where u1 and u2 are float32 arrays of length n and rho and tau_true are the
        true correlation and Kendall's tau at each point.
    """
    from scipy.stats import t as student_t
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=np.float64)
    rho = rho_fn(w)
    rho = np.clip(rho, -0.999, 0.999)   # numerical guard

    z = rng.standard_normal((n, 2))
    z1 = z[:, 0]
    z2 = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
    chi2 = rng.chisquare(df, size=n)
    scale = np.sqrt(df / chi2)
    x1 = z1 * scale
    x2 = z2 * scale

    u1 = student_t.cdf(x1, df)
    u2 = student_t.cdf(x2, df)
    tau_true = (2.0 / np.pi) * np.arcsin(rho)

    return u1.astype(np.float32), u2.astype(np.float32), rho, tau_true


def make_codine_sample_fn(net_obj):
    """Return a sample_fn(wv, n, s) bound to one fitted network, for marginal_uniformity_check.

    A factory instead of a closure built in a loop over several networks, which would use the last network assigned
    (see conditional_copula_flow.make_flow_sample_fn).

    Parameters:
    net_obj (ConditionalCopulaNet): A trained and calibrated network.

    Returns:
    callable: sample_fn(wv, n, s) -> np.ndarray, drawing n samples at covariate value wv with seed s.
    """
    return lambda wv, n, s: sample_from_conditional_codine(net_obj, wv, n_samples=n, seed=s)
