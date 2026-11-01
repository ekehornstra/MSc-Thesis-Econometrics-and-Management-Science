"""conditional_copula_flow.py

Conditional copula flow (C-CopFlow): a normalising flow for the conditional copula density c(u1, u2 | w).

A base sample z ~ N(0, I) is mapped by K conditional coupling layers to x, and u = Phi(x), the elementwise
standard normal CDF, puts the result on the unit square. The layers alternate which coordinate is transformed,
and the conditioner network of a layer sees the other coordinate and the covariate w. The density follows from
the change-of-variables formula, log c(u | w) = log p_Z(z) + log|dz/dx| + log|dx/du|, so it is normalised by
construction for every w. Training is maximum likelihood on the observed pseudo-observations alone. No
normalising constant Z(w) is estimated and no uniform negatives are drawn, unlike in C-CODINE
(codine.py), whose uniform draws rarely reach the corner regions that matter for tail dependence.

The construction is that of Wen and Torkkola (2019) for a Gaussian copula (standard normal base, Cholesky
transform, normal CDF), with the linear transform replaced by a nonlinear flow conditioned on w. They name the
conditioning of flows on covariates as an open point and leave it as future work (their Section 5). Two coupling
transforms are available: an affine map (Dinh et al., 2016; used for copula flows by Bossemeyer, 2020, and Wiese
et al., 2019) and a monotonic rational-quadratic spline (Durkan et al., 2019; used for copula flows by Kamthe
et al., 2021). w reaches the conditioner in one of three ways, 'w_only', 'spline_basis' or 'film', as in
C-CODINE.

The module also holds the trainers (fixed number of epochs, and early stopping on the validation CVPE), the
density evaluator, the sampler, the Kendall's tau curve, a self-consistency check, and a diagnostic for spline
flows.

References:
Bossemeyer, L. (2020). CM flows: Copula density estimation with normalizing flows [Master's thesis]. University
    of Edinburgh.
Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. arXiv:1605.08803.
Dumoulin, V., Perez, E., Schucher, N., Strub, F., de Vries, H., Courville, A., & Bengio, Y. (2018).
    Feature-wise transformations. Distill, 3(7), e11.
Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. Advances in Neural
    Information Processing Systems, 32.
Kamthe, S., Assefa, S., & Deisenroth, M. (2021). Copula flows for synthetic data generation. arXiv:2101.00598.
Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1 convolutions. Advances in Neural
    Information Processing Systems, 31.
Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked autoregressive flow for density estimation. Advances
    in Neural Information Processing Systems, 30.
Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general
    conditioning layer. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).
Wen, R., & Torkkola, K. (2019). Deep generative quantile-copula models for probabilistic forecasting.
    arXiv:1907.10697.
Wiese, M., Knobloch, R., & Korn, R. (2019). Copula & marginal flows: Disentangling the marginal from its joint.
    arXiv:1907.03361.
"""

import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau
from tqdm.auto import tqdm

from codine import margin_uniformity_penalty
from copula_families import QUAD_N, _spline_basis_w
from data_split import train_test_split_chrono
from run_support import get_logger

log = get_logger()

LOG_2PI = float(np.log(2 * np.pi))


# ── standard normal helpers ─────────────────────────────────────────────────

def standard_normal_logpdf(z: torch.Tensor) -> torch.Tensor:
    """log N(z; 0, 1), elementwise.

    Parameters:
    z (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Log-density, same shape as z. Caller sums over dims as needed.
    """
    return -0.5 * (z ** 2 + LOG_2PI)


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, elementwise: u = Phi(x).

    Phi (and not a sigmoid) maps the flow output to the unit interval, following Bossemeyer (2020). Wiese et al.
    (2019) use the sigmoid.

    Parameters:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Phi(x), same shape as x.
    """
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def normal_icdf(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard normal inverse CDF, elementwise: x = Phi^-1(u).

    u is clamped away from {0,1} for numerical stability (erfinv blows up
    at the boundary).

    Parameters:
    u (torch.Tensor): Input tensor, values in [0, 1].
    eps (float): Clamp margin away from 0 and 1. Default 1e-6.

    Returns:
    torch.Tensor: Phi^-1(u), same shape as u.
    """
    u = torch.clamp(u, eps, 1.0 - eps)
    return np.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


# ── rational-quadratic spline ───────────────────────────────────────────────

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
MAX_PENALTY_DENSITY = 100.0         # cap on the density inside the margin penalty, so a corner spike cannot dominate the gradient
DEFAULT_MIN_DERIVATIVE = 1e-3
# Ceiling on the spline's local derivative, the analogue of scale_bound in the affine coupling. The density
# scales with the local derivative, so without a ceiling the spline could represent arbitrarily tall density
# peaks. A tuning constant, not derived from theory.
DEFAULT_MAX_DERIVATIVE = 50.0


def _derivative_raw_at_target(target_derivative, min_derivative, max_derivative):
    """Raw (pre-sigmoid) parameter whose bounded derivative equals a target value.

    Inverts the log-space parameterisation of rational_quadratic_spline,
    derivative = exp(log_min + (log_max - log_min) * sigmoid(raw)). It is used in two places:
    1. unconstrained_rqs pads the derivative array with this value at both ends, so that the spline's derivative
       matches the slope 1 of the identity tails at +-tail_bound.
    2. _init_near_identity_spline sets the bias of the derivative outputs to this value, so that the interior
       derivatives also start at 1.

    Parameters:
    target_derivative (float): Derivative to reproduce (1.0 in both uses).
    min_derivative (float): Floor of the derivative range.
    max_derivative (float): Ceiling of the derivative range.

    Returns:
    float: The raw value, a plain Python float used to fill a constant tensor slice (not differentiated).
    """
    log_min = math.log(min_derivative)
    log_max = math.log(max_derivative)
    log_target = math.log(target_derivative)
    p = (log_target - log_min) / (log_max - log_min)
    p = min(max(p, 1e-6), 1 - 1e-6)  # guard against exact 0/1 -> logit(+-inf)
    return math.log(p / (1 - p))


def unconstrained_rqs(inputs, unnormalized_widths, unnormalized_heights,
                       unnormalized_derivatives, inverse=False, tail_bound=4.0,
                       min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                       min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                       min_derivative=DEFAULT_MIN_DERIVATIVE,
                       max_derivative=DEFAULT_MAX_DERIVATIVE):
    """Rational-quadratic spline on the whole real line (identity outside [-tail_bound, tail_bound]).

    Inside [-tail_bound, tail_bound] the transform is the spline. Its derivative at both ends is fixed to 1, so that
    it meets the identity tails with a continuous derivative.

    Parameters:
    inputs (torch.Tensor): Values to transform, any shape.
    unnormalized_widths (torch.Tensor): Raw per-bin width logits, shape (..., num_bins).
    unnormalized_heights (torch.Tensor): Raw per-bin height logits, same shape as unnormalized_widths.
    unnormalized_derivatives (torch.Tensor): Raw interior-knot derivative logits, shape (..., num_bins - 1) before
        the boundary padding applied here.
    inverse (bool): If True, apply the inverse (x -> z) transform instead of the forward (z -> x) one. Default False.
    tail_bound (float): The spline domain is [-tail_bound, tail_bound]. Default 4.0.
    min_bin_width (float): Floor on each bin's width, as a fraction of the domain. Default DEFAULT_MIN_BIN_WIDTH.
    min_bin_height (float): Floor on each bin's height, as a fraction of the domain. Default DEFAULT_MIN_BIN_HEIGHT.
    min_derivative (float): Floor on the spline's local derivative. Default DEFAULT_MIN_DERIVATIVE.
    max_derivative (float): Ceiling on the spline's local derivative. Default DEFAULT_MAX_DERIVATIVE.

    Returns:
    tuple: (outputs, logabsdet), both the same shape as inputs.
    """
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside = ~inside

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    outputs[outside] = inputs[outside]
    logabsdet[outside] = 0.0

    if inside.sum() == 0:
        return outputs, logabsdet

    unnormalized_derivatives = torch.nn.functional.pad(unnormalized_derivatives, (1, 1))
    # Boundary derivative = 1 (the slope of the identity tails), for continuity.
    constant = _derivative_raw_at_target(1.0, min_derivative, max_derivative)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    out_in, ld_in = rational_quadratic_spline(
        inputs[inside],
        unnormalized_widths[inside, :],
        unnormalized_heights[inside, :],
        unnormalized_derivatives[inside, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width, min_bin_height=min_bin_height,
        min_derivative=min_derivative, max_derivative=max_derivative,
    )
    outputs[inside] = out_in
    logabsdet[inside] = ld_in
    return outputs, logabsdet


def rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights,
                               unnormalized_derivatives, inverse=False,
                               left=0., right=1., bottom=0., top=1.,
                               min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                               min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                               min_derivative=DEFAULT_MIN_DERIVATIVE,
                               max_derivative=DEFAULT_MAX_DERIVATIVE):
    """Monotonic rational-quadratic spline transform on a bounded domain (Durkan et al., 2019).

    Parameters:
    inputs (torch.Tensor): Values to transform, already restricted to [left, right] (forward) or [bottom, top]
        (inverse) by the caller.
    unnormalized_widths (torch.Tensor): Raw per-bin width logits, shape (..., num_bins).
    unnormalized_heights (torch.Tensor): Raw per-bin height logits, same shape as unnormalized_widths.
    unnormalized_derivatives (torch.Tensor): Raw knot derivative logits, shape (..., num_bins + 1) (already
        boundary-padded by the caller).
    inverse (bool): If True, apply the inverse (x -> z) transform instead of the forward (z -> x) one. Default False.
    left (float): Lower bound of the input axis. Default 0.
    right (float): Upper bound of the input axis. Default 1.
    bottom (float): Lower bound of the output axis. Default 0.
    top (float): Upper bound of the output axis. Default 1.
    min_bin_width (float): Floor on each bin's width, as a fraction of the domain. Default DEFAULT_MIN_BIN_WIDTH.
    min_bin_height (float): Floor on each bin's height, as a fraction of the domain. Default DEFAULT_MIN_BIN_HEIGHT.
    min_derivative (float): Floor on the spline's local derivative. Default DEFAULT_MIN_DERIVATIVE.
    max_derivative (float): Ceiling on the spline's local derivative. Default DEFAULT_MAX_DERIVATIVE.

    Returns:
    tuple: (outputs, logabsdet), both the same shape as inputs.
    """
    import torch.nn.functional as F
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # The derivative is bounded in log space, as the affine coupling bounds log(scale) with scale_bound * tanh.
    # A linear-space sigmoid cap would already sit near max_derivative at raw = 0 (about 25 for min 1e-3,
    # max 50), where an untrained network's outputs lie. The log-space interpolation spreads the transition
    # over the whole range [min_derivative, max_derivative].
    log_min_d = math.log(min_derivative)
    log_max_d = math.log(max_derivative)
    derivatives = torch.exp(log_min_d + (log_max_d - log_min_d) * torch.sigmoid(unnormalized_derivatives))

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = _searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_p1 = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives + input_derivatives_p1 - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives + input_derivatives_p1 - 2 * input_delta))
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all(), "RQS inverse: negative discriminant -- monotonicity violated"

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_p1 - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_p1 * root.pow(2)
                                                       + 2 * input_delta * theta_one_minus_theta
                                                       + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_p1 - 2 * input_delta) * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_p1 * theta.pow(2)
                                                       + 2 * input_delta * theta_one_minus_theta
                                                       + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


def _searchsorted(bin_locations, inputs, eps=1e-6):
    """Find which spline bin each input falls into.

    Parameters:
    bin_locations (torch.Tensor): Cumulative bin edges, shape (..., num_bins + 1).
    inputs (torch.Tensor): Values to locate, shape (...,).
    eps (float): Nudges the last edge outward so a value exactly at the
        upper boundary still lands in the last bin. Default 1e-6.

    Returns:
    torch.Tensor: Bin index per input, same shape as inputs.
    """
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


class ConditionalAffineCoupling(nn.Module):
    """One affine coupling layer of C-CopFlow (bivariate, d = 2).

    One coordinate (the target) is transformed as y = target * exp(log_s) + t (the affine coupling layer of
    Dinh et al., 2016), where log_s and t come from a small MLP that sees the other coordinate (cond) and the covariate w. Which coordinate is transformed alternates across
    stacked layers (see ConditionalCopulaFlow), because a single layer would let x2 depend on x1 but not the other
    way round. log_s is bounded as scale_bound * tanh(...), which keeps exp(log_s) from blowing up early in training.

    conditioning_mode sets how w reaches the conditioner:
    - 'w_only': w is concatenated with cond (2 inputs).
    - 'spline_basis': w is replaced by the fixed cubic truncated-power spline basis (copula_families._spline_basis_w)
      before concatenation with cond.
    - 'film': cond alone feeds the main network. A linear map from w gives a gain and a shift that rescale and shift
      the first hidden layer, h <- gamma(w) * h + beta(w) (FiLM; Perez et al., 2018; Dumoulin et al., 2018), with
      gamma = 1 + gamma_raw. gamma is unbounded, so a unit can be damped, amplified or negated. The map is
      initialised at zero, so the modulation starts as the identity.
    """

    def __init__(self, hidden: int = 64, scale_bound: float = 3.5,
                 conditioning_mode: str = 'w_only', spline_knots: torch.Tensor = None):
        """Build the conditioner network of the layer.

        Parameters:
        hidden (int): Hidden layer width. Default 64.
        scale_bound (float): Bound on the log-scale output, via scale_bound * tanh(...). Default 3.5.
        conditioning_mode (str): 'w_only', 'spline_basis' or 'film', see the class docstring. Default 'w_only'.
        spline_knots (torch.Tensor): Fixed knot positions for _spline_basis_w, registered as a buffer (part of
            state_dict, moves with .to(device)). Required if conditioning_mode is 'spline_basis'. Optional otherwise.
        """
        super().__init__()
        if conditioning_mode not in ('w_only', 'spline_basis', 'film'):
            raise ValueError(f"conditioning_mode must be 'w_only', 'spline_basis', "
                              f"or 'film', got {conditioning_mode!r}")
        self.scale_bound = scale_bound
        self.conditioning_mode = conditioning_mode

        if conditioning_mode == 'spline_basis':
            if spline_knots is None:
                raise ValueError("conditioning_mode='spline_basis' requires spline_knots")
            self.register_buffer('spline_knots', spline_knots)

        if conditioning_mode == 'film':
            # The main net sees cond only. w enters through gamma and beta on the first hidden layer. film_net is
            # zero-initialised, so the modulation is a no-op at the start, and linear3 is zeroed too
            # (_init_near_identity_affine), so that the whole layer starts as the identity.
            self.linear1 = nn.Linear(1, hidden)
            self.linear2 = nn.Linear(hidden, hidden)
            self.linear3 = nn.Linear(hidden, 2)
            self.film_net = nn.Linear(1, 2 * hidden)
            nn.init.zeros_(self.film_net.weight)
            nn.init.zeros_(self.film_net.bias)
            _init_near_identity_affine(self.linear3)
        else:
            if conditioning_mode == 'spline_basis':
                in_dim = 1 + 4 + len(spline_knots)
            else:
                in_dim = 2
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 2),
            )
            _init_near_identity_affine(self.net[-1])

    def _params(self, cond: torch.Tensor, w: torch.Tensor):
        """Compute this layer's bounded log-scale and shift from the conditioning input.

        Parameters:
        cond (torch.Tensor): The conditioning coordinate.
        w (torch.Tensor): Standardised covariate value.

        Returns:
        tuple: (log_s, t), the bounded log-scale and shift.
        """
        if self.conditioning_mode == 'film':
            h = torch.tanh(self.linear1(cond.unsqueeze(-1)))
            gamma_raw, beta = self.film_net(w.unsqueeze(-1)).chunk(2, dim=-1)
            gamma = 1.0 + gamma_raw   # unbounded: can damp, amplify or negate a unit
            h = gamma * h + beta
            h = torch.tanh(self.linear2(h))
            out = self.linear3(h)
        else:
            if self.conditioning_mode == 'spline_basis':
                inp = torch.cat([cond.unsqueeze(-1), _spline_basis_w(w, self.spline_knots)], dim=-1)
            else:
                inp = torch.stack([cond, w], dim=-1)
            out = self.net(inp)
        log_s_raw, t = out[..., 0], out[..., 1]
        log_s = self.scale_bound * torch.tanh(log_s_raw)
        return log_s, t

    def forward(self, cond: torch.Tensor, target: torch.Tensor, w: torch.Tensor):
        """Map the z-coordinate to the x-coordinate (used when sampling).

        Parameters:
        cond (torch.Tensor): The conditioning coordinate (unchanged).
        target (torch.Tensor): The coordinate being transformed.
        w (torch.Tensor): Standardised covariate value.

        Returns:
        tuple: (y, log_s), the transformed coordinate and its log-det
            contribution.
        """
        log_s, t = self._params(cond, w)
        y = target * torch.exp(log_s) + t
        return y, log_s  # log_s IS the log-det contribution for this coordinate

    def inverse(self, cond: torch.Tensor, target: torch.Tensor, w: torch.Tensor):
        """Map the x-coordinate to the z-coordinate (used when evaluating density).

        Parameters:
        cond (torch.Tensor): The conditioning coordinate (unchanged).
        target (torch.Tensor): The coordinate being inverted.
        w (torch.Tensor): Standardised covariate value.

        Returns:
        tuple: (x, -log_s), the inverted coordinate and its log-det
            contribution.
        """
        log_s, t = self._params(cond, w)
        x = (target - t) * torch.exp(-log_s)
        return x, -log_s


def _init_near_identity_affine(final_linear: nn.Linear):
    """Zero-initialise the final linear layer of an affine conditioner, so the coupling layer starts as the identity.

    With zero weight and zero bias, log_s_raw = 0 and t = 0 for every input, so log_s = scale_bound * tanh(0) = 0 and
    y = target * exp(0) + 0 = target. Kingma and Dhariwal (2018) use the same initialisation (the last layer of each
    coupling network at zero, so that each affine coupling layer starts as the identity). Unlike the spline case (_init_near_identity_spline), no bias offset is needed, because
    zero is already the identity point of an affine transform.

    Parameters:
    final_linear (nn.Linear): The conditioner's last layer, modified in place.
    """
    nn.init.zeros_(final_linear.weight)
    nn.init.zeros_(final_linear.bias)


def _init_near_identity_spline(final_linear: nn.Linear, num_bins: int,
                                min_derivative: float, max_derivative: float):
    """Initialise the final linear layer of a spline conditioner, so the coupling layer starts near the identity.

    The weight is set to zero, so the raw output at initialisation is the bias alone. The bias is 0 for the width and
    height outputs (softmax(0) gives uniform bins) and the constant from _derivative_raw_at_target for the derivative
    outputs, so that every interior derivative starts at 1, matching the fixed boundary derivative.

    Parameters:
    final_linear (nn.Linear): The conditioner's last layer, modified in place.
    num_bins (int): Number of spline bins. The first 2 * num_bins outputs are widths and heights.
    min_derivative (float): Floor of the derivative range.
    max_derivative (float): Ceiling of the derivative range.
    """
    nn.init.zeros_(final_linear.weight)
    bias = torch.zeros_like(final_linear.bias)
    deriv_const = _derivative_raw_at_target(1.0, min_derivative, max_derivative)
    bias[2 * num_bins:] = deriv_const  # widths/heights slices (indices 0:2*num_bins) stay 0
    with torch.no_grad():
        final_linear.bias.copy_(bias)


class ConditionalSplineCoupling(nn.Module):
    """One rational-quadratic spline coupling layer of C-CopFlow (bivariate, d = 2; Durkan et al., 2019).

    Same role as ConditionalAffineCoupling (one coordinate is transformed, conditioned on the other and w), but the
    transform is a monotonic spline that can bend within a coupling step, and not only scale and shift. Kamthe et al.
    (2021) use rational-quadratic spline flows (autoregressive, not coupling) for copula flows. conditioning_mode has the same options and meaning as in
    ConditionalAffineCoupling ('w_only', 'spline_basis', 'film'). Only the output of the conditioner differs: it
    produces 3 * num_bins - 1 numbers (unnormalised bin widths, heights and interior derivatives).

    Outside [-tail_bound, tail_bound] the transform is the identity, which keeps it defined on the whole real line.
    The boundary derivatives are fixed at 1, so the spline meets the identity tails with a continuous derivative.
    max_derivative caps the local derivative (see DEFAULT_MAX_DERIVATIVE), the analogue of the bound scale_bound
    puts on the affine layer. The final linear layer is zero-weight-initialised with a bias that makes the layer start
    near the identity (_init_near_identity_spline).
    """

    def __init__(self, hidden: int = 64, num_bins: int = 8, tail_bound: float = 4.0,
                 conditioning_mode: str = 'w_only', spline_knots: torch.Tensor = None,
                 min_derivative: float = DEFAULT_MIN_DERIVATIVE,
                 max_derivative: float = DEFAULT_MAX_DERIVATIVE):
        """Build the conditioner network of the layer.

        Parameters:
        hidden (int): Hidden layer width. Default 64.
        num_bins (int): Number of spline bins. Default 8.
        tail_bound (float): The spline domain is [-tail_bound, tail_bound], the identity outside. Default 4.0, which
            covers more than 99.99% of the mass of a standard normal.
        conditioning_mode (str): 'w_only', 'spline_basis' or 'film'. Default 'w_only'.
        spline_knots (torch.Tensor): Fixed knot positions for _spline_basis_w, registered as a buffer (part of
            state_dict, moves with .to(device)). Required if conditioning_mode is 'spline_basis'. Optional otherwise.
        min_derivative (float): Floor on the spline's local derivative. Default DEFAULT_MIN_DERIVATIVE.
        max_derivative (float): Ceiling on the spline's local derivative. Default DEFAULT_MAX_DERIVATIVE.
        """
        super().__init__()
        if conditioning_mode not in ('w_only', 'spline_basis', 'film'):
            raise ValueError(f"conditioning_mode must be 'w_only', 'spline_basis', "
                              f"or 'film', got {conditioning_mode!r}")
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.conditioning_mode = conditioning_mode
        self.min_derivative = min_derivative
        self.max_derivative = max_derivative
        out_dim = 3 * num_bins - 1

        if conditioning_mode == 'spline_basis':
            if spline_knots is None:
                raise ValueError("conditioning_mode='spline_basis' requires spline_knots")
            self.register_buffer('spline_knots', spline_knots)

        if conditioning_mode == 'film':
            self.linear1 = nn.Linear(1, hidden)
            self.linear2 = nn.Linear(hidden, hidden)
            self.linear3 = nn.Linear(hidden, out_dim)
            self.film_net = nn.Linear(1, 2 * hidden)
            nn.init.zeros_(self.film_net.weight)
            nn.init.zeros_(self.film_net.bias)
            _init_near_identity_spline(self.linear3, num_bins, min_derivative, max_derivative)
        else:
            in_dim = (4 + len(spline_knots) + 1) if conditioning_mode == 'spline_basis' else 2
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, out_dim),
            )
            _init_near_identity_spline(self.net[-1], num_bins, min_derivative, max_derivative)

    def _spline_params(self, cond: torch.Tensor, w: torch.Tensor):
        """Compute this layer's unnormalised spline parameters from the conditioning input.

        Parameters:
        cond (torch.Tensor): The conditioning coordinate.
        w (torch.Tensor): Standardised covariate value.

        Returns:
        tuple: (widths, heights, derivatives), the raw (pre-normalisation)
            spline parameters, split from the conditioner network's
            combined output.
        """
        if self.conditioning_mode == 'film':
            h = torch.tanh(self.linear1(cond.unsqueeze(-1)))
            gamma_raw, beta = self.film_net(w.unsqueeze(-1)).chunk(2, dim=-1)
            gamma = 1.0 + gamma_raw   # unbounded: can damp, amplify or negate a unit
            h = gamma * h + beta
            h = torch.tanh(self.linear2(h))
            out = self.linear3(h)
        else:
            if self.conditioning_mode == 'spline_basis':
                inp = torch.cat([cond.unsqueeze(-1), _spline_basis_w(w, self.spline_knots)], dim=-1)
            else:
                inp = torch.stack([cond, w], dim=-1)
            out = self.net(inp)
        widths = out[..., :self.num_bins]
        heights = out[..., self.num_bins:2 * self.num_bins]
        derivatives = out[..., 2 * self.num_bins:]
        return widths, heights, derivatives

    def forward(self, cond: torch.Tensor, target: torch.Tensor, w: torch.Tensor):
        """Map the z-coordinate to the x-coordinate (used when sampling).

        Parameters:
        cond (torch.Tensor): The conditioning coordinate (unchanged).
        target (torch.Tensor): The coordinate being transformed.
        w (torch.Tensor): Standardised covariate value.

        Returns:
        tuple: (y, logabsdet), the transformed coordinate and its log-det
            contribution.
        """
        widths, heights, derivatives = self._spline_params(cond, w)
        y, logabsdet = unconstrained_rqs(target, widths, heights, derivatives,
                                          inverse=False, tail_bound=self.tail_bound,
                                          min_derivative=self.min_derivative,
                                          max_derivative=self.max_derivative)
        return y, logabsdet

    def inverse(self, cond: torch.Tensor, target: torch.Tensor, w: torch.Tensor):
        """Map the x-coordinate to the z-coordinate (used when evaluating density).

        Parameters:
        cond (torch.Tensor): The conditioning coordinate (unchanged).
        target (torch.Tensor): The coordinate being inverted.
        w (torch.Tensor): Standardised covariate value.

        Returns:
        tuple: (x, logabsdet), the inverted coordinate and its log-det
            contribution.
        """
        widths, heights, derivatives = self._spline_params(cond, w)
        x, logabsdet = unconstrained_rqs(target, widths, heights, derivatives,
                                          inverse=True, tail_bound=self.tail_bound,
                                          min_derivative=self.min_derivative,
                                          max_derivative=self.max_derivative)
        return x, logabsdet


# ── flow ────────────────────────────────────────────────────────────────────

class ConditionalCopulaFlow(nn.Module):
    """Conditional copula flow (bivariate, d = 2): a stack of coupling layers followed by the normal CDF.

    Stacks n_layers coupling layers (affine or spline), alternating which coordinate is transformed.
    conditioning_mode is forwarded unchanged to every layer, and all layers of one flow share the same mode. The
    default is n_layers=8, the value Bossemeyer (2020) found best in a grid search over {4, 8, 16} coupling layers.
    """

    def __init__(self, n_layers: int = 8, hidden: int = 64, scale_bound: float = 3.5,
                 conditioning_mode: str = 'w_only', coupling_type: str = 'affine',
                 num_bins: int = 8, tail_bound: float = 4.0, spline_knots=None,
                 min_derivative: float = DEFAULT_MIN_DERIVATIVE,
                 max_derivative: float = DEFAULT_MAX_DERIVATIVE):
        """Build the stack of coupling layers and initialise the standardisation state.

        Parameters:
        n_layers (int): Number of coupling layers. Default 8.
        hidden (int): Hidden layer width per coupling layer. Default 64.
        scale_bound (float): Log-scale bound, used only if coupling_type is 'affine'. Default 3.5.
        conditioning_mode (str): 'w_only', 'spline_basis' or 'film', passed to every coupling layer. Default 'w_only'.
        coupling_type (str): 'affine' or 'spline', the transform of every layer. Default 'affine'.
        num_bins (int): Spline bins per layer, used only if coupling_type is 'spline'. Default 8.
        tail_bound (float): Spline domain [-tail_bound, tail_bound], used only if coupling_type is 'spline'. Default 4.0.
        spline_knots (torch.Tensor): Knot positions for conditioning_mode 'spline_basis', whatever the coupling type.
            Optional otherwise.
        min_derivative (float): Floor on the spline's local derivative, used only if coupling_type is 'spline'.
            Default DEFAULT_MIN_DERIVATIVE.
        max_derivative (float): Ceiling on the spline's local derivative, used only if coupling_type is 'spline'.
            Default DEFAULT_MAX_DERIVATIVE.
        """
        super().__init__()
        if coupling_type not in ('affine', 'spline'):
            raise ValueError(f"coupling_type must be 'affine' or 'spline', got {coupling_type!r}")
        self.n_layers = n_layers
        self.conditioning_mode = conditioning_mode
        self.coupling_type = coupling_type

        if coupling_type == 'affine':
            self.layers = nn.ModuleList([
                ConditionalAffineCoupling(hidden=hidden, scale_bound=scale_bound,
                                           conditioning_mode=conditioning_mode,
                                           spline_knots=spline_knots)
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                ConditionalSplineCoupling(hidden=hidden, num_bins=num_bins, tail_bound=tail_bound,
                                           conditioning_mode=conditioning_mode,
                                           spline_knots=spline_knots,
                                           min_derivative=min_derivative,
                                           max_derivative=max_derivative)
                for _ in range(n_layers)
            ])
        self._transform_idx = [i % 2 for i in range(n_layers)]
        self._w_mean = 0.0
        self._w_std = 1.0

    def _standardize_w(self, w: torch.Tensor) -> torch.Tensor:
        """Standardise the covariate using the stored training mean/std.

        Parameters:
        w (torch.Tensor): Raw covariate values.

        Returns:
        torch.Tensor: Standardised covariate values.
        """
        return (w - self._w_mean) / self._w_std

    def sample_x(self, z: torch.Tensor, w: torch.Tensor):
        """Map z to x, forward direction, used for sampling.

        Parameters:
        z (torch.Tensor): (n, 2) base-distribution samples.
        w (torch.Tensor): (n,) raw covariate values.

        Returns:
        tuple: (x, logdet_total), the transformed points and the total
            log-det of dz->dx (unused for sampling but returned for
            symmetry/debugging).
        """
        w_std = self._standardize_w(w)
        x0, x1 = z[:, 0].clone(), z[:, 1].clone()
        logdet_total = torch.zeros(z.shape[0], device=z.device)
        for layer, t_idx in zip(self.layers, self._transform_idx):
            if t_idx == 1:
                y1, log_s = layer.forward(x0, x1, w_std)
                x1 = y1
            else:
                y0, log_s = layer.forward(x1, x0, w_std)
                x0 = y0
            logdet_total = logdet_total + log_s
        x = torch.stack([x0, x1], dim=-1)
        return x, logdet_total

    def invert_to_z(self, x: torch.Tensor, w: torch.Tensor):
        """Map x to z, inverse direction, used for density evaluation.

        Parameters:
        x (torch.Tensor): (n, 2) points in x-space.
        w (torch.Tensor): (n,) raw covariate values.

        Returns:
        tuple: (z, logdet_total), the base-distribution points and the
            total log-det of dx->dz, as needed for the
            change-of-variables density formula.
        """
        w_std = self._standardize_w(w)
        x0, x1 = x[:, 0].clone(), x[:, 1].clone()
        logdet_total = torch.zeros(x.shape[0], device=x.device)
        for layer, t_idx in zip(reversed(list(self.layers)), reversed(self._transform_idx)):
            if t_idx == 1:
                z1, log_s = layer.inverse(x0, x1, w_std)
                x1 = z1
            else:
                z0, log_s = layer.inverse(x1, x0, w_std)
                x0 = z0
            logdet_total = logdet_total + log_s
        z = torch.stack([x0, x1], dim=-1)
        return z, logdet_total

    def log_density(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Exact log copula density log c(u|w), via change of variables.

        Chains both transformations: u -> x (elementwise Gaussian ICDF)
        -> z (flow inverse).

            p_U(u) = p_Z(z) * |dz/dx| * |dx/du|

        No Monte Carlo and no normaliser calibration: exact by construction.

        Parameters:
        u (torch.Tensor): (n, 2) points in [0, 1]^2.
        w (torch.Tensor): (n,) raw covariate values.

        Returns:
        torch.Tensor: (n,) log-density values.
        """
        x = normal_icdf(u)                                    # (n,2)
        log_jac_u_to_x = -standard_normal_logpdf(x).sum(dim=-1)      # log|dx/du|, summed over dims
        z, logdet_x_to_z = self.invert_to_z(x, w)              # log|dz/dx|
        log_pz = standard_normal_logpdf(z).sum(dim=-1)
        return log_pz + logdet_x_to_z + log_jac_u_to_x

    def sample(self, w: torch.Tensor, seed: int = None) -> torch.Tensor:
        """Draw one copula sample per requested w value.

        Parameters:
        w (torch.Tensor): (n,) covariate values, one per sample to draw.
        seed (int): Optional seed for reproducible sampling. Optional.

        Returns:
        torch.Tensor: (n, 2) sampled pairs.
        """
        n = w.shape[0]
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            z = torch.randn(n, 2, generator=g)
        else:
            z = torch.randn(n, 2)
        x, _ = self.sample_x(z, w)
        u = normal_cdf(x)
        return u


# ── training ────────────────────────────────────────────────────────────────

def train_conditional_copula_flow(u1, u2, w, n_epochs=1500, batch_size=32, lr=1e-3,
                                   seed=42, n_layers=8, hidden=64, scale_bound=3.5,
                                   conditioning_mode='w_only', coupling_type='affine',
                                   num_bins=8, tail_bound=4.0,
                                   min_derivative=DEFAULT_MIN_DERIVATIVE,
                                   max_derivative=DEFAULT_MAX_DERIVATIVE,
                                   grad_clip_norm=5.0, weight_decay=0.0,
                                   checkpoint_every=100, show_progress=True,
                                   val_u1=None, val_u2=None, val_w=None,
                                   patience=None, grid_n=QUAD_N, position=None,
                                   margin_penalty=0.0, margin_n_w=8, margin_grid=20, margin_every=1):
    """Train a ConditionalCopulaFlow by maximum likelihood, optionally with early stopping.

    The loss is -mean(log_density(u, w)) over shuffled minibatches. Without validation data, a copy of the weights is
    stored every checkpoint_every epochs. With val_u1, val_u2, val_w and patience, the validation CVPE
    (model_evaluation.cvpe_only) is computed at every checkpoint, only the best weights are kept, and training stops
    after patience checkpoints without an improvement of more than 1e-6.

    state_dict() does not contain _w_mean and _w_std (plain floats, not parameters). A flow built to load one of the
    returned checkpoints must have _w_mean and _w_std set to the values of the trained flow, otherwise they stay at
    the defaults (0.0, 1.0) and every density evaluation is wrong.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    w (np.ndarray): Covariate, shape (n,), any real scale (standardised internally).
    n_epochs (int): Number of epochs (the maximum with early stopping). Default 1500.
    batch_size (int): Minibatch size, shared with CODINE and C-CODINE (see codine.train_codine). Reference
        implementations of flows use 100 to 2048 on much larger data sets: Papamakarios et al. (2017) use 100, and
        Kamthe et al. (2021) use 512 to 2048 on 29,556 to 1,615,917 training points (their Table 4). Default 32.
    lr (float): Adam learning rate. Default 1e-3.
    seed (int): Torch and numpy seed. Default 42.
    n_layers (int): Number of coupling layers. Default 8.
    hidden (int): Hidden layer width per coupling layer. Default 64.
    scale_bound (float): Log-scale bound per affine coupling layer. Default 3.5.
    conditioning_mode (str): 'w_only', 'spline_basis' or 'film', see ConditionalAffineCoupling. Default 'w_only'.
    coupling_type (str): 'affine' or 'spline'. Default 'affine'.
    num_bins (int): Spline bins per layer, used only if coupling_type is 'spline'. Default 8.
    tail_bound (float): Spline domain [-tail_bound, tail_bound], used only if coupling_type is 'spline'. Default 4.0.
    min_derivative (float): Floor on the spline's local derivative, used only if coupling_type is 'spline'.
        Default DEFAULT_MIN_DERIVATIVE.
    max_derivative (float): Ceiling on the spline's local derivative, used only if coupling_type is 'spline'. It
        caps the height of a density peak the spline can represent. Default DEFAULT_MAX_DERIVATIVE.
    grad_clip_norm (float): Maximum L2 norm of the gradient per step. If None, no clipping. Default 5.0.
    weight_decay (float): Adam weight decay. Default 0.0.
    checkpoint_every (int): Epochs between checkpoints (stored weights, or validation checks). Default 100.
    show_progress (bool): Show the tqdm epoch bar. Pass False when this runs in a ProcessPoolExecutor, because the
        bars of concurrent processes interleave. Default True.
    val_u1 (np.ndarray): First variable's validation pseudo-observations. Optional.
    val_u2 (np.ndarray): Second variable's validation pseudo-observations. Optional.
    val_w (np.ndarray): Validation covariate values. Optional.
    patience (int): Number of consecutive non-improving checkpoints tolerated. Early stopping needs val_u1, val_u2,
        val_w and patience. With any of them missing, all n_epochs are run, val_cvpe is empty and best_epoch is
        None. Default None.
    grid_n (int): Number of Gauss-Legendre nodes of the validation CVPE, used only with validation data. A smaller
        value (for example 100) makes repeated calls cheaper and the stopping decision noisier, not the reported
        metric. Default copula_families.QUAD_N.
    position (int): Row of the tqdm bar, so that concurrent workers can draw to separate lines (best effort, not
        every Jupyter frontend redraws in place). Only meaningful with show_progress=True. Default None.
    margin_every (int): The penalty is added on every margin_every-th minibatch with its weight multiplied by margin_every. Default 1.
    margin_penalty (float): Weight lambda of codine.margin_uniformity_penalty, added to the negative log-likelihood of every minibatch. The
        density is capped at MAX_PENALTY_DENSITY inside the penalty. 0 turns it off. Default 0.0.
    margin_n_w (int): Number of covariate values of the minibatch on which the penalty is evaluated. Default 8.
    margin_grid (int): Grid points per coordinate of the penalty. Default 20.

    Returns:
    tuple: (flow, losses, checkpoints, val_cvpe, best_epoch). losses is the per-epoch mean training loss without the margin penalty.
        checkpoints maps epoch to state_dict (only the best weights with early stopping). val_cvpe maps epoch to
        validation CVPE and is empty, and best_epoch is None, if no validation data were given.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(u1)
    U = torch.tensor(np.column_stack([u1, u2]), dtype=torch.float32)
    W = torch.tensor(np.asarray(w, dtype=np.float32))

    w_mean = float(W.mean())
    w_std = float(W.std()) if float(W.std()) > 1e-8 else 1.0

    spline_knots = None
    if conditioning_mode == 'spline_basis':
        w_std_train = (W - w_mean) / w_std
        spline_knots = torch.quantile(w_std_train, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))

    flow = ConditionalCopulaFlow(n_layers=n_layers, hidden=hidden, scale_bound=scale_bound,
                                  conditioning_mode=conditioning_mode, coupling_type=coupling_type,
                                  num_bins=num_bins, tail_bound=tail_bound, spline_knots=spline_knots,
                                  min_derivative=min_derivative, max_derivative=max_derivative)
    flow._w_mean = w_mean
    flow._w_std = w_std

    use_early_stop = (val_u1 is not None and val_u2 is not None
                       and val_w is not None and patience is not None)
    if use_early_stop:
        from model_evaluation import cvpe_only

    optimizer = optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    flow.train()
    losses = []
    checkpoints = {}
    val_cvpe = {}
    best_epoch = None
    best_val_cvpe = float('inf')
    epochs_since_improve = 0

    pbar = tqdm(range(1, n_epochs + 1), desc='Training conditional copula flow',
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
            optimizer.zero_grad()
            log_c = flow.log_density(u_batch, w_batch)
            loss = -log_c.mean()
            base_loss = loss.item()
            if margin_penalty > 0 and batch_step % margin_every == 0:
                loss = loss + margin_every * margin_penalty * margin_uniformity_penalty(
                    lambda u, ww: torch.exp(flow.log_density(u, ww).clamp(max=math.log(MAX_PENALTY_DENSITY))),
                    w_batch[:margin_n_w], margin_grid, normalise=False)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_loss += base_loss
            n_batches += 1
            batch_step += 1
        losses.append(epoch_loss / n_batches)
        if epoch % max(1, n_epochs // 20) == 0:
            pbar.set_postfix(nll=f"{losses[-1]:.4f}")

        if epoch % checkpoint_every == 0 or epoch == n_epochs:
            flow.eval()
            # With early stopping only the current winner's weights are kept (every checkpoint's CVPE stays in
            # val_cvpe). Without it, the weights of every checkpoint are stored.
            if not use_early_stop:
                checkpoints[epoch] = copy.deepcopy(flow.state_dict())

            if use_early_stop:
                with torch.no_grad():
                    density_fn = lambda pts, wv: eval_flow_density(flow, pts, wv)
                    cvpe_here = cvpe_only(density_fn, val_u1, val_u2, val_w, grid_n=grid_n)
                val_cvpe[epoch] = cvpe_here
                if cvpe_here < best_val_cvpe - 1e-6:
                    best_val_cvpe, best_epoch, epochs_since_improve = cvpe_here, epoch, 0
                    checkpoints.clear()
                    checkpoints[epoch] = copy.deepcopy(flow.state_dict())
                else:
                    epochs_since_improve += 1
                if show_progress:
                    pbar.set_postfix(nll=f"{losses[-1]:.4f}", val_cvpe=f"{cvpe_here:.5f}")
                if epochs_since_improve >= patience:
                    if show_progress:
                        log.info(f"[early_stop] no improvement for {patience} checkpoints "
                              f"({patience * checkpoint_every} epochs), stopping at "
                              f"epoch {epoch} (best={best_epoch}, "
                              f"val_CVPE={best_val_cvpe:.5f})")
                    flow.train()
                    break

            flow.train()

    flow.eval()
    return flow, losses, checkpoints, val_cvpe, best_epoch


def train_conditional_copula_flow_early_stop(u1, u2, w, b, val_frac=0.25, max_epochs=8000,
                                              checkpoint_every=100, seed=42, grid_n=QUAD_N,
                                              patience=5, verbose=True, **train_kwargs):
    """Train a ConditionalCopulaFlow with early stopping on the CVPE of a held-out validation block.

    The validation block is carved out of u1, u2 and w with the purge-gap chronological split
    (data_split.train_test_split_chrono, purge = b), as in codine.train_conditional_codine_early_stop.
    The CVPE check runs inside train_conditional_copula_flow at every checkpoint. After training, the best checkpoint
    is reloaded, because training may run patience checkpoints past it.

    torch.set_num_threads(1) is set at the start, so the function is safe to submit to a ProcessPoolExecutor.

    Parameters:
    u1 (np.ndarray): First variable's training-portion pseudo-observations only.
    u2 (np.ndarray): Second variable's training-portion pseudo-observations only.
    w (np.ndarray): Training-portion covariate values only.
    b (float): Purge gap length (pass b=1 for i.i.d. synthetic data).
    val_frac (float): Fraction reserved for validation. Default 0.25.
    max_epochs (int): Maximum number of epochs. Default 8000.
    checkpoint_every (int): Epochs between validation checks. Default 100.
    seed (int): Training seed. Default 42.
    grid_n (int): Number of Gauss-Legendre nodes of the validation CVPE, used for the stopping decision only.
        Default copula_families.QUAD_N.
    patience (int): Number of consecutive non-improving checkpoints tolerated. Default 5.
    verbose (bool): Log the one-line summary at the end. Pass False when many calls run in a ProcessPoolExecutor.
        Default True.
    **train_kwargs: Forwarded to train_conditional_copula_flow (n_layers, hidden, scale_bound, conditioning_mode,
        coupling_type, num_bins, tail_bound, min_derivative, max_derivative, grad_clip_norm, weight_decay,
        batch_size, lr). show_progress defaults to False.

    Returns:
    dict: {'flow' (best checkpoint, reloaded), 'losses', 'checkpoints', 'val_cvpe' (epoch -> validation CVPE, up to
        the early exit), 'best_epoch', 'best_val_cvpe', 'sub_train_mask', 'val_mask'}.
    """
    torch.set_num_threads(1)

    train_kwargs.setdefault('show_progress', False)

    n = len(u1)
    sub_train_mask, val_mask = train_test_split_chrono(n, test_frac=val_frac, purge=int(round(b)), one_sided=True)
    u1_sub, u2_sub, w_sub = u1[sub_train_mask], u2[sub_train_mask], w[sub_train_mask]
    u1_val, u2_val, w_val = u1[val_mask], u2[val_mask], w[val_mask]

    flow, losses, checkpoints, val_cvpe, best_epoch = train_conditional_copula_flow(
        u1_sub, u2_sub, w_sub, n_epochs=max_epochs, seed=seed,
        checkpoint_every=checkpoint_every,
        val_u1=u1_val, val_u2=u2_val, val_w=w_val,
        patience=patience, grid_n=grid_n,
        **train_kwargs)

    if best_epoch is None:
        raise RuntimeError(
            "early stopping found no finite validation CVPE at any checkpoint (best_epoch is None): the "
            "validation CVPE was NaN/inf, the run diverged or the density collapsed. Nothing is returned "
            "rather than a network picked without a criterion.")
    best_val_cvpe = val_cvpe[best_epoch]

    n_layers = train_kwargs.get('n_layers', 8)
    hidden = train_kwargs.get('hidden', 64)
    scale_bound = train_kwargs.get('scale_bound', 3.5)
    conditioning_mode = train_kwargs.get('conditioning_mode', 'w_only')
    coupling_type = train_kwargs.get('coupling_type', 'affine')
    num_bins = train_kwargs.get('num_bins', 8)
    tail_bound = train_kwargs.get('tail_bound', 4.0)
    min_derivative = train_kwargs.get('min_derivative', DEFAULT_MIN_DERIVATIVE)
    max_derivative = train_kwargs.get('max_derivative', DEFAULT_MAX_DERIVATIVE)
    spline_knots = (flow.layers[0].spline_knots
                     if conditioning_mode == 'spline_basis' else None)

    # Reload the best checkpoint (training may have run patience checkpoints past it).
    best_flow = ConditionalCopulaFlow(n_layers=n_layers, hidden=hidden, scale_bound=scale_bound,
                                       conditioning_mode=conditioning_mode, coupling_type=coupling_type,
                                       num_bins=num_bins, tail_bound=tail_bound, spline_knots=spline_knots,
                                       min_derivative=min_derivative, max_derivative=max_derivative)
    best_flow._w_mean, best_flow._w_std = flow._w_mean, flow._w_std
    best_flow.load_state_dict(checkpoints[best_epoch])
    best_flow.eval()

    if verbose:
        log.info(f"[early_stop] best epoch={best_epoch}  val_CVPE={best_val_cvpe:.5f}  "
              f"(stopped at {max(val_cvpe.keys())}/{max_epochs}, "
              f"sub_train n={sub_train_mask.sum()}, val n={val_mask.sum()})")

    return {
        'flow': best_flow, 'losses': losses, 'checkpoints': checkpoints,
        'val_cvpe': val_cvpe, 'best_epoch': best_epoch, 'best_val_cvpe': best_val_cvpe,
        'sub_train_mask': sub_train_mask, 'val_mask': val_mask,
    }


# ── evaluation and diagnostics ──────────────────────────────────────────────

def eval_flow_density(flow: ConditionalCopulaFlow, u: np.ndarray, w) -> np.ndarray:
    """Evaluate the density c(u | w) of a trained flow at the points u.

    The density is exact, so no calibration step is needed.

    Parameters:
    flow (ConditionalCopulaFlow): A trained flow.
    u (np.ndarray): (n, 2) points in [0, 1]^2.
    w (np.ndarray or float): Covariate value(s), shape (n,), or a single float used for all n points.

    Returns:
    np.ndarray: (n,) density values.
    """
    flow.eval()
    u = np.asarray(u, dtype=np.float32)
    n = u.shape[0]
    w_arr = np.full(n, w, dtype=np.float32) if np.isscalar(w) else np.asarray(w, dtype=np.float32)
    with torch.no_grad():
        u_t = torch.tensor(u, dtype=torch.float32)
        w_t = torch.tensor(w_arr, dtype=torch.float32)
        log_c = flow.log_density(u_t, w_t)
    return torch.exp(log_c).numpy()


def sample_from_flow(flow: ConditionalCopulaFlow, w_value: float, n_samples: int = 20_000,
                      seed: int = 1) -> np.ndarray:
    """Draw samples from the flow at a fixed w, directly (no grid or rejection step).

    Parameters:
    flow (ConditionalCopulaFlow): A trained flow.
    w_value (float): The covariate value to condition on.
    n_samples (int): Number of samples. Default 20,000.
    seed (int): Sampling seed. Default 1.

    Returns:
    np.ndarray: (n_samples, 2) sampled pairs.
    """
    flow.eval()
    w_t = torch.full((n_samples,), float(w_value), dtype=torch.float32)
    with torch.no_grad():
        u = flow.sample(w_t, seed=seed)
    return u.numpy()


def flow_kendall_tau(flow: ConditionalCopulaFlow, w_value: float,
                      n_samples: int = 20_000, seed: int = 1) -> float:
    """Kendall's tau at a fixed w, estimated from samples drawn from the flow.

    Parameters:
    flow (ConditionalCopulaFlow): A trained flow.
    w_value (float): The covariate value to condition on.
    n_samples (int): Number of samples to draw. Default 20,000.
    seed (int): Sampling seed. Default 1.

    Returns:
    float: Estimated Kendall's tau at w_value.
    """
    samples = sample_from_flow(flow, w_value, n_samples=n_samples, seed=seed)
    tau, _ = kendalltau(samples[:, 0], samples[:, 1])
    return float(tau)


def flow_calibration_curve(flow: ConditionalCopulaFlow, w_grid: np.ndarray,
                            n_samples: int = 20_000, seed: int = 1) -> np.ndarray:
    """Kendall's tau-hat(w) evaluated across a grid of w values, the flow's calibration curve.

    Parameters:
    flow (ConditionalCopulaFlow): A trained flow.
    w_grid (np.ndarray): Covariate values at which to evaluate tau-hat.
    n_samples (int): MC samples per w value. Default 20,000.
    seed (int): Sampling seed. Default 1.

    Returns:
    np.ndarray: tau-hat evaluated at each point in w_grid.
    """
    return np.array([flow_kendall_tau(flow, float(wv), n_samples=n_samples, seed=seed)
                      for wv in w_grid])


def flow_self_consistency_test(flow: ConditionalCopulaFlow, w_values: list,
                                n_mc: int = 10_000, n_seeds: int = 10, seed: int = 0,
                                tol: float = 0.01) -> dict:
    """Check that the flow's density integrates to 1 at each w in w_values.

    The integral is estimated by Monte Carlo over uniform draws on the unit square, with n_seeds independent seeds
    per w. A w passes if |mean estimate - 1| <= tol, the same effect-size rule as in the CODINE self-consistency tests (the
    standard error and z-score are diagnostics only). The density is exact by
    construction, so the test should pass everywhere. It checks the change-of-variables bookkeeping of the
    implementation and is not a validity check of the estimator, as the self-consistency test is for CODINE.

    Parameters:
    flow (ConditionalCopulaFlow): A trained flow.
    w_values (list): Covariate values to test at.
    n_mc (int): Number of uniform draws per seed and w. Default 10,000.
    n_seeds (int): Number of independent seeds per w. Default 10.
    seed (int): First seed. Default 0.
    tol (float): Tolerance on |mean - 1|. Default 0.01.

    Returns:
    dict: Keyed by w, each entry a dict with 'integral', 'se', 'z', 'deviation', 'tol' and 'passed', plus the overall flag 'all_passed'.
    """
    results = {}
    all_passed = True
    for wv in w_values:
        integrals = []
        for s in range(seed, seed + n_seeds):
            rng = np.random.default_rng(s)
            u_mc = rng.uniform(0, 1, size=(n_mc, 2)).astype(np.float32)
            c_vals = eval_flow_density(flow, u_mc, float(wv))
            integrals.append(float(c_vals.mean()))
        integrals = np.array(integrals)
        mean_integral = float(integrals.mean())
        se = float(integrals.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else np.nan
        z = float((mean_integral - 1.0) / se) if se > 0 else np.inf
        deviation = abs(mean_integral - 1.0)
        passed = deviation <= tol
        all_passed = all_passed and passed
        results[float(wv)] = {"integral": mean_integral, "se": se, "z": z, "deviation": deviation, "tol": tol, "passed": passed}
        log.info(f"  w={wv:+.3f}:  E[ĉ|w] = {mean_integral:.4f}  (SE={se:.4f}, z={z:.2f})  "
              f"{'PASS ✓' if passed else 'FAIL ✗'}")
    results["all_passed"] = all_passed
    return results


def plot_flow_training_loss(losses: list, ax=None) -> None:
    """Plot the negative log-likelihood over the training epochs.

    Parameters:
    losses (list): Per-epoch mean training losses, as returned by train_conditional_copula_flow.
    ax (matplotlib.axes.Axes): Axes to plot on. If None, a new figure is created. Optional.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses, lw=1, color='grey')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Conditional copula flow training loss")


def diagnose_spline_flow(flow, u1, u2, w, tail_bound=4.0, sample_size=2000, seed=0,
                          variant_name=""):
    """Diagnose a trained spline-coupling flow: tail-bound coverage and derivative saturation.

    1. Tail-bound coverage: the fraction of x = Phi^-1(u) outside [-tail_bound, tail_bound], which the spline
       passes through the identity map whatever it has learned.
    2. Derivative saturation: the fraction of the learned spline derivatives (evaluated on the supplied data) within
       5% of the flow's own max_derivative, that is, pinned at the ceiling.

    For a flow with coupling_type other than 'spline' the checks are meaningless, so a message is logged and None
    is returned.

    Parameters:
    flow (ConditionalCopulaFlow): A trained flow.
    u1 (np.ndarray): First variable's pseudo-observations (the data the flow was fitted to).
    u2 (np.ndarray): Second variable's pseudo-observations.
    w (np.ndarray): Covariate values.
    tail_bound (float): Should match the flow's own tail_bound. Default 4.0.
    sample_size (int): Number of points sampled for the check. Default 2000.
    seed (int): Seed of the point sample. Default 0.
    variant_name (str): Label used in log messages and plot titles. Default "".

    Returns:
    dict or None: {'frac_x_outside_tail_bound', 'max_abs_x', 'frac_derivative_saturated', 'derivative_p99', 'fig'},
        or None if the flow does not use spline coupling.
    """
    if flow.coupling_type != 'spline':
        log.info(f"{variant_name}: coupling_type={flow.coupling_type!r}, not 'spline' -- skipping.")
        return None

    rng = np.random.default_rng(seed)
    n = len(u1)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    u = np.stack([np.asarray(u1)[idx], np.asarray(u2)[idx]], axis=1).astype(np.float32)
    w_s = np.asarray(w)[idx].astype(np.float32)
    u_t = torch.as_tensor(u)
    w_t = torch.as_tensor(w_s)

    x_t = normal_icdf(u_t)
    frac_outside = (x_t.abs() > tail_bound).float().mean().item()
    max_abs_x = x_t.abs().max().item()

    w_std = flow._standardize_w(w_t)
    x0, x1 = x_t[:, 0].clone(), x_t[:, 1].clone()
    all_derivs = []
    min_d, max_d = flow.layers[0].min_derivative, flow.layers[0].max_derivative
    log_min_d, log_max_d = math.log(min_d), math.log(max_d)
    with torch.no_grad():
        for layer, t_idx in zip(reversed(list(flow.layers)), reversed(flow._transform_idx)):
            cond, target = (x0, x1) if t_idx == 1 else (x1, x0)
            _, _, raw_deriv = layer._spline_params(cond, w_std)
            deriv = torch.exp(log_min_d + (log_max_d - log_min_d) * torch.sigmoid(raw_deriv))
            all_derivs.append(deriv.flatten().numpy())
            if t_idx == 1:
                x1, _ = layer.inverse(x0, x1, w_std)
            else:
                x0, _ = layer.inverse(x1, x0, w_std)

    all_derivs = np.concatenate(all_derivs)
    frac_saturated = float((all_derivs > 0.95 * max_d).mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(x_t.numpy().ravel(), bins=60, color='steelblue', alpha=0.8)
    axes[0].axvline(tail_bound, color='red', ls='--', label=f'tail_bound=±{tail_bound}')
    axes[0].axvline(-tail_bound, color='red', ls='--')
    axes[0].set_title(f'{variant_name}\nx=Φ⁻¹(u): {frac_outside:.2%} outside tail_bound')
    axes[0].set_xlabel('x'); axes[0].legend()

    axes[1].hist(all_derivs, bins=60, color='darkorange', alpha=0.8)
    axes[1].axvline(max_d, color='red', ls='--',
                     label=f'max_derivative={max_d}')
    axes[1].set_title(f'{variant_name}\nderivatives: {frac_saturated:.2%} within 5% of ceiling')
    axes[1].set_xlabel('derivative value'); axes[1].legend()
    plt.tight_layout()

    return {'frac_x_outside_tail_bound': frac_outside, 'max_abs_x': max_abs_x,
            'frac_derivative_saturated': frac_saturated,
            'derivative_p99': float(np.quantile(all_derivs, 0.99)), 'fig': fig}


def make_flow_sample_fn(flow_obj):
    """Return a sampler sample_fn(wv, n, s) bound to one fitted flow.

    The factory makes each closure keep its own flow. A plain lambda built inside a loop over several flows would
    use whichever flow was assigned last. It serves the real-data marginal-uniformity check and, through
    synthetic_study.py, the synthetic positive control, so both use one definition.

    Parameters:
    flow_obj (ConditionalCopulaFlow): A trained flow.

    Returns:
    callable: sample_fn(wv, n, s) -> np.ndarray, drawing n samples at covariate value wv with seed s.
    """
    return lambda wv, n, s: sample_from_flow(flow_obj, wv, n_samples=n, seed=s)
