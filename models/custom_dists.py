from typing import Dict
import math

from torch.distributions.utils import _standard_normal, lazy_property
import torch
from torch.distributions import Categorical, constraints
from torch.distributions.distribution import Distribution
from torch_scatter import segment_coo
import torch.nn as nn
from pykeops.torch import Vi, Vj, LazyTensor


__all__ = ["MultivariateNormal","MixtureSameFamily"]

def compile_with_dynamic_shapes(fn):
    if torch.cuda.is_available():
        torch._dynamo.config.dynamic_shapes = True
        return torch.compile(fn)
    else:
        return fn

class MixtureSameFamily(Distribution):
    r"""
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all component are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` component) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.

    Examples::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        >>> # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)

        >>> # Construct Gaussian Mixture Model in 2D consisting of 5 equally
        >>> # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
        ...          torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

        >>> # Construct a batch of 3 Gaussian Mixture Models in 2D each
        >>> # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
        ...         torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting component.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}
    has_rsample = False

    def __init__(
        self, mixture_distribution,
            component_distribution,
            validate_args=None,
            scal_mults=None,
            relu=False,
            cell=None,
            image_weights=None,
    ):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution
        self.scal_mults = scal_mults
        self.relu = relu
        self.cell = cell
        self.image_weights = image_weights

        if not isinstance(self._mixture_distribution, Categorical):
            raise ValueError(
                " The Mixture distribution needs to be an "
                " instance of torch.distributions.Categorical"
            )

        if not isinstance(self._component_distribution, Distribution):
            raise ValueError(
                "The Component distribution need to be an "
                "instance of torch.distributions.Distribution"
            )

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError(
                    f"`mixture_distribution.batch_shape` ({mdbs}) is not "
                    "compatible with `component_distribution."
                    f"batch_shape`({cdbs})"
                )

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError(
                f"`mixture_distribution component` ({km}) does not"
                " equal `component_distribution.batch_shape[-1]`"
                f" ({kc})"
            )
        self._num_component = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super().__init__(
            batch_shape=cdbs, event_shape=event_shape, validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        batch_shape_comp = batch_shape + (self._num_component,)
        new = self._get_checked_instance(MixtureSameFamily, _instance)
        new._component_distribution = self._component_distribution.expand(
            batch_shape_comp
        )
        new._mixture_distribution = self._mixture_distribution.expand(batch_shape)
        new._num_component = self._num_component
        new._event_ndims = self._event_ndims
        event_shape = new._component_distribution.event_shape
        super(MixtureSameFamily, new).__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        # FIXME this may have the wrong shape when support contains batched
        # parameters
        return self._component_distribution.support

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def component_distribution(self):
        return self._component_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(
            probs * self.component_distribution.mean, dim=-1 - self._event_ndims
        )  # [B, E]

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(
            probs * self.component_distribution.variance, dim=-1 - self._event_ndims
        )
        var_cond_mean = torch.sum(
            probs * (self.component_distribution.mean - self._pad(self.mean)).pow(2.0),
            dim=-1 - self._event_ndims,
        )
        return mean_cond_var + var_cond_mean

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs

        return torch.sum(cdf_x * mix_prob, dim=-1)

    #@compile_with_dynamic_shapes
    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)

        # Log mixture probabilities [B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits, dim=-1)  # [B, k]

        A = self.component_distribution.covariance_matrix.reshape(-1, 3, 3)
        L = torch.linalg.cholesky(A)  # or use self.component_distribution._unbroadcasted_scale_tril if available
        half_log_det = torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # shape (M,)
        gamma = A.inverse().reshape(-1, 9) / 2
        mu = self.component_distribution.loc.to(torch.float32)
        M = mu.shape[0]
        N = x.shape[0]
        if self.scal_mults is not None:
            weight_tensor = (self.scal_mults * torch.exp(log_mix_prob) *
                         torch.exp(-half_log_det) * (2 * torch.pi) ** (-3 / 2))
        else:
            weight_tensor = (torch.exp(log_mix_prob) *
                         torch.exp(-half_log_det) * (2 * torch.pi) ** (-3 / 2))
        weight_tensor = weight_tensor.contiguous()
        if self.cell is not None:
            # --- 27-Image Convention ---
            lattice_indices = torch.tensor(
                [[i, j, k] for i in [-1, 0, 1]
                 for j in [-1, 0, 1]
                 for k in [-1, 0, 1]],
                dtype=mu.dtype,
                device=mu.device
            )  # shape: (27, 3)
            n_images = lattice_indices.shape[0]
            shift_vectors = lattice_indices @ self.cell.to(torch.float32)  # shape: (27, 3)
            # Create shifted versions of mu:
            mu_shifted = mu.unsqueeze(0) + shift_vectors.unsqueeze(1)  # shape: (27, M, 3)

            # LazyTensor setup:
            x_i = LazyTensor(x.view(1, N, 1, 3))  # shape: (1, N, 1, 3)
            mu_j = LazyTensor(mu_shifted.view(n_images, 1, M, 3))  # shape: (27, 1, M, 3)
            gamma_j = LazyTensor(gamma.view(1, 1, M, gamma.size(1)))  # shape: (1, 1, M, 9); broadcasts over 27
            wt = weight_tensor.view(M, 1)
            # Compute weighted squared distances:
            diff_ij = x_i - mu_j  # shape: (27, N, M, 3)
            sqdist_ij = diff_ij.weightedsqnorm(gamma_j)  # shape: (27, N, M, 1)
            K_ij = (-sqdist_ij).exp()  # shape: (27, N, M, 1)

            # Multiply kernel values by weight_tensor and sum over the 27 shifts:
            contrib_per_shift = K_ij @ wt  # shape: (27, N, 1)
            weighted = contrib_per_shift.sum(dim=0).squeeze(-1)  # shape: (N,)
        else:
            component_log_prob = (-Vi(x.squeeze(1)).weightedsqdist(Vj(mu), Vj(gamma))).exp()
            src = component_log_prob @ weight_tensor
            weighted = src
        if self.relu:
            p = torch.relu(weighted)  # [S, B]
        else:
            p = weighted
        return p

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
            )

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(
            xs[:-1]
            + torch.Size(pad_ndims * [1])
            + xs[-1:]
            + torch.Size(self._event_ndims * [1])
        )
        return x

    def __repr__(self):
        args_string = (
            f"\n  {self.mixture_distribution},\n  {self.component_distribution}"
        )
        return "MixtureSameFamily" + "(" + args_string + ")"

def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)

@compile_with_dynamic_shapes
def _batch_mahalanobis(bL,
                       bx
                       ):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c

    M_swap = (
        torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
    )  # shape = b x c

    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    Id = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    L = torch.linalg.solve_triangular(L_inv, Id, upper=False)
    return L


class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precision matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal

    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.

        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
        r_mat=None
    ):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                covariance_matrix.shape[:-2], loc.shape[:-1]
            )
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
        else:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
        self.loc = loc.expand(batch_shape + (-1,))

        event_shape = self.loc.shape[-1:]
        self.r_mat = r_mat
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(MultivariateNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(
            self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return (
            self._unbroadcasted_scale_tril.pow(2)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

    #@compile_with_dynamic_shapes
    def log_prob(self, value, links, cell):
        if self._validate_args:
            self._validate_sample(value)
        if links is None:
            diff = value - self.loc
            M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
            half_log_det = (
                self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            )
            return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det
        else:
            if cell is not None:
                # Instead of computing diff = value - self.loc[links[:, 1]]
                value = value.squeeze(1)[links[:, 0]]
                raw_diff = value - self.loc[links[:, 1]]
                # cell should be provided or extracted from the ASE Atoms object
                d_min = minimum_image_vector(raw_diff, cell)
                # Now use d_min to compute the Mahalanobis distance
                M = _batch_mahalanobis(self._unbroadcasted_scale_tril[links[:, 1]], d_min)
            else:
                value = value.squeeze(1)[links[:, 0]]
                diff = value - self.loc[links[:, 1]]

                M = _batch_mahalanobis(self._unbroadcasted_scale_tril[links[:, 1]], diff)
            half_log_det = (
                self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            )
            log_p = -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det[links[:, 1]]
            return log_p

    def entropy(self):
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)

def minimum_image_vector(d, cell):
    """
    Compute the minimum image of the difference vector d for a given cell.

    Args:
        d (torch.Tensor): Difference vector of shape (..., 3).
        cell (torch.Tensor): Cell matrix of shape (3, 3) with lattice vectors as rows.

    Returns:
        torch.Tensor: The minimum image difference vector, of shape (..., 3).
    """
    inv_cell = torch.inverse(cell)  # shape (3,3)
    # Convert to fractional coordinates
    f = torch.matmul(d, inv_cell)
    # Wrap into the interval [-0.5, 0.5]
    f_wrapped = f - differentiable_round(f)
    # Convert back to Cartesian coordinates
    d_min = torch.matmul(f_wrapped, cell)
    return d_min
def get_all_mu_images(mu: torch.Tensor, cell) -> torch.Tensor:
    """
    Generate all periodic images of the mu vectors using the real-space lattice vectors
    from self.cell. This function assumes self.cell is a (3,3) tensor containing the
    lattice vectors (one per row).

    Args:
        mu (torch.Tensor): Tensor of shape (M, 3) containing the mu vectors (Gaussian centers).

    Returns:
        torch.Tensor: Tensor of shape (27, M, 3) containing all periodic images of mu.
    """
    # Create all combinations of lattice index shifts (-1, 0, 1) in each dimension.
    device = mu.device  # Ensure the tensor is on the same device as mu
    lattice_indices = torch.tensor(
        [[i, j, k] for i in [-1, 0, 1]
                   for j in [-1, 0, 1]
                   for k in [-1, 0, 1]],
        dtype=mu.dtype,
        device=device
    )  # Shape: (27, 3)

    # Calculate the actual shift vectors by multiplying the index combinations with the cell lattice vectors.
    # Here, self.cell is expected to be of shape (3, 3).
    shift_vectors = lattice_indices @ cell  # Resulting shape: (27, 3)

    # Add each shift to every mu vector.
    # mu.unsqueeze(0) expands mu to shape (1, M, 3)
    # shift_vectors.unsqueeze(1) expands shift_vectors to shape (27, 1, 3)
    # The result broadcasts to (27, M, 3), i.e. 27 images for each mu.
    mu_images = mu.unsqueeze(0) + shift_vectors.unsqueeze(1)

    return mu_images


def get_lattice_shifts(cell: torch.Tensor,
                       shell_size: int = 1,
                       max_dist: float = None) -> torch.Tensor:
    """
    Generate integer lattice shifts in the range [-shell_size, shell_size],
    multiply by `cell` to get shift vectors, and optionally filter by `max_dist`.

    Args:
        cell (torch.Tensor): (3, 3) cell matrix
        shell_size (int): how many lattice repeats in each direction, e.g. 1 => [-1..1]
        max_dist (float): if provided, shift vectors whose length exceeds this distance are discarded

    Returns:
        torch.Tensor: shape (K, 3) containing the retained shift vectors
    """
    device = cell.device
    shifts = []
    for i in range(-shell_size, shell_size + 1):
        for j in range(-shell_size, shell_size + 1):
            for k in range(-shell_size, shell_size + 1):
                offsets = torch.tensor([i, j, k], dtype=cell.dtype, device=device)
                shift_vec = offsets @ cell  # shape (3,)
                if max_dist is not None:
                    dist = shift_vec.norm()
                    if dist > max_dist:
                        # skip if beyond cutoff
                        continue
                shifts.append(shift_vec)
    if len(shifts) == 0:
        # as a fallback, always include the no-shift
        shifts.append(torch.zeros(3, dtype=cell.dtype, device=device))
    return torch.stack(shifts, dim=0)  # shape (K, 3)
