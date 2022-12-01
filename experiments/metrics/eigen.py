import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy


def cosine_sim(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between input vectors u, v.
    If u, v are matrices, computes the cosine similarity between any two column vectors of u and v.

    Args:
        u (torch.Tensor): Vector or matrix.
        v (torch.Tensor): Vector or matrix.

    Returns:
        torch.Tensor: Cosine similarity.
    """
    if len(u.shape) == 1:
        u = u.reshape(*u.shape, 1)
    if len(v.shape) == 1:
        v = v.reshape(*v.shape, 1)
    u /= torch.norm(u, dim=0)
    v /= torch.norm(v, dim=0)

    cosine_similarities = torch.einsum("ij,jk->ik", u.T, v)
    return cosine_similarities


def eigen_dissim(
    u: torch.Tensor, v: torch.Tensor, su=None, sv=None, metric="l2", reduce=True
) -> torch.Tensor:
    """Give the dissimilarity of eigenvectors u, v, weighted by their eigenvalue magnitueds s, s_

    Args:
        u (torch.Tensor): Vector or matrix.
        v (torch.Tensor): Vector or matrix.
        su (_type_, optional): Weighting for u. Defaults to None.
        sv (_type_, optional):  Weighting for v. Defaults to None.
        metric (str, optional): Metric that will be used for the calculation of dissimilarity. Defaults to "l2".
        reduce (bool, optional): If True it will calculate the mean. Defaults to True.

    Returns:
        torch.Tensor: _description_
    """
    if len(u.shape) == 1:
        u = u.reshape(*u.shape, 1)
    if len(v.shape) == 1:
        v = v.reshape(*v.shape, 1)
    N = u.shape[-1]
    if su is not None and sv is not None:
        assert len(su.shape) <= 1 and len(sv.shape) <= 1
        weights = torch.einsum("i,j->ij", su, sv)
    else:
        weights = 1

    cosine_similarities = torch.abs(torch.einsum("ij,jk->ik", u.T, v))
    kronecker_delta = torch.eye(N, device=u.device)
    if metric == "bce":
        y, t = cosine_similarities, kronecker_delta
        if reduce:
            dissim = binary_cross_entropy(y, t, reduction="sum").item()
        else:
            dissim = binary_cross_entropy(y, t, reduction="none")
    elif metric == "ce":
        y = cosine_similarities
        t = torch.argmax(kronecker_delta, dim=1)
        if reduce:
            dissim = cross_entropy(y, t, reduction="sum").item()
        else:
            dissim = cross_entropy(y, t, reduction="sum")
    elif metric == "l1":
        deviations = (kronecker_delta - cosine_similarities).abs()
        dissim = weights * deviations
        if reduce:
            dissim = torch.sum(dissim)
    elif metric == "l2":
        deviations = kronecker_delta - cosine_similarities
        dissim = weights * deviations ** 2
        if reduce:
            dissim = torch.sum(dissim)
    if reduce:
        dissim /= N ** 2
    return dissim


def subspace_sim(V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Computes subspace similarity of V,W.
    We suppose that V, W contain an orthogonal basis and have the same dimension d. We calculate:
    sim(V,W)=Tr(P_V*P_W)/Sqrt(Tr(P_V)Tr(P_W))
    Now: Tr(P_V)=Tr(P_W)=d and
    Tr(P_V*P_W)=Tr((V*V.T)*(W*W.T))=Tr((W.T*V)*(V.T*W))=||flatten(V.T*W)||_2^2

    Args:
        V (torch.Tensor): Subspace V as matrix of basis vectors.
        W (torch.Tensor): Subspace W as matrix of basis vectors.

    Returns:
        torch.Tensor: Subspace similarity.
    """
    assert V.shape[0] == W.shape[0], "V, W must be subspaces of the same vector space"
    V /= torch.linalg.norm(V, dim=0)
    W /= torch.linalg.norm(W, dim=0)
    d = min([V.shape[1], W.shape[1]])
    return ((V.T @ W).reshape(-1) ** 2).sum() / d


def random_baseline_subsim(V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Compute expected overlap of random vectorspaces V',W' with
    dim(V') = dim(V) and dim(W') = dim(W)

    Args:
        V (torch.Tensor): Subspace V as matrix of basis vectors.
        W (torch.Tensor): Subspace V as matrix of basis vectors.

    Returns:
        torch.Tensor: Subspace similarity for random baseline
    """

    assert V.shape[0] == W.shape[0], "V, W must be subspaces of the same vector space"
    return max([V.shape[1], W.shape[1]]) / V.shape[0]
