"""GPTAQ (arXiv 2504.02692, ICML 2025) on top of the group-wise symmetric GPTQ of gptq_dense_layerwise_26b.py: the layer is calibrated so that its output on the QUANTISED-stream input X matches the output the FULL-PRECISION model gives on its own input X~ (W X~), not W X.
Extra input compared with GPTQ: D = (X~ - X)^T X  [n, n] (X, X~ are [tokens, n]); P = ((D L) masked strictly upper) L^T with L = U^T, U the upper Cholesky factor of H^-1 that GPTQ already uses (L^T = U). Column update inside a group block, with W_j the current value of column j before it is quantised:
    W[:, j:] -= E_j U[j, j:] - W_j P[j, j:]      (E_j = (W_j - Q_j) / U[j, j])
and for the columns after the block:  W[:, end:] -= E U[blk, end:] - W_blk P[blk, end:]  (W_blk = the pre-quantisation columns of the block).
Test: python3 gptaq.py  (synthetic, CPU or GPU) compares GPTQ and GPTAQ output error against W X~."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for p in ('/tools', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'faster-llm-poc', 'poc', 'tools')):
    if os.path.isdir(p): sys.path.insert(0, p)
from ct_to_gptq import quantize_rows  # noqa: E402
G = 128


@torch.no_grad()
def hinv_upper(H, damp):
    H = H.clone(); dead = torch.diag(H) == 0; H[dead, dead] = 1
    H += damp * torch.diag(H).mean() * torch.eye(H.shape[0], device=H.device)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    return torch.linalg.cholesky(Hinv, upper=True)


@torch.no_grad()
def gptaq_rows(weight, U, P=None):
    """Quantise [rows, cols] weight; U upper Cholesky factor of H^-1; P (or None = plain GPTQ). Returns (q int8 [rows, cols], scales [rows, cols/G])."""
    weight = weight.clone(); rows, cols = weight.shape
    q_all = torch.empty(rows, cols, dtype=torch.int8, device=weight.device); scales = torch.empty(rows, cols // G, device=weight.device)
    for start in range(0, cols, G):
        end = start + G; block = weight[:, start:end].clone(); errors = torch.zeros_like(block); pre = torch.zeros_like(block)
        _, scale, _ = quantize_rows(block, G); scales[:, start // G] = scale[:, 0]
        ub = U[start:end, start:end]; pb = P[start:end, start:end] if P is not None else None
        for i in range(G):
            col = block[:, i]; pre[:, i] = col
            q = torch.clamp(torch.round(col / scale[:, 0]), -8, 7); q_all[:, start + i] = q.to(torch.int8)
            err = (col - q * scale[:, 0]) / ub[i, i]
            block[:, i:] -= err.unsqueeze(1) * ub[i, i:].unsqueeze(0)
            if pb is not None: block[:, i:] += col.unsqueeze(1) * pb[i, i:].unsqueeze(0)
            errors[:, i] = err
        weight[:, end:] -= errors @ U[start:end, end:]
        if P is not None: weight[:, end:] += pre @ P[start:end, end:]
    return q_all, scales


@torch.no_grad()
def p_matrix(D, U):
    """P = ((D L) strictly-upper-masked) L^T with L = U^T."""
    L = U.t(); M = torch.triu(torch.ones_like(D), diagonal=1)
    return ((D @ L) * M) @ L.t()


@torch.no_grad()
def quantise(weight, X, Xt=None, damp=0.01):
    """Convenience for tests: X, Xt [tokens, n]. Returns dequantised weight."""
    H = X.t() @ X; U = hinv_upper(H, damp)
    P = p_matrix((Xt - X).t() @ X, U) if Xt is not None else None
    q, sc = gptaq_rows(weight, U, P)
    return q.float() * sc.repeat_interleave(G, 1)


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(0)
    n, m, T = 512, 256, 8192
    for noise in (0.0, 0.05, 0.15):
        A = torch.randn(n, n, device=dev) / n ** 0.5 + torch.eye(n, device=dev)       # correlated activations
        X = torch.randn(T, n, device=dev) @ A
        Xt = X + noise * torch.randn(T, n, device=dev) @ A * X.std()                  # full-precision-stream input: differs from the quantised-stream input
        Wt = torch.randn(m, n, device=dev) * 0.05
        ref = Xt @ Wt.t()
        res = {}
        for name, xt in (('GPTQ', None), ('GPTAQ', Xt)):
            Wq = quantise(Wt, X, xt); res[name] = ((X @ Wq.t() - ref) ** 2).sum().item() / (ref ** 2).sum().item()
        print(f'input mismatch {noise:.2f}: relative output error vs W X~  GPTQ {res["GPTQ"]:.5f}  GPTAQ {res["GPTAQ"]:.5f}  ({100 * (1 - res["GPTAQ"] / res["GPTQ"]):+.1f}%)')
