"""
compactifai_full.py  –  DEBUG + safe rank limit
==============================================
A NumPy reference for CompactifAI with:
• power-of-two factorisation (no tiny 2×2×… choke)
• adaptive rank cap so MPO params never exceed dense params
• full debug prints for each TT-SVD step
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Sequence, Callable, Any


# ═══════════════  MPO helper  ═══════════════ #
class MPO:
    def __init__(self, cores: List[np.ndarray],
                 d_in: List[int], d_out: List[int]) -> None:
        self.cores, self.d_in, self.d_out = cores, d_in, d_out
        self.k = len(cores)

    def to_matrix(self) -> np.ndarray:
        T = self.cores[0]
        for p in range(1, self.k):
            T = np.tensordot(T, self.cores[p], axes=([-1], [0]))
        T = np.squeeze(T, axis=(0, -1))
        return T.reshape(int(np.prod(self.d_in)),
                         int(np.prod(self.d_out)))

    def n_params(self) -> int:
        return sum(core.size for core in self.cores)


# ═══════════════  Compressor  ═══════════════ #
class CompactifAI:
    def __init__(self, k: int = 2, chi: int = 32) -> None:
        if k < 2:
            raise ValueError("k must be ≥ 2")
        self.k, self.chi = k, chi

    # ─── step-2-4: TT-SVD → MPO ─── #
    def compress_matrix(self, W: np.ndarray) -> MPO:
        m, n = W.shape
        d_in, d_out = self._factor_dims(m, n, self.k)
        phys = [d for pair in zip(d_in, d_out) for d in pair]
        print(f"[DEBUG] interleave dims = {phys}")
        T = W.reshape(*phys)

        cores: List[np.ndarray] = []
        chi_left = 1
        param_budget = W.size      # never exceed this

        for p in range(self.k):
            d1, d2 = d_in[p], d_out[p]
            T = T.reshape(chi_left * d1 * d2, -1)
            print(f"[DEBUG] step {p+1}/{self.k}  mat {T.shape}")

            U, S, Vt = np.linalg.svd(T, full_matrices=False)
            full_rank = len(S)

            # ----- adaptive rank cap ----- #
            # tentative rank = min(chi, full_rank)
            rank = min(self.chi, full_rank)
            # ensure adding this core won’t blow param budget
            while True:
                est_params = (chi_left * d1 * d2 * rank)        # this core
                # rough lower-bound for remaining cores (worst-case rank==rank)
                remain_cores = (self.k - p - 1)
                est_params += remain_cores * (rank * 4 * 4)     # minimal dims 4×4
                if est_params < param_budget or rank == 1:
                    break
                rank //= 2
            print(f"        full rank {full_rank}  truncated rank {rank}")

            U, S, Vt = U[:, :rank], S[:rank], Vt[:rank]
            core = U.reshape(chi_left, d1, d2, rank)
            print(f"        core shape {core.shape}  params so far "
                  f"{sum(c.size for c in cores)+core.size:,d}")
            cores.append(core)

            T = np.diag(S) @ Vt
            chi_left = rank

            if p < self.k - 1:
                T = T.reshape(chi_left, *phys[2*(p+1):])

        print(f"[DEBUG] dense params {W.size} → MPO params {sum(c.size for c in cores)}")
        return MPO(cores, d_in, d_out)

    # ─── smarter factor split ─── #
    @staticmethod
    def _factor_dims(m: int, n: int, k: int) -> Tuple[List[int], List[int]]:
        def split(val: int) -> List[int]:
            factors, remain = [], val
            for _ in range(k - 1):
                root = 1 << int(np.floor(np.log2(remain ** (1 / (k - len(factors))))))
                root = max(4, root)            # force ≥4
                while remain % root:
                    root >>= 1
                factors.append(root)
                remain //= root
            factors.append(remain)
            return factors
        d_in, d_out = split(m), split(n)
        print(f"[DEBUG] factor_dims  m={m}->{d_in}  n={n}->{d_out}")
        return d_in, d_out

    # ─── optional sensitivity / healing ─── #
    @staticmethod
    def layer_sensitivity(
        W_list: Sequence[np.ndarray],
        loss_fn: Callable[[Sequence[np.ndarray]], float],
        eps: float = 1e-3
    ) -> List[float]:
        base = loss_fn(W_list)
        out  = []
        for i, W in enumerate(W_list):
            delta = eps * np.linalg.norm(W)
            noise = delta * np.random.randn(*W.shape) / np.linalg.norm(W)
            pert  = [w if j != i else w + noise for j, w in enumerate(W_list)]
            out.append(abs(loss_fn(pert) - base) / delta)
        return out

    @staticmethod
    def healing(mpo_list: Sequence[MPO],
                fine_tune_fn: Callable[[Sequence[MPO]], Any]) -> Any:
        return fine_tune_fn(mpo_list)


# ═══════════════  Demo  ═══════════════ #
if __name__ == "__main__":
    np.random.seed(0)
    W1 = np.random.randn(128, 128).astype(np.float32)
    W2 = np.random.randn(128,  64).astype(np.float32)
    layers = [W1, W2]

    def toy_loss(ws): return sum(np.linalg.norm(w)**2 for w in ws)

    comp = CompactifAI(k=2, chi=32)      # safe baseline
    print("=== CompactifAI diagnostic test ===")
    sens = comp.layer_sensitivity(layers, toy_loss)
    for i, s in sorted(enumerate(sens, 1), key=lambda x: x[1]):
        print(f"  sensitivity layer {i}: {s:.2e}")

    def analyse(idx, W):
        mpo = comp.compress_matrix(W)
        rel = np.linalg.norm(W - mpo.to_matrix()) / np.linalg.norm(W)
        print(f"\n>>> Layer {idx}")
        print(f"    params {W.size:,d} → {mpo.n_params():,d}  "
              f"compression {(W.size/mpo.n_params()):.2f}×")
        print(f"    relative error {rel:.2%}")
        assert rel < 0.20, "Relative error exceeds 20 %!"

    for i, W in enumerate(layers, 1):
        analyse(i, W)

    comp.healing([], lambda _: print("\nHealing skipped in demo"))
    print("\n✅  All diagnostics passed")
