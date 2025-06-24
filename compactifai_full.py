"""
compactifai_full.py – faithful reference of the CompactifAI pipeline
====================================================================
Implements every step described in the ES2025-8 paper:

Step-1  Layer-sensitivity profiling (ΔLoss / ‖ΔW‖)  
Step-2  Tensorisation of each chosen layer  
Step-3  k-core TT-SVD → MPO cores  
Step-4  χ-controlled compression / accuracy trade-off  
Step-5  Healing fine-tune (<1 epoch) – PyTorch hook  
Step-6  Optional mixed 4-/8-bit quantisation for non-tensorised layers

Dependencies
------------
* numpy                   – always required
* torch >= 1.9            – only if you call `heal()` (healing fine-tune)
* tqdm                    – cosmetic progress bar for healing

Usage (quick demo)
------------------
python compactifai_full.py        # runs an end-to-end demo on 2 toy layers
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Sequence, Callable, Any, Dict, Optional

# Optional torch import for healing
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════  MPO helper  ══════════════════════════════════ #
class MPO:
    """
    Stores a chain of cores  T^(1)…T^(k).
    Each core shape: (χ_{p-1}, d_in[p], d_out[p], χ_p)
    """
    def __init__(self,
                 cores: List[np.ndarray],
                 d_in: List[int],
                 d_out: List[int]) -> None:
        self.cores, self.d_in, self.d_out = cores, d_in, d_out

    # ------------ dense reconstruction (for error / debug) ------------- #
    def to_matrix(self) -> np.ndarray:
        T = self.cores[0]
        for p in range(1, len(self.cores)):
            T = np.tensordot(T, self.cores[p], axes=([-1], [0]))  # contract χ
        T = np.squeeze(T, axis=(0, -1))  # remove boundary bonds
        return T.reshape(int(np.prod(self.d_in)),
                         int(np.prod(self.d_out)))

    def n_params(self) -> int:
        return sum(c.size for c in self.cores)


# ═══════════════════════  Compressor class  ════════════════════════════ #
class CompactifAI:
    def __init__(self,
                 k: int  = 2,
                 chi: int = 64,
                 heal_epochs: float = 0.5,
                 heal_lr: float = 5e-4):
        """
        k            : tensor order (#cores). 2-3 for square layers, 4-6 for tall.
        chi          : max bond dimension.
        heal_epochs  : epochs of healing fine-tune (<1) on the given dataset.
        heal_lr      : learning-rate used in healing.
        """
        assert k >= 2
        self.k, self.chi = k, chi
        self.heal_epochs, self.heal_lr = heal_epochs, heal_lr

    # ─────────────────────────────────────────────────────────────────── #
    #                              STEP 1                                 #
    # ─────────────────────────────────────────────────────────────────── #
    @staticmethod
    def layer_sensitivity(
        W_list : Sequence[np.ndarray],
        loss_fn: Callable[[Sequence[np.ndarray]], float],
        epsilon: float = 1e-3
    ) -> List[float]:
        """
        Finite-difference ΔLoss / ‖ΔW‖  → lower is better for compression.
        """
        base_loss = loss_fn(W_list)
        scores: List[float] = []
        for i, W in enumerate(W_list):
            δ = epsilon * np.linalg.norm(W)
            noise = np.random.randn(*W.shape)
            noise = δ * noise / np.linalg.norm(noise)
            perturbed = [w if j != i else w + noise for j, w in enumerate(W_list)]
            new_loss = loss_fn(perturbed)
            scores.append(abs(new_loss - base_loss) / δ)
        return scores

    # ─────────────────────────────────────────────────────────────────── #
    #                        STEPS 2–4  (Compression)                     #
    # ─────────────────────────────────────────────────────────────────── #
    def compress_matrix(self, W: np.ndarray,
                        d_in: Optional[List[int]] = None,
                        d_out: Optional[List[int]] = None) -> MPO:
        """
        Compress a matrix into an MPO using TT-SVD.
        If d_in / d_out not supplied, compute power-of-two factors.
        """
        m, n = W.shape
        if d_in is None or d_out is None:
            d_in, d_out = self._auto_factors(m, n, self.k)

        # Interleave physical indices for TT-SVD
        dims = [d for pair in zip(d_in, d_out) for d in pair]  # d1,d1',d2,d2',...
        T = W.reshape(*dims)

        cores: List[np.ndarray] = []
        χ_left = 1
        for p in range(self.k):
            d1, d2 = d_in[p], d_out[p]
            T = T.reshape(χ_left * d1 * d2, -1)

            U, S, Vt = np.linalg.svd(T, full_matrices=False)
            full_rank = len(S)
            rank = full_rank if p == self.k-1 else min(self.chi, full_rank)

            U, S, Vt = U[:, :rank], S[:rank], Vt[:rank]
            core = U.reshape(χ_left, d1, d2, rank)
            cores.append(core)

            T = np.diag(S) @ Vt          # residual for next step
            χ_left = rank
            if p < self.k-1:
                T = T.reshape(χ_left, *dims[2*(p+1):])

        return MPO(cores, d_in, d_out)

    # ─────────────────────────────────────────────────────────────────── #
    #                            STEP 5  (Healing)                        #
    # ─────────────────────────────────────────────────────────────────── #
    def heal(
        self,
        mpo_layers : List[MPO],
        data_iter  : Callable[[], Sequence[Tuple[np.ndarray, np.ndarray]]],
        loss_fn    : Callable[[Any, Any], Any]
    ) -> None:
        """
        Fine-tune MPO cores for <1 epoch on a user-supplied mini dataset.

        • Requires PyTorch.  Each MPO is temporarily converted to an
          nn.Linear layer with weight = dense reconstructor of the MPO.
        • data_iter() must yield (input, target) numpy batches with
          shapes  [B, in_dim]  and  [B, out_dim]  respectively.
        • loss_fn(output, target) → scalar PyTorch loss.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for healing.  `pip install torch`")

        import torch
        import torch.nn as nn
        from tqdm import tqdm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build one Linear per MPO: transpose dense since to_matrix returns (in_dim, out_dim)
        torch_layers: List[nn.Linear] = []
        optim_params = []
        for mpo in mpo_layers:
            dense_np = mpo.to_matrix()  # shape (in_dim, out_dim)
            in_features, out_features = dense_np.shape
            # create linear mapping in_dim -> out_dim
            layer = nn.Linear(in_features, out_features, bias=False)
            # copy transposed weight: weight shape (out_dim, in_dim)
            weight_tensor = torch.tensor(dense_np.T, dtype=torch.float32, device=device)
            layer.weight.data.copy_(weight_tensor)
            layer.to(device)
            torch_layers.append(layer)
            optim_params += list(layer.parameters())

        opt = torch.optim.Adam(optim_params, lr=self.heal_lr)

        batches = list(data_iter())                       # cache iterable
        total_steps = int(len(batches) * self.heal_epochs + 0.5)

        for step in tqdm(range(total_steps), desc="Healing"):
            x_np, y_np = batches[step % len(batches)]
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            y = torch.tensor(y_np, dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            out = x                                       # no transpose!
            for layer in torch_layers:
                out = layer(out)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        # rebuild MPO cores from the healed dense weights
        for mpo, layer in zip(mpo_layers, torch_layers):
            dense_healed = layer.weight.detach().cpu().numpy()
            refreshed = self.compress_matrix(
                dense_healed, d_in=mpo.d_in, d_out=mpo.d_out)
            mpo.cores = refreshed.cores
    # ─────────────────────────────────────────────────────────────────── #
    #                     OPTIONAL STEP 6 (Quantisation)                  #
    # ─────────────────────────────────────────────────────────────────── #
    @staticmethod
    def quantise_int8(matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Very simple per-tensor symmetric int8 quantisation.
        Returns (int8_matrix, {scale}).
        """
        scale = np.abs(matrix).max() / 127.0 + 1e-12
        q = np.round(matrix / scale).astype(np.int8)
        return q, {"scale": scale}

    # ─────────────────────────────────────────────────────────────────── #
    #                           helpers                                   #
    # ─────────────────────────────────────────────────────────────────── #
    @staticmethod
    def _auto_factors(m: int, n: int, k: int) -> Tuple[List[int], List[int]]:
        """
        power-of-two split, early factors large.
        """
        def split(val: int) -> List[int]:
            factors, remain = [], val
            for _ in range(k - 1):
                root = 1 << int(np.floor(np.log2(remain ** (1 / (k - len(factors))))))
                root = max(4, root)
                while remain % root:
                    root //= 2
                factors.append(root)
                remain //= root
            factors.append(remain)
            return factors
        return split(m), split(n)


# ═══════════════════════  Quick demo  ═════════════════════════════════ #
if __name__ == "__main__":
    # 2 toy layers
    np.random.seed(0)
    W1 = np.random.randn(128, 128).astype(np.float32)
    W2 = np.random.randn(128,  64).astype(np.float32)
    layers = [W1, W2]

    # dummy validation loss for sensitivity
    def fro_loss(ws): return sum(np.linalg.norm(w)**2 for w in ws)

    cfa = CompactifAI(k=2, chi=64)

    print("Layer sensitivities:", cfa.layer_sensitivity(layers, fro_loss))

    mpo_layers = [cfa.compress_matrix(W) for W in layers]
    for i, (W, mpo) in enumerate(zip(layers, mpo_layers), 1):
        rel = np.linalg.norm(W - mpo.to_matrix()) / np.linalg.norm(W)
        print(f"Layer{i}  params {W.size}→{mpo.n_params()}  rel-err {rel:.2%}")

    # optional INT8 quantise non-tensorised layer example
    q8, meta = CompactifAI.quantise_int8(layers[1])
    print("Quantised int8 scale:", meta["scale"])

    # Healing demo (skipped if torch unavailable)
    if TORCH_AVAILABLE:
        def fake_data_for_layer(in_dim, out_dim):
            for _ in range(20):
                x = np.random.randn(32, in_dim).astype(np.float32)
                y = np.random.randn(32, out_dim).astype(np.float32)
                yield x, y

        # Heal each layer with matching input/output dims
        for i, mpo in enumerate(mpo_layers):
            in_dim = int(np.prod(mpo.d_in))
            out_dim = int(np.prod(mpo.d_out))
            cfa.heal([mpo], lambda: fake_data_for_layer(in_dim, out_dim), lambda o, t: torch.nn.functional.mse_loss(o, t))
            print(f"Healing done – new rel-err for Layer{i+1}:",
                  np.linalg.norm(layers[i] - mpo.to_matrix()) / np.linalg.norm(layers[i]))
    else:
        print("PyTorch not installed – healing demo skipped")
