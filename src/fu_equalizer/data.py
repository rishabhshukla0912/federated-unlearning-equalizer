"""Read client files, build 9x6 input windows and classical estimates."""
import h5py
import numpy as np

from .config import (DATA_ROOT, CLIENT_SNR_DB, FORGET_CLIENT, NUM_SUBCARRIERS,
                     WINDOW_SIZE, CENTER, NUM_FEATURES, PARTS, IQ_GAIN_DB, IQ_PHASE_DEG)


def read_frames(client, start, n_frames, root=DATA_ROOT):
    """Return rx symbols y, channel h, tx symbols and labels, shape (n_frames, 1024)."""
    path = root / client / f"{client}_dataset.mat"
    a, n = start * NUM_SUBCARRIERS, n_frames * NUM_SUBCARRIERS
    with h5py.File(path, "r") as f:
        rx = f["rxSymbols_all"][:, a:a + n]
        h = f["channelH_all"][:, a:a + n]
        tx = f["txSymbols_all"][:, a:a + n]
        lab = f["allLabels"][:, a:a + n]
    cx = lambda v: (v[0] + 1j * v[1]).reshape(n_frames, NUM_SUBCARRIERS)
    return cx(rx), cx(h), cx(tx), lab.ravel().astype(np.int64).reshape(n_frames, NUM_SUBCARRIERS)


def mmse_equalize(y, h, snr_db):
    s2 = 10 ** (-snr_db / 10.0)
    return np.conj(h) * y / (np.abs(h) ** 2 + s2)


def mmse_unbiased(y, h, snr_db):
    s2 = 10 ** (-snr_db / 10.0)
    h2 = np.abs(h) ** 2
    return mmse_equalize(y, h, snr_db) * (h2 + s2) / np.maximum(h2, 1e-12)


def zf_equalize(y, h):
    return y / np.where(np.abs(h) < 1e-8, 1e-8, h)


def apply_iq_imbalance(y, gain_db=IQ_GAIN_DB, phase_deg=IQ_PHASE_DEG):
    """y' = mu*y + nu*conj(y), Eq. (9) of the paper."""
    g = 10 ** (gain_db / 20.0)
    phi = np.deg2rad(phase_deg)
    mu = (1 + g * np.exp(-1j * phi)) / 2
    nu = (1 - g * np.exp(1j * phi)) / 2
    return mu * y + nu * np.conj(y)


def windows(y, h, z):
    """Sliding 9-subcarrier windows of [Re y, Im y, Re h, Im h, Re z, Im z]."""
    feats = np.stack([y.real, y.imag, h.real, h.imag, z.real, z.imag], axis=-1)
    idx = np.arange(CENTER, NUM_SUBCARRIERS - CENTER)
    offs = np.arange(-CENTER, CENTER + 1)
    x = feats[:, idx[:, None] + offs[None, :], :]
    return x.reshape(-1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32), idx


def load_client(client, eval_part="val", iq=False, root=DATA_ROOT):
    """Train windows (frames 0-149) and windows of one evaluation partition.

    eval_part is "val", "pilot", "test" or a (start, n_frames) tuple.
    Features are standardized with the client's training mean and std.
    If iq is True the IQ-imbalance model is applied to the forget client.
    """
    snr = CLIENT_SNR_DB[client]
    tr0, trn = PARTS["train"]
    ev0, evn = PARTS[eval_part] if isinstance(eval_part, str) else eval_part
    out = {"client": client, "snr_db": snr}
    stats = None
    for tag, (s, n) in (("train", (tr0, trn)), ("eval", (ev0, evn))):
        y, h, tx, lab = read_frames(client, s, n, root)
        if iq and client == FORGET_CLIENT:
            y = apply_iq_imbalance(y)
        z = mmse_equalize(y, h, snr)
        x, idx = windows(y, h, z)
        if stats is None:
            mu = x.mean(axis=(0, 1)); sd = np.maximum(x.std(axis=(0, 1)), 1e-6)
            stats = (mu, sd)
        out[f"X_{tag}"] = ((x - stats[0]) / stats[1]).astype(np.float32)
        out[f"y_{tag}"] = lab[:, idx].ravel()
        out[f"tx_{tag}"] = tx[:, idx].ravel()
        out[f"zmm_{tag}"] = z[:, idx].ravel()
        out[f"zub_{tag}"] = mmse_unbiased(y, h, snr)[:, idx].ravel()
        out[f"zf_{tag}"] = zf_equalize(y, h)[:, idx].ravel()
    out["feat_mean"], out["feat_std"] = stats
    return out
