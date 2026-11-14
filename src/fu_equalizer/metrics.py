"""Symbol metrics and the classical retained reference (Sec. V-A)."""
import numpy as np

from .config import NUM_CLASSES


def build_constellation(clients_data):
    const = np.zeros(NUM_CLASSES, dtype=np.complex128)
    seen = np.zeros(NUM_CLASSES, dtype=bool)
    for d in clients_data:
        for lbl in np.unique(d["y_train"]):
            if not seen[lbl]:
                const[lbl] = d["tx_train"][d["y_train"] == lbl].mean()
                seen[lbl] = True
        if seen.all():
            break
    if not seen.all():
        raise RuntimeError(f"constellation incomplete: {(~seen).sum()} labels unseen")
    return const


def nearest_symbol(z, const):
    out = np.empty(len(z), dtype=np.int64)
    for s in range(0, len(z), 200_000):
        blk = z[s:s + 200_000]
        out[s:s + 200_000] = np.argmin(np.abs(blk[:, None] - const[None, :]), axis=1)
    return out


def symbol_to_bits(labels):
    l = labels.astype(np.uint16)
    return ((l[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.int8)


def ser_ber(y_true, y_pred):
    ser = float(np.mean(y_true != y_pred))
    ber = float(np.mean(symbol_to_bits(y_true) != symbol_to_bits(y_pred)))
    return {"acc": 1.0 - ser, "SER": ser, "BER": ber}


def classical_reference(d, const):
    """Best of MMSE, unbiased MMSE and ZF with nearest-symbol decisions."""
    res = {name: ser_ber(d["y_eval"], nearest_symbol(d[key], const))
           for name, key in (("MMSE", "zmm_eval"), ("MMSE-unbiased", "zub_eval"), ("ZF", "zf_eval"))}
    best = max(res, key=lambda k: res[k]["acc"])
    return best, res


def utility_gate(retained_acc_pct, classical_ref_pct, num_classes=NUM_CLASSES):
    """A_ret >= max(10 x chance, 0.5 x classical reference), Eq. (7)."""
    thr = max(10 * 100.0 / num_classes, 0.5 * classical_ref_pct)
    return retained_acc_pct >= thr, thr
