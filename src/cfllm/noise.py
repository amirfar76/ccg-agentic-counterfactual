# cfllm/noise.py
import copy, hashlib, random

def _stable_int_seed(*parts) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")

def perturb_metrics_for_estimate(metrics: dict, seed_key, scale: float = 3e-3) -> dict:
    """
    Add tiny, seed-locked jitter to environment metrics so your 'estimated CF'
    is close-but-not-identical to the true CF. Deterministic for a given seed_key.

    Args:
      metrics: dict with keys like total_throughput_mbps, avg_delay_ms, per_ue_throughput_mbps
      seed_key: anything hashable (use the env rng_run you used for the CF)
      scale: relative jitter magnitude (default 0.3%); tune to adjust edit-distance

    Returns:
      Jittered copy of metrics.
    """
    rnd = random.Random(_stable_int_seed("est-env-noise", seed_key))
    out = copy.deepcopy(metrics)

    def _bump(x, rel=scale):
        try:
            x = float(x)
        except Exception:
            return x
        mag = max(1.0, abs(x))
        eps = (rnd.random() * 2.0 - 1.0) * rel * mag
        val = x + eps
        return 0.0 if val < 0 else val

    if "total_throughput_mbps" in out:
        out["total_throughput_mbps"] = _bump(out["total_throughput_mbps"], scale)

    if "avg_delay_ms" in out:
        out["avg_delay_ms"] = _bump(out["avg_delay_ms"], scale * 0.5)

    if isinstance(out.get("per_ue_throughput_mbps"), list):
        out["per_ue_throughput_mbps"] = [
            _bump(v, scale) for v in out["per_ue_throughput_mbps"]
        ]

    return out

