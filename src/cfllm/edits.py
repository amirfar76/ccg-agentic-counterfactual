import random
import re
from typing import Tuple

SCHEDULERS = ["rr", "pf", "mt"]

def make_base_prompt(num_ues: int, scheduler: str, traffic_mbps: float, duration_s: float) -> str:
    return (
        f"""Configure a single LTE base station to serve {num_ues} users using the {scheduler.upper()} scheduler. 
        Run the scenario for {duration_s} seconds with approximately {traffic_mbps} Mbps per-user offered load. 
        Report throughput per UE and average end-to-end latency."""
    ).strip()

def edit_prompt(prompt: str) -> str:
    # Allowable edits: tweak numbers slightly or toggle scheduler keyword synonyms
    # - change a number in-place by +/- 1 (for small ints) or +/-10% for floats
    # - toggle scheduler between rr/pf/mt if present
    p = prompt

    # maybe flip scheduler word first
    for sch in SCHEDULERS:
        if re.search(rf"\b{sch}\b", p, flags=re.IGNORECASE):
            choices = [s for s in SCHEDULERS if s.lower() != sch.lower()]
            new = random.choice(choices)
            p = re.sub(rf"\b{sch}\b", new, p, flags=re.IGNORECASE)
            break

    # then tweak the first number we see
    nums = list(re.finditer(r"(-?\d+\.?\d*)", p))
    if nums:
        m = random.choice(nums)
        val = float(m.group(1))
        if val.is_integer():
            val2 = int(val) + random.choice([-1, 1])
            if val2 < 1: val2 = 1
            rep = str(val2)
        else:
            val2 = val * (1.0 + random.choice([-0.1, 0.1]))
            rep = f"{val2:.2f}"
        p = p[:m.start()] + rep + p[m.end():]
    return p
