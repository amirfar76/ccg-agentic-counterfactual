import re
from typing import List
from rouge_score import rouge_scorer



def _extract_metrics_from_report(text: str):
    # Try to pull total throughput, per-UE throughputs, and average delay
    NUM = r"([0-9]+(?:\.[0-9]+)?)"
    t_total = None
    per_ues = []
    delay_ms = None

    m = re.search(r"total throughput.*?"+NUM+r"\s*mbps", text, flags=re.I)
    if m:
        t_total = float(m.group(1))

    # collect many Mbps numbers (often includes per-UE)
    for m in re.finditer(r"\b([0-9]+(?:\.[0-9]+)?)\s*mbps\b", text, flags=re.I):
        per_ues.append(float(m.group(1)))

    m = re.search(r"(average )?(delay|latency).*?"+NUM+r"\s*(ms|milliseconds?)", text, flags=re.I)
    if m:
        # group index depends on the regex above; last numeric capture is group(3)
        # safer to search again for the numeric capture only:
        m2 = re.search(NUM, m.group(0))
        if m2:
            delay_ms = float(m2.group(1))

    return t_total, per_ues, delay_ms

def _rel_err(a, b, eps=1e-9):
    if a is None or b is None:
        return None
    denom = max(abs(b), eps)
    return abs(a - b) / denom

def numeric_norm(y_pred: str, y_true: str):
    """
    Normalized numeric distance in [0,1].
    We parse totals, per-UE avg, and avg delay; compute clipped relative errors and average them.
    Returns 1.0 if nothing parseable.
    """
    t_y, per_y, d_y = _extract_metrics_from_report(y_pred)
    t_t, per_t, d_t = _extract_metrics_from_report(y_true)

    errs = []

    e = _rel_err(t_y, t_t);             errs.append(min(e,1.0) if e is not None else None)
    if per_y and per_t:
        e = _rel_err(sum(per_y)/len(per_y), sum(per_t)/len(per_t))
        errs.append(min(e,1.0))
    else:
        errs.append(None)
    e = _rel_err(d_y, d_t);             errs.append(min(e,1.0) if e is not None else None)

    valid = [x for x in errs if x is not None]
    return float(sum(valid)/len(valid)) if valid else 1.0




def distance_rouge_l(y: str, yprime: str) -> float:
    # we convert similarity to a "distance" by using 1 - F1
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(y, yprime)
    f1 = scores['rougeL'].fmeasure
    return 1.0 - float(f1)

def _extract_numbers(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]

def distance_numeric(y: str, yprime: str) -> float:
    a = _extract_numbers(y)
    b = _extract_numbers(yprime)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1e6  # large penalty if one has no numbers
    # compare on truncated min length
    n = min(len(a), len(b))
    if n == 0:
        return 1e6
    s = 0.0
    for i in range(n):
        s += abs(a[i] - b[i])
    # average absolute difference
    return s / n



# --- begin: normalized Levenshtein edit distance (pure Python, no deps) ---

def _norm_text_for_edit(s: str) -> str:
    s = str(s)
    # collapse whitespace and lowercase for robustness
    return " ".join(s.split()).lower()

def _levenshtein(a: str, b: str) -> int:
    # O(len(a)*len(b)) DP with two rows; fine for short reports
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * m
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = curr
    return prev[m]

def levenshtein_norm(y1: str, y2: str) -> float:
    """Normalized edit distance in [0,1] after light text normalization."""
    a = _norm_text_for_edit(y1)
    b = _norm_text_for_edit(y2)
    denom = max(len(a), len(b), 1)
    return _levenshtein(a, b) / denom

# --- end: normalized Levenshtein ---







METRICS = {
    "rouge": distance_rouge_l,
    "numeric": distance_numeric,
    "levenshtein_norm": levenshtein_norm,
    "edit": levenshtein_norm, 
}
# register
METRICS["numeric_norm"] = numeric_norm
