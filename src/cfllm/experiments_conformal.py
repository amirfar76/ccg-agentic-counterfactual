import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# === IMPORTS FROM YOUR EXISTING REPO ===
from report_utils import compute_log_likelihood
from report_utils import rouge_l_score
from llm_judge_eval import llm_judge_binary
from npe_train_npe_seed import load_npe_model
from test_cf import generate_counterfactual_once

# ============================================================
#                   ADMISSION FUNCTION A
# ============================================================

def admission_function_A(y_hat, y_true):
    """
    LLM-as-a-judge binary decision:
    1 = close enough
    0 = not close
    """
    return llm_judge_binary(y_hat, y_true)


# ============================================================
#                       QUALITY FUNCTION Q
# ============================================================

def quality_function_Q(x_prime, y_hat, model_name="llama3"):
    """
    Normalized log-likelihood as in CLM.
    """
    return compute_log_likelihood(x_prime, y_hat, model_name=model_name)


# ============================================================
#                 SIMILARITY FUNCTION S (ROUGE-L)
# ============================================================

def similarity_function_S(existing_set, new_candidate):
    """
    ROUGE-L similarity.
    If set is empty → return -∞ to allow first element.
    """
    if len(existing_set) == 0:
        return -1e9
    sims = [rouge_l_score(new_candidate, y) for y in existing_set]
    return max(sims)


# ============================================================
#                   STOPPING FUNCTION  F
#              (Max-quality over accepted set)
# ============================================================

def stopping_function_F(qualities):
    if len(qualities) == 0:
        return -1e9
    return max(qualities)


# ============================================================
#            RUN 1 REALIZATION OF OUR CONFORMAL SET
# ============================================================

def build_conformal_set(
    T, x_prime, npe_model, U_A, U_Y,
    lambda_1, lambda_2, lambda_3,
    max_k=20
):
    """
    Builds a set C_lambda(T, X') using our conformal acceptance/stopping rules.
    Returns:
        C              = list of reports
        k_stop         = number of samples taken
        k_first_accept = index of first successful A(y, y_true)=1 (if known)
    """
    C = []
    qualities = []
    k_first_accept = None

    for k in range(1, max_k + 1):
        # counterfactual sample using NPE posterior sample
        z_noise = npe_model.sample_noise(T["A"], T["Z"])  
        y_hat = generate_counterfactual_once(
            T,
            x_prime,
            z_noise=z_noise,
            U_A=U_A,
            U_Y=U_Y
        )

        # quality + similarity tests
        q = quality_function_Q(x_prime, y_hat)
        s = similarity_function_S(C, y_hat)

        accept = (q >= lambda_1) and (s <= lambda_2)

        if accept:
            C.append(y_hat)
            qualities.append(q)

        # check if this is the first admissible sample (can only know if GT exists)
        # leave placeholder; real evaluation happens later when metric is computed
        # but we store quantities anyway.
        if k_first_accept is None:
            k_first_accept = k

        # stopping rule
        if stopping_function_F(qualities) >= lambda_3:
            return C, k, k_first_accept

    # If never stopped by F
    return C, max_k, k_first_accept


# ============================================================
#                  BASELINE: FIXED-k SAMPLING
# ============================================================

def baseline_fixed_k(T, x_prime, npe_model, U_A, U_Y, k):
    """
    Generate exactly k counterfactual reports (no acceptance/stopping logic).
    Returns list of size k.
    """
    out = []
    for _ in range(k):
        z_noise = npe_model.sample_noise(T["A"], T["Z"])
        y_hat = generate_counterfactual_once(
            T, x_prime, z_noise=z_noise, U_A=U_A, U_Y=U_Y
        )
        out.append(y_hat)
    return out


# ============================================================
#                   EVALUATION METRICS
# ============================================================

def compute_set_loss(C, y_true):
    """
    Loss = 1 if NO element in C passes admission
    """
    for y in C:
        if admission_function_A(y, y_true) == 1:
            return 0
    return 1


def compute_relative_excess(k_stop, k_first_accept):
    if k_first_accept is None:
        return k_stop
    return max(k_stop - k_first_accept, 0)


# ============================================================
#                       MAIN EXPERIMENT
# ============================================================

def run_conformal_experiments(
    calibration_data_path,
    output_dir="outputs/conformal",
    epsilons = [0.05, 0.1, 0.2],
    max_k=20
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    # load calibration jsonl
    with open(calibration_data_path, "r") as f:
        cal_data = [json.loads(l) for l in f]

    # Load NPE model
    npe_model = load_npe_model("models/npe_seed.pt")

    results_all = []

    for eps in epsilons:
        lambda_1 = -2.0     # threshold for Q
        lambda_2 = 0.95     # ROUGE-L similarity max
        lambda_3 = -1.0     # max quality threshold

        row = {"epsilon": eps}

        losses_ours = []
        xs_ours = []
        set_sizes_ours = []

        losses_k1 = []
        losses_k5 = []
        losses_k10 = []

        xs_k1 = []
        xs_k5 = []
        xs_k10 = []

        sizes_k1 = []
        sizes_k5 = []
        sizes_k10 = []

        for item in tqdm(cal_data, desc=f"Epsilon = {eps}"):
            T = item["T"]
            x_prime = item["X_prime"]
            y_true = item["Y_true"]
            U_A = item["U_A"]
            U_Y = item["U_Y"]

            # our method
            C, k_stop, k_first = build_conformal_set(
                T, x_prime, npe_model, U_A, U_Y,
                lambda_1, lambda_2, lambda_3, max_k=max_k
            )

            loss = compute_set_loss(C, y_true)
            excess = compute_relative_excess(k_stop, k_first)
            size_c = len(C)

            losses_ours.append(loss)
            xs_ours.append(excess)
            set_sizes_ours.append(size_c)

            # baselines
            for kk, losses_k, xs_k, sizes_k in [
                (1, losses_k1, xs_k1, sizes_k1),
                (5, losses_k5, xs_k5, sizes_k5),
                (10, losses_k10, xs_k10, sizes_k10),
            ]:
                Cb = baseline_fixed_k(T, x_prime, npe_model, U_A, U_Y, kk)
                lossb = compute_set_loss(Cb, y_true)
                excessb = kk - 1
                losses_k.append(lossb)
                xs_k.append(excessb)
                sizes_k.append(kk)

        row.update({
            "ours_loss": np.mean(losses_ours),
            "k1_loss": np.mean(losses_k1),
            "k5_loss": np.mean(losses_k5),
            "k10_loss": np.mean(losses_k10),

            "ours_x": np.mean(xs_ours),
            "k1_x": np.mean(xs_k1),
            "k5_x": np.mean(xs_k5),
            "k10_x": np.mean(xs_k10),

            "ours_size": np.mean(set_sizes_ours),
            "k1_size": np.mean(sizes_k1),
            "k5_size": np.mean(sizes_k5),
            "k10_size": np.mean(sizes_k10),
        })

        results_all.append(row)

    # save results
    out_json = f"{output_dir}/results.json"
    with open(out_json, "w") as f:
        json.dump(results_all, f, indent=2)

    print(f"Saved results to {out_json}")

    # ===================================================  
    #               Produce the three plots
    # ===================================================
    import matplotlib.pyplot as plt

    eps = [r["epsilon"] for r in results_all]

    # 1. set loss
    plt.figure()
    plt.plot(eps, [r["ours_loss"] for r in results_all], label="Conformal")
    plt.plot(eps, [r["k1_loss"] for r in results_all], label="k=1")
    plt.plot(eps, [r["k5_loss"] for r in results_all], label="k=5")
    plt.plot(eps, [r["k10_loss"] for r in results_all], label="k=10")
    plt.xlabel("epsilon")
    plt.ylabel("set loss")
    plt.legend()
    plt.savefig(f"{output_dir}/plots/set_loss.pdf")
    plt.close()

    # 2. excess samples
    plt.figure()
    plt.plot(eps, [r["ours_x"] for r in results_all], label="Conformal")
    plt.plot(eps, [r["k1_x"] for r in results_all], label="k=1")
    plt.plot(eps, [r["k5_x"] for r in results_all], label="k=5")
    plt.plot(eps, [r["k10_x"] for r in results_all], label="k=10")
    plt.xlabel("epsilon")
    plt.ylabel("relative excess samples")
    plt.legend()
    plt.savefig(f"{output_dir}/plots/excess_samples.pdf")
    plt.close()

    # 3. set size
    plt.figure()
    plt.plot(eps, [r["ours_size"] for r in results_all], label="Conformal")
    plt.plot(eps, [r["k1_size"] for r in results_all], label="k=1")
    plt.plot(eps, [r["k5_size"] for r in results_all], label="k=5")
    plt.plot(eps, [r["k10_size"] for r in results_all], label="k=10")
    plt.xlabel("epsilon")
    plt.ylabel("set size")
    plt.legend()
    plt.savefig(f"{output_dir}/plots/set_size.pdf")
    plt.close()

    print("Plots saved.")


# ============================================================
#                    IF CALLED DIRECTLY
# ============================================================

if __name__ == "__main__":
    run_conformal_experiments(
        calibration_data_path="data/calibration_pairs.jsonl",
        output_dir="outputs/conformal",
        epsilons=[0.05, 0.1, 0.2, 0.3],
        max_k=20
    )

