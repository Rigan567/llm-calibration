import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_mse(csv_path):
    df = pd.read_csv(csv_path)

    # Force numeric
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    bert = pd.to_numeric(df["bert"], errors="coerce")

    # Keep only valid rows where both are in [0, 1]
    mask = (
        conf.notna()
        & bert.notna()
        & (conf >= 0.0) & (conf <= 1.0)
        & (bert >= 0.0) & (bert <= 1.0)
    )

    conf = conf[mask]
    bert = bert[mask]

    if len(conf) == 0:
        raise ValueError(f"No valid rows after filtering in {csv_path}")
    
    dropped = len(df) - len(conf)
    if dropped > 0:
        print(f"{csv_path}: dropped {dropped} rows (confidence > 1 or invalid)")

    return ((bert - conf) ** 2).mean()


def extract_test_case(filename, model_name):
    suffix = f"_{model_name}_metrics.csv"
    if not filename.endswith(suffix):
        return None
    return filename[:-len(suffix)]


def main(model_dirs, model_names):
    # results[model_name][test_case] = mse
    results = {m: {} for m in model_names}
    all_test_cases = set()

    for model_dir, model_name in zip(model_dirs, model_names):
        for filename in os.listdir(model_dir):
            if not filename.endswith("_metrics.csv"):
                continue

            test_case = extract_test_case(filename, model_name)
            if test_case is None:
                continue  # filename belongs to a different model

            csv_path = os.path.join(model_dir, filename)
            mse = compute_mse(csv_path)

            results[model_name][test_case] = mse
            all_test_cases.add(test_case)

    # ---- VALIDATION ----
    for model_name in model_names:
        missing = all_test_cases - results[model_name].keys()
        if missing:
            raise RuntimeError(
                f"Model '{model_name}' is missing test cases: {sorted(missing)}"
            )

    test_cases = sorted(all_test_cases)

    # ---- PLOTTING ----
    num_models = len(model_names)
    x = np.arange(len(test_cases))
    bar_width = 0.8 / num_models

    plt.figure(figsize=(14, 6))

    for i, model_name in enumerate(model_names):
        mse_values = [results[model_name][tc] for tc in test_cases]

        plt.bar(
            x + i * bar_width,
            mse_values,
            width=bar_width,
            label=model_name,
        )

    plt.xlabel("Test Case")
    plt.ylabel("Mean Squared Error (BERT vs Confidence)")
    plt.title("MSE per Test Case and Model (BERT)")
    plt.xticks(
        x + bar_width * (num_models - 1) / 2,
        test_cases,
        rotation=45,
        ha="right",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs=3, required=True)
    parser.add_argument("--model_names", nargs=3, required=True)

    args = parser.parse_args()
    main(args.model_dirs, args.model_names)
