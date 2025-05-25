import argparse

from src.ids_experiment import run_experiment_knn, run_experiment_pso_knn

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IDS experiment using kNN or PSO+kNN algorithm."
    )

    parser.add_argument(
        "--algorithm",
        choices = ["kNN", "PSO+kNN"],
        default = "kNN",
        help="Which algorithm to run: kNN or PSO+kNN."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"[INFO] Running IDS experiment with {args.algorithm} algorithm.")
    config_path = "./config.yml"
    if args.algorithm == "kNN":
        run_experiment_knn(config_path)
    else:
        run_experiment_pso_knn(config_path)