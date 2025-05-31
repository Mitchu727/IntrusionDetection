import math
from datetime import datetime

import yaml
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split
)
from src.algorithms.pso import (
    generate_population,
    transform_particle_to_binary_array,
    particle_swarm_optimization_step
)
from src.utils import get_project_root
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

PSO_KNN_RESULTS_CSV = "results_PSO+kNN.csv"
KNN_RESULTS_CSV = "results_kNN.csv"

def log_results(df: pd.DataFrame, filename: str):
    header = not Path(filename).exists()
    df.to_csv(filename, mode="a", index=False, header=header)

def preprocess_data(dataset: pd.DataFrame):
    # Drop the unnamed index column
    dataset = dataset.drop(columns=[dataset.columns[0]])

    # Set the target labels
    y_orig = dataset["Attack_type"].values

    # Connect three normal activities into one label "normal"
    normal_activities = {"MQTT_Publish", "Thing_Speak", "Wipro_bulb"}
    y_multi = np.where(np.isin(y_orig, list(normal_activities)), "Normal", y_orig)

    # Set the target labels to binary (0 - normal activity, 1 - attack)
    y_binary = np.where(y_multi == "Normal", 0, 1)

    # Set the features (everything except the Attack_type column)
    X = dataset.drop(columns=["Attack_type"])

    # One-hot encode only the "proto" and "service" columns
    X = pd.get_dummies(X, columns=["proto", "service"])

    return X.values, y_binary, y_multi

def load_dataset(n_dims: int):
    path = get_project_root() / "datasets" / "RT_IOT2022.csv"
    dataset = pd.read_csv(path, header=0)

    X, y_binary, y_multi = preprocess_data(dataset)
    X = MinMaxScaler().fit_transform(X)
    X = PCA(n_components=n_dims).fit_transform(X)

    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
        X, y_binary, y_multi,
        test_size=0.3,
        random_state=42,
        stratify=y_multi
    )

    labels, counts = np.unique(y_multi, return_counts=True)
    for label, count in zip(labels, counts):
        print(f"{label}: {count}")

    return X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test

def idr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_attacks = tp + fn
    idr = tp / total_attacks
    return idr

def run_experiment_pso_knn(config_path: str):
    print(f"[INFO] Running PSO+kNN aglorithm experiment...")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    (X_train, X_test, 
     y_bin_train, y_bin_test, 
     y_multi_train, y_multi_test) = load_dataset(config["dimensions_number"])

    # 1. Initialize population
    population, velocities = generate_population(
        config['population_size'],
        config['dimensions_number']
    )
    local_bests = population
    local_bests_fitness = [-math.inf for _ in population]
    global_best = None
    global_best_fitness = -math.inf

    # 2. While stopping criterion is not met TODO add other stopping criterion than number of populations
    for iteration in range(config["iterations_number"]):
        for i, particle in enumerate(population):
            mask = transform_particle_to_binary_array(particle)
            X_sub= X_train[:, mask]

            print(f"[Iteration {iteration + 1}][Particle {i + 1}]: {particle}")
            print(f"[Iteration {iteration + 1}][Particle {i + 1}]: Shape after selection -> {X_sub.shape}")

            knn = KNeighborsClassifier(n_neighbors=config["k"])
            scores = cross_val_score(knn, X_sub, y_bin_train, cv=config["cross_validation_folds"])
            fitness = scores.mean()

            if fitness > local_bests_fitness[i]:
                local_bests_fitness[i] = fitness
                local_bests[i] = particle
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best = particle

            print(f"[Iteration {iteration + 1}][Particle {i + 1}]: Cross Validation Scores -> {scores}")
            print(f"[Iteration {iteration + 1}][Particle {i + 1}]: Average CV Score -> {fitness}")

        new_population = []
        new_velocities = []
        for i in range(config['population_size']):
            new_particle, new_velocity = particle_swarm_optimization_step(local_bests[i], global_best, population[i], velocities[i], config)
            new_population.append(new_particle)
            new_velocities.append(new_velocity)

        population = new_population
        velocities = new_velocities
        print(f"[Iteration {iteration + 1}]: New population Size -> {len(population)}")
        print(f"[Iteration {iteration + 1}]: New velocity Size -> {len(velocities)}")

    print(f"[INFO] Global best: {global_best}")
    print(f"[INFO] Global best fitness: {global_best_fitness}")

    # 3. Evaluate on the test set (binary)
    print(f"[INFO] Evaluation in progress...")
    mask = transform_particle_to_binary_array(global_best)
    X_train_sub = X_train[:, mask]
    X_test_sub = X_test[:, mask]

    knn = KNeighborsClassifier(n_neighbors=config["k"])
    knn.fit(X_train_sub, y_bin_train)
    y_bin_pred = knn.predict(X_test_sub)

    overall_accuracy = accuracy_score(y_bin_test, y_bin_pred)
    overall_idr = idr_score(y_bin_test, y_bin_pred)

    # 4. One-vs-all for each attack type
    attack_types = [c for c in np.unique(y_multi_train) if c != "Normal"]
    rows = []
    timestamp = datetime.now().isoformat()

    for attack in attack_types:
        y_train_ova = (y_multi_train == attack).astype(int)
        y_test_ova = (y_multi_test == attack).astype(int)

        knn_ova = KNeighborsClassifier(n_neighbors=config["k"])
        knn_ova.fit(X_train_sub, y_train_ova)
        y_pred_ova = knn_ova.predict(X_test_sub)

        accuracy_ova = accuracy_score(y_test_ova, y_pred_ova)
        idr_ova = idr_score(y_test_ova, y_pred_ova)

        rows.append({
            "timestamp": timestamp,
            "algorithm": "PSO+kNN",
            "k": config["k"],
            "population_size": config["population_size"],
            "iterations": config["iterations_number"],
            "n_dims": config["dimensions_number"],
            "global_CV_fitness": global_best_fitness,
            "attack_type": attack,
            "accuracy": accuracy_ova,
            "IDR": idr_ova
        })

    rows.append({
        "timestamp": timestamp,
        "algorithm": "PSO+kNN",
        "k": config["k"],
        "population_size": config["population_size"],
        "iterations": config["iterations_number"],
        "n_dims": config["dimensions_number"],
        "global_CV_fitness": global_best_fitness,
        "attack_type": "all_attacks",
        "accuracy": overall_accuracy,
        "IDR": overall_idr
    })

    logs_df = pd.DataFrame(rows)
    log_results(logs_df, PSO_KNN_RESULTS_CSV)
    print(f"[INFO] PSO+kNN test results logged to {PSO_KNN_RESULTS_CSV}")

def run_experiment_knn(config_path: str):
    print(f"[INFO] Running kNN aglorithm experiment...")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    (X_train, X_test, 
     y_bin_train, y_bin_test, 
     y_multi_train, y_multi_test) = load_dataset(config["dimensions_number"])
    
    knn = KNeighborsClassifier(n_neighbors=config["k"])
    knn.fit(X_train, y_bin_train)

    print(f"[INFO] Evaluation in progress...")

    # Binary classification evaluation
    y_bin_pred = knn.predict(X_test)
    overall_accuracy = accuracy_score(y_bin_test, y_bin_pred)
    overall_idr = idr_score(y_bin_test, y_bin_pred)

    # One-vs-all evaluation for each attack type
    attack_types = [c for c in np.unique(y_multi_train) if c != "Normal"]
    rows = []
    timestamp = datetime.now().isoformat()

    for attack in attack_types:
        y_train_ova = (y_multi_train == attack).astype(int)
        y_test_ova = (y_multi_test == attack).astype(int)

        knn_ova = KNeighborsClassifier(n_neighbors=config["k"])
        knn_ova.fit(X_train, y_train_ova)
        y_pred_ova = knn_ova.predict(X_test)

        accuracy_ova = accuracy_score(y_test_ova, y_pred_ova)
        idr_ova = idr_score(y_test_ova, y_pred_ova)

        rows.append({
            "timestamp": timestamp,
            "algorithm": "kNN",
            "k": config["k"],
            "n_dims": config["dimensions_number"],
            "attack_type": attack,
            "accuracy": accuracy_ova,
            "IDR": idr_ova
        })

    rows.append({
        "timestamp": timestamp,
        "algorithm": "kNN",
        "k": config["k"],
        "n_dims": config["dimensions_number"],
        "attack_type": "all_attacks",
        "accuracy": overall_accuracy,
        "IDR": overall_idr
    })

    logs_df = pd.DataFrame(rows)
    log_results(logs_df, KNN_RESULTS_CSV)
    print(f"[INFO] kNN test results logged to {KNN_RESULTS_CSV}")
