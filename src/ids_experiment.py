import math
import time

import yaml
from sklearn.metrics import confusion_matrix
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

# UWAGI
# autorzy nie opisują jak dokonują analizy PCA

def preprocess_data(dataset: pd.DataFrame):
    # Drop the unnamed index column
    dataset = dataset.drop(columns=[dataset.columns[0]])

    # Set the target labels
    y = dataset["Attack_type"].values

    # Set the target labels to binary (0 - normal activity, 1 - attack)
    normal_activities = {"MQTT_Publish", "Thing_Speak", "Wipro_bulb"}
    y_binary = np.where(np.isin(y, list(normal_activities)), 0, 1)

    # Set the features (everything except the Attack_type column)
    X = dataset.drop(columns=["Attack_type"])

    # One-hot encode only the "proto" and "service" columns
    X = pd.get_dummies(X, columns=["proto", "service"])

    return X.values, y_binary

def load_dataset(n_dims: int):
    path = get_project_root() / "datasets" / "RT_IOT2022.csv"
    dataset = pd.read_csv(path, header=0)

    X, y = preprocess_data(dataset)
    X = MinMaxScaler().fit_transform(X)
    X = PCA(n_components=n_dims).fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# TODO
def idr_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(cm.shape)
    TN, FP, FN, TP = cm.ravel()
    total_attacks = TP + FN
    idr = (total_attacks - FN) / total_attacks
    return idr

def run_experiment_pso_knn(config_path: str):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    X_train, X_test, y_train, y_test = load_dataset(config["dimensions_number"])

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
            X_with_selected_attributes = X_train[:, transform_particle_to_binary_array(particle)]
            print(f"[Iteration {iteration + 1}][Particle {i + 1}]: {particle}")
            print(f"[Iteration {iteration + 1}][Particle {i + 1}]: Shape after selection -> {X_with_selected_attributes.shape}")
            knn = KNeighborsClassifier(n_neighbors=config["k"])
            scores = cross_val_score(knn, X_with_selected_attributes, y_train, cv=config["cross_validation_folds"])
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

    # 3. Evaluate on the test set
    X_test = X_test[:, transform_particle_to_binary_array(global_best)]
    knn = KNeighborsClassifier(n_neighbors=config["k"])
    knn.fit(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    print(f"[INFO] Test accuracy: {test_accuracy:.3f}")

def run_experiment_knn(config_path: str):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    X_train, X_test, y_train, y_test = load_dataset(config["dimensions_number"])
    knn = KNeighborsClassifier(n_neighbors=config["k"])
    knn.fit(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    print(f"[INFO] Test accuracy: {test_accuracy:.3f}")


# Save timestamp
    # print(dataset[1].unique())
    # print(dataset[2].unique())
    # print(dataset[3].unique())
    # print(dataset[6].unique())
    # print(X_transformed.shape)
    # X_transformed = X_transformed[:, [False, True, True, True, False, True, False, True, True, False]]
    # print(X_transformed.shape)
    ## Classification


    # start = time.time()
    # scores = cross_val_score(knn, X_transformed, Y, cv=10)
    # print("Cross Validation Scores: ", scores)
    # print("Average CV Score: ", scores.mean())
    # print("Number of CV Scores used in Average: ", len(scores))
    # end = time.time()
    # print("Time elapsed: ", end - start)
    # print(explained_variance)
    # load configuration file

