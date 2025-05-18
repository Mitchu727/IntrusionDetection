import math
import time

import yaml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from src.pso import generate_population, transform_particle_to_binary_array, particle_swarm_optimization_step
from src.utils import get_project_root
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


# UWAGI
# autorzy nie opisują jak dokonują analizy PCA

def preprocess_data(dataset):
    X = dataset.iloc[:, 0:41]
    Y = dataset.iloc[:, 41]
    X_onehot_encoded = pd.get_dummies(X, columns=[1,2,3])
    return X_onehot_encoded.values, Y.values

def load_kdd_dataset(number_of_dimensions):
    path = get_project_root() / "datasets" /"KDD_CUP_1999"/"kddcup.data_10_percent_corrected"
    dataset = pd.read_csv(path, header=None)
    X, Y = preprocess_data(dataset)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=number_of_dimensions)
    X = pca.fit_transform(X)
    ### FOR EXPERIMENTS -> remove before proper experiment
    X = X[:10000, :]
    Y = Y[:10000]
    print(X.shape)
    return X, Y

# def idr_score(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#     print(cm.shape)
#     TN, FP, FN, TP = cm.ravel()
#     total_attacks = TP + FN
#     idr = (total_attacks - FN) / total_attacks
#     return idr

if __name__ == "__main__":

    with open(get_project_root() / "src" / "resources" / "config.yml") as file:
        config = yaml.safe_load(file)

    X, Y = load_kdd_dataset(config["dimensions_number"])

    # 1. initialize population
    population, velocities = generate_population(config['population_size'], config['dimensions_number'])
    local_bests = population
    local_bests_fitness = [-math.inf for _ in population]
    global_best = None
    global_best_fitness = -math.inf
    # 2. while stopping criterion is not met TODO add other stopping criterion than number of populations
    for iteration in range(config["iterations_number"]):
        for i, particle in enumerate(population):
            print(particle)
            X_with_selected_attributes = X[:, transform_particle_to_binary_array(particle)]
            print(X_with_selected_attributes.shape)
            knn = KNeighborsClassifier(n_neighbors=config["k"])
            scores = cross_val_score(knn, X_with_selected_attributes, Y, cv=config["cross_validation_folds"])
            fitness = scores.mean()
            if fitness > local_bests_fitness[i]:
                local_bests_fitness[i] = fitness
                local_bests[i] = particle
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best = particle
            print("Cross Validation Scores: ", scores)
            print("Average CV Score: ", fitness)
            print("Number of CV Scores used in Average: ", len(scores))
        new_population = []
        new_velocities = []
        for i in range(config['population_size']):
            new_particle, new_velocity = particle_swarm_optimization_step(local_bests[i], global_best, population[i], velocities[i], config)
            new_population.append(new_particle)
            new_velocities.append(new_velocity)
        population = new_population
        velocities = new_velocities
        print("Population Size: ", len(population))
        print("Velocity Size: ", len(velocities))


    print(global_best)
    print(global_best_fitness)

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

