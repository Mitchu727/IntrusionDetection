import math
import random

def generate_random_particle(number_of_attributes):
    return [random.randint(0, 1) for _ in range(number_of_attributes)]

def generate_random_velocity(number_of_attributes):
    return [random.random() for _ in range(number_of_attributes)]

def transform_particle_to_binary_array(particle):
    return [True if dimension == 1 else False for dimension in particle]

def generate_population(population_size, number_of_attributes):
    return ([generate_random_particle(number_of_attributes) for _ in range(population_size)], 
            [generate_random_velocity(number_of_attributes) for _ in range(population_size)])

def particle_swarm_optimization_step(pbest_particle, gbest_particle, old_particle, old_velocity, config):
    w = config['w']
    c1 = config['c1']
    c2 = config['c2']
    particle = []
    velocity = []
    for i in range(len(pbest_particle)):
        rand1 = random.random()
        rand2 = random.random()
        velocity_new = w*old_velocity[i] + c1*rand1*(pbest_particle[i] - old_particle[i]) + c2*rand2*(gbest_particle[i] - old_particle[i])
        sigmoid_value = 1/(1+math.exp(-velocity_new))
        if random.random() < sigmoid_value:
            particle_new = 1
        else:
            particle_new = 0
        particle.append(particle_new)
        velocity.append(velocity_new)
    return particle, velocity