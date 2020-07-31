#!/usr/bin/env python
from cosc343world import Creature, World
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import itertools

MATRIX_ROWS = 11
MATRIX_COLUMNS = 36

# You can change this number to specify how many generations creatures are going to evolve over.
numGenerations = 5000

# You can change this number to specify how many turns there are in the simulation of the world for a given generation.
numTurns = 100

# You can change this number to change the world type.  You have two choices - world 1 or 2 (described in
# the assignment 2 pdf document).
worldType=2

# You can change this number to modify the world size.
gridSize=24

# You can set this mode to True to have the same initial conditions for each simulation in each generation - good
# for development, when you want to have some determinism in how the world runs from generation to generation.
repeatableMode=False

#Average fitness amongst a generation
averageFitness = []

# This is a class implementing you creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behaviour by producing actions in response to percepts.
class MyCreature(Creature):

    # Initialisation function.  This is where your creature
    # should be initialised with a chromosome in a random state.  You need to decide the format of your
    # chromosome and the model that it's going to parametrise.
    #
    # Input: numPercepts - the size of the percepts list that the creature will receive in each turn
    #        numActions - the size of the actions list that the creature must create on each turn
    def __init__(self, numPercepts, numActions):

        # Place your initialisation code here.  Ideally this should set up the creature's chromosome
        # and set it to some random state.
        self.chromosome = np.random.rand(MATRIX_ROWS,MATRIX_COLUMNS)
        self.fitness = 0

        # Do not remove this line at the end - it calls the constructors of the parent class.
        Creature.__init__(self)


    # This is the implementation of the agent function, which will be invoked on every turn of the simulation,
    # giving your creature a chance to perform an action.  You need to implement a model here that takes its parameters
    # from the chromosome and produces a set of actions from the provided percepts.
    #
    # Input: percepts - a list of percepts
    #        numAction - the size of the actions list that needs to be returned
    def AgentFunction(self, percepts, numActions):

        # At the moment the percepts are ignored and the actions is a list of random numbers.  You need to
        # replace this with some model that maps percepts to actions.  The model
        # should be parametrised by the chromosome.

        extendedPercepts = []

        # This extends the percepts into a binary format which the creatures are able to understand
        for percept in percepts:
            if percept == 0:
                extendedPercepts.extend([1, 0, 0, 0])
            elif percept == 1:
                extendedPercepts.extend([0, 1, 0, 0])
            elif percept == 2:
                extendedPercepts.extend([0, 0, 1, 0])
            elif percept == 3:
                extendedPercepts.extend([0, 0, 0, 1])

        # Multiply the percepts by the weights in the creatures chromosomes (to determine the creatures actions)
        actions = np.matmul(self.chromosome, extendedPercepts)

        return actions.tolist()

# Select the top 20% of the population, based on their fitness score.
# Input: the old population
# Return: a list of elite creatures
def selection(old_population):

    old_population.sort(key=lambda x: x.fitness, reverse=True)

    elite_creatures = []
    for i in range(7):
        elite_creatures.append(old_population[i])

    return elite_creatures


# Runs a two point crossover on every row of each parents chromosome
# Input - two parent creatures
# Returns - a child creature
def crossOver(creature1, creature2):
    child1 = MyCreature(9, 11)
    child1_chromosome = np.empty([MATRIX_ROWS,MATRIX_COLUMNS])

    for row in range(MATRIX_ROWS):
        chromosome1 = creature1.chromosome[row]
        chromosome2 = creature2.chromosome[row]

        index1 = random.randint(1, MATRIX_COLUMNS - 2)
        index2 = random.randint(1, MATRIX_COLUMNS - 2)

        if index2 >= index1:
            index2 += 1
        else:  # Swap the two cx points
            index1, index2 = index2, index1

        child1_chromosome[row] = np.concatenate([chromosome1[:index1],chromosome2[index1:index2],chromosome1[index2:]])

    child1.chromosome = child1_chromosome

    return(child1)

# Alternative crossover method, which took a row from each parent
# until a new child had a full chromosome
#
# Input - two parent creatures
#
# Returns - a child creature

# def crossOverRows(creature1, creature2):
#     child = MyCreature(MATRIX_ROWS, MATRIX_COLUMNS)
#
#     child_chromosome = np.empty([MATRIX_ROWS,MATRIX_COLUMNS])
#
#     i = 0
#
#     while i < MATRIX_ROWS:
#         if i != 10:
#             child_chromosome[i] = creature1.chromosome[i]
#             child_chromosome[i+1] = creature2.chromosome[i+1]
#         else:
#             child_chromosome[i] = creature1.chromosome[i]
#
#         i += 2
#
#     child.chromosome = child_chromosome
#
#     return child


# This function is called after every simulation, passing a list of the old population of creatures, whose fitness
# you need to evaluate and whose chromosomes you can use to create new creatures.
#
# Input: old_population - list of objects of MyCreature type that participated in the last simulation.  You
#                         can query the state of the creatures by using some built-in methods as well as any methods
#                         you decide to add to MyCreature class.  The length of the list is the size of
#                         the population.  You need to generate a new population of the same size.  Creatures from
#                         old population can be used in the new population - simulation will reset them to their
#                         starting state (not dead, new health, etc.).
#
# Returns: a list of MyCreature objects of the same length as the old_population.
def newPopulation(old_population):
    global numTurns

    nSurvivors = 0
    avgLifeTime = 0
    fitnessScore = 0
    fitnessScores = []

    # For each individual you can extract the following information left over
    # from the evaluation.  This will allow you to figure out how well an individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have a the end of simulation (0 if dead), the tick number
    # indicating the time of creature's death (if dead).  You should use this information to build
    # a fitness function that scores how the individual did in the simulation.
    for individual in old_population:

        # You can read the creature's energy at the end of the simulation - it will be 0 if creature is dead.
        energy = individual.getEnergy()

        # This method tells you if the creature died during the simulation
        dead = individual.isDead()

        # If the creature is dead, you can get its time of death (in units of turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath
        else:
            nSurvivors += 1
            avgLifeTime += numTurns

        if individual.isDead() == False:
            timeOfDeath = numTurns

        #Calculate a fitness value for the individual
        individual.fitness = energy + (timeOfDeath * 100)
        fitnessScore += individual.fitness

    fitnessScore = fitnessScore / len(old_population)

    averageFitness.append(fitnessScore)

    elite_creatures = selection(old_population)

    newSet = []

    # Take the creatures which did not die and add them to the new population
    for elite in old_population:
        if elite.isDead() == False:
            newSet.append(elite)

    # Mix 5 of the elite creatures with completely random genetics
    for i in range(5):
        randnum = random.randint(0, len(elite_creatures) - 2)
        newSet.append(crossOver(elite_creatures[randnum], MyCreature(9, 11)))

    remainingRequired = w.maxNumCreatures() - len(newSet)

    # Mix randomly selected elite creatures to create as many children as still needed
    i = 1
    while i <= int(remainingRequired):
        randnum = random.randint(0, len(elite_creatures)-2)
        randnum2 = random.randint(0, len(elite_creatures)-2)
        while randnum == randnum2:
            randnum2 = random.randint(0, len(elite_creatures)-2)
        newSet.append(crossOver(elite_creatures[randnum], elite_creatures[randnum2]))
        i += 1

    # Here are some statistics, which you may or may not find useful
    avgLifeTime = float(avgLifeTime)/float(len(population))
    print("Simulation stats:")
    print("  Survivors    : %d out of %d" % (nSurvivors, len(population)))
    print("  Average Fitness Score :", fitnessScore)
    print("  Avg life time: %.1f turns" % avgLifeTime)

    # The information gathered above should allow you to build a fitness function that evaluates fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting parents and
    # spawning then new creatures.


    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals is returned for the next generation.

    new_population = newSet

    return new_population


# Pygame window sometime doesn't spawn unless Matplotlib figure is not created, so best to keep the following two
# calls here.  You might also want to use matplotlib for plotting average fitness over generations.
plt.close('all')
fh=plt.figure()

# Create the world.  The worldType specifies the type of world to use (there are two types to chose from);
# gridSize specifies the size of the world, repeatable parameter allows you to run the simulation in exactly same way.
w = World(worldType=worldType, gridSize=gridSize, repeatable=repeatableMode)

#Get the number of creatures in the world
numCreatures = w.maxNumCreatures()

#Get the number of creature percepts
numCreaturePercepts = w.numCreaturePercepts()

#Get the number of creature actions
numCreatureActions = w.numCreatureActions()

# Create a list of initial creatures - instantiations of the MyCreature class that you implemented
population = list()
for i in range(numCreatures):
   c = MyCreature(numCreaturePercepts, numCreatureActions)
   population.append(c)

# Pass the first population to the world simulator
w.setNextGeneration(population)

# Runs the simulation to evaluate the first population
w.evaluate(numTurns)

# Show the visualisation of the initial creature behaviour (you can change the speed of the animation to 'slow',
# 'normal' or 'fast')
w.show_simulation(titleStr='Initial population', speed='fast')

for i in range(numGenerations):
    print("\nGeneration %d:" % (i+1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evaluate the next population
    w.evaluate(numTurns)

    # Show the visualisation of the final generation (you can change the speed of the animation to 'slow', 'normal' or
    # 'fast')
    if i==numGenerations-1:
        w.show_simulation(titleStr='Final population', speed='fast')


plt.plot(averageFitness, '-g')
plt.show()



