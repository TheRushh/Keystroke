from scipy.spatial.distance import cityblock, euclidean
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
np.set_printoptions(suppress = True)
import pandas
import GA

csv = pandas.read_csv("keystroke.csv")
columns = csv.columns.values[3:]
subjects = csv["subject"].unique()
mean_vector = {}
sheep = []
goat = []
lambs = []
wolves = []

#defense
for subject in subjects:
    userScores = []
    imposterScores = []
    genuine_accept = 0
    genuine_reject = 0
    imposter_accept = 0
    imposter_reject = 0
    data = csv.loc[csv.subject == subject , columns]
    # Training using every 10th pattern
    train_vector = data[0::10]
    imposter_data = csv.loc[csv.subject != subject , columns]
    mean_vector[subject] = train_vector.mean().values
    
    for i in range (data.shape[0]):
        score = 1/(1+euclidean (data.iloc[i].values , mean_vector[subject]))
        userScores.append(score)
        if score >= .70:
            genuine_accept+=1
        else:
            genuine_reject+=1
    # print(genuine_accept, genuine_accept+genuine_reject)

    for i in range (data.shape[0]):
        score = 1/(1+euclidean (imposter_data.iloc[i].values , mean_vector[subject]))
        imposterScores.append(score)
        if score >= .70:
            imposter_accept+=1
        else:
            imposter_reject+=1
    # print(imposter_accept, imposter_accept+imposter_reject)

    if genuine_accept/(genuine_accept+genuine_reject) >=.80:
        sheep.append(subject)
    if genuine_reject/(genuine_accept+genuine_reject) >= .80:
        goat.append(subject)
    if imposter_accept/(imposter_accept+imposter_reject) >= .60:
        lambs.append(subject)

    
    # print(subject, genuine_accept, genuine_reject, imposter_accept, imposter_reject)    

    # To see fpr vs tpr graph uncomment lines below

    # print(subject)
    # fpr, tpr, thresholds = roc_curve( [1]*len(userScores) + [0]*len(imposterScores), userScores+imposterScores)
    # roc_auc = auc(fpr,tpr)
    # plt.gcf().canvas.set_window_title(subject)
    # plt.plot(fpr, tpr, c='g',label= subject , linewidth = 4)
    # plt.xlabel('fpr')
    # plt.ylabel('tpr')
    # plt.show()
   
#attack
for subject in subjects:
    successAttack = 0
    failAttack = 0
    data = csv.loc[csv.subject == subject , columns]
    attackSubjects = csv.loc[csv.subject != subject, 'subject'].unique()
    for attackSubject in attackSubjects:
        score = 1/(1+euclidean(mean_vector[subject], mean_vector[attackSubject]))
        if score >= .70:
            successAttack+=1
        else:
            failAttack+=1
    # print (subject, successAttack, failAttack)

    if successAttack/(successAttack+failAttack) >= .65:
        wolves.append(subject)

print('Sheep:  ', sheep)
print('Goats:  ', goat)
print('Lambs:  ', lambs)
print('Wolves: ', wolves)

#Biometric adversary

print('Biometric adversary, populated with sheep and mating started using one random sheep')
equation_inputs = mean_vector[sheep[np.random.randint(0,high=len(sheep))]]
num_weights = 31
sol_per_pop = len(sheep)
num_parents_mating = 2

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
npop = []
for subject in sheep:
    npop.append(np.array(mean_vector[subject]))
new_population = np.array(npop)

num_generations = 50

best_result = 0
best_result_previous = 0

for generation in range(num_generations):
    print("Generation : ", generation)
    # Measing the fitness of each chromosome in the population.
    fitness = GA.cal_pop_fitness(equation_inputs, new_population)
    
    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # print(new_population)
    # The best result in the current iteration.
    best_result_previous = best_result
    best_result = np.max(np.sum(new_population*equation_inputs, axis=1))
    print("Best result : ", best_result)

    if best_result - best_result_previous <= 0.001 :
        break
    # elif best_result - best_result_previous <= 0.01:
    #     break
    else:
        continue


# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = GA.cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))

# print("Best solution : ", new_population[best_match_idx, :][0][0])
print("Best solution fitness : ", fitness[best_match_idx])