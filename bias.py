import pandas as pd
import numpy as np

def rand_symptoms(n):
    r = np.random.randint(0, 2, n)
    return r


def get_disease_probability(disease_column, person_symptoms):
    disease_sum_prob = 1

    for i in range(len(person_symptoms)):

        if (person_symptoms[i] == 1):
            disease_sum_prob *= disease_column[i] * person_symptoms[i]

    return disease_sum_prob


def get_all_disease_probabilities_for_person(symptoms, person_symptoms, dis_probabilities):
    all_diseases = symptoms.columns[1:]

    symptoms_prob = np.ones(symptoms.shape[1] - 1)

    for i in range(len(all_diseases)):
        symptoms_prob[i] = get_disease_probability(symptoms[all_diseases[i]], person_symptoms) * dis_probabilities[i]

    return symptoms_prob


if __name__ == "__main__":
    symptoms = pd.read_csv("symptom.csv", delimiter=';')
    diseases = pd.read_csv("disease.csv", delimiter=';')

    dis_prob = diseases['количество пациентов'] / list(diseases['количество пациентов'])[-1]

    dis_prob = list(dis_prob)[:-1]

    arr_rand_symplots = rand_symptoms(symptoms.shape[0])

    dis_probs = list(diseases['количество пациентов'] / list(diseases['количество пациентов'])[-1])
    del dis_probs[-1]

    sympt_number = symptoms.shape[0]

    perso = rand_symptoms(sympt_number)

    diseases_probs = get_all_disease_probabilities_for_person(symptoms, perso, dis_probs)

    print(symptoms.columns[np.argmax(diseases_probs) + 1])