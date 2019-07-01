import numpy as np
from gensim.models import KeyedVectors

w_vectors = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',
    binary=True,
    limit=100000
)
w_vectors.init_sims(replace=True)
w_vectors.save('w_vectors')

w_vectors = KeyedVectors.load('w_vectors', mmap='r')
print("Model saved and loaded.")

relation_names = ['capital-world', 'currency', 'city-in-state', 'family', 'gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative', 'gram6-nationality-adjective']
relations = {}

for r_name in relation_names:
    relations[r_name] = []

part1_file = open("word-test.v1.txt", "r")
lines = part1_file.read().split(":")
lines.pop(0)
for line in lines:
    line = line.split("\n")
    for k, v in relations.items():
        if line[0].split(" ")[1] == k:
            for i in range(len(line) - 2):
                v.append(line[i + 1])


def find_vector(words):
    new_vector = w_vectors[words[1]] - w_vectors[words[0]] + w_vectors[words[2]]
    return new_vector


def cos_sim(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)


correctly_answered_questions = 0
questions_attempted = 0
for k, v in relations.items():

    word = ""
    for word_pairs in v:
        result = 0
        try:
            word_pair = word_pairs.split(" ")
            res_vector = find_vector(word_pair)

            for check_word in w_vectors.vocab:
                cs = cos_sim(res_vector, w_vectors[check_word])
                if cs > result:
                    result = cs
                    word = check_word

            questions_attempted += 1
            if word == word_pair[len(word_pair) - 1]:
                correctly_answered_questions += 1
        except Exception as e:
            questions_attempted += 1
            continue

    print(k, "accuracy is", (correctly_answered_questions / questions_attempted) * 100)

evaluation_part_one = (correctly_answered_questions / questions_attempted) * 100
print('Testing accuracy for analogy experiment is', evaluation_part_one)
