import math
from scipy import spatial
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class NewEuclidGlobalWeightCalculator:
    def __init__(self, containers):
        self.containers = containers

    def __var_power(self, var):
        total_connections = sum([len(container.atoms) for container in self.containers])

        def get_var_connections(var):
            return sum([len(list(filter(lambda x: x == var, container.atoms))) for container in self.containers])

        return 1.0 - 1.0 * get_var_connections(var) / total_connections

    def distance(self, first_container, second_container):
        vector = set(first_container.atoms)
        vector.update(second_container.atoms)

        def get_number_of_vars(container, var):
            return len(list(filter(lambda x: x == var, container.atoms)))

        result = math.sqrt(sum([
            (self.__var_power(var) * (
                    get_number_of_vars(first_container, var) - get_number_of_vars(second_container, var))) ** 2
            for var in vector
        ]))

        return result


class SciPyEuclidTFIDFCalculator:
    def __init__(self, containers, normalize=True):
        self.normalize = normalize
        self.containers = containers
        self.series = {
            c.name: self.__container_to_series(c, normalize)
            for c in containers
        }
        self.icf_cache = {}

    def __container_to_series(self, container, normalize):
        return pd.DataFrame(
            {'terms': [a.value for a in container.atoms]}
        )['terms'].value_counts(normalize=normalize)

    def __container_series(self, container, normalize=True):
        if container.name in self.series:
            return self.series[container.name]
        return self.__container_to_series(container, normalize)

    def __term_frequency(self, term, series):
        if term not in series.index.tolist():
            return 0.0
        return series[term]

    def __inverse_container_frequency(self, term):
        if term in self.icf_cache:
            return self.icf_cache[term]

        number_of_containers = len(self.containers)

        number_of_containers_with_term = len(list(filter(lambda ser: term in ser.index.tolist(), self.series.values())))
        if number_of_containers_with_term == 0:
            return 0
        result = math.log(1.0 * number_of_containers / number_of_containers_with_term)

        self.icf_cache[term] = result

        return result

    def distance(self, first_container, second_container):
        f_series = self.__container_series(first_container)
        s_series = self.__container_series(second_container)

        variables = list(set(list(f_series.index) + list(s_series.index)))

        f_vector, s_vector = [], []

        for var in variables:
            icf = self.__inverse_container_frequency(var)
            f_vector.append(self.__term_frequency(var, f_series) * icf)
            s_vector.append(self.__term_frequency(var, s_series) * icf)

        result = spatial.distance.euclidean(f_vector, s_vector)

        return result


class NewEuclidTFIDFCalculator:
    def __init__(self, containers):
        self.containers = containers
        self.icf_cache = {}

    def __term_frequency(self, term, container):
        number_of_term = len(list(filter(lambda x: x == term, container.atoms)))
        total_terms = len(container.atoms)
        if total_terms == 0.0:
            return 0.0
        result = 1.0 * number_of_term / total_terms
        return result

    def __inverse_container_frequency(self, term):
        if term in self.icf_cache:
            return self.icf_cache[term]

        number_of_containers = len(self.containers)
        number_of_containers_with_term = len(list(filter(lambda c: term in c.atoms, self.containers)))
        if number_of_containers_with_term == 0:
            return 0
        result = math.log(1.0 * number_of_containers / number_of_containers_with_term)

        self.icf_cache[term] = result

        return result

    def __term_weight(self, term, container):
        return self.__term_frequency(term, container) * self.__inverse_container_frequency(term)

    def distance(self, first_container, second_container):
        vector = set(first_container.atoms)
        vector.update(second_container.atoms)

        result = math.sqrt(sum([
            (self.__term_weight(term, first_container) - self.__term_weight(term, second_container)) ** 2
            for term in vector
        ]))
        return result


class NewCosineSimilarityCalculator:
    def __init__(self, containers):
        self.containers = containers
        self.variables = set([item for sublist in [c.atoms for c in self.containers] for item in sublist])

    def __vectors_product(self, first_container, second_container):
        variables = set(first_container.atoms + second_container.atoms)
        result = sum([
            len(list(filter(lambda x: x == var, first_container.atoms))) * len(
                list(filter(lambda x: x == var, second_container.atoms)))
            for var in variables
        ])
        return result

    def __vectors_power(self, first_container, second_container):
        variables = set(first_container.atoms + second_container.atoms)
        first_power = math.sqrt(sum([
            len(list(filter(lambda x: x == var, first_container.atoms))) ** 2
            for var in variables
        ]))
        second_power = math.sqrt(sum([
            len(list(filter(lambda x: x == var, second_container.atoms))) ** 2
            for var in variables
        ]))
        return first_power * second_power

    def distance(self, first_container, second_container):
        vectors_product = self.__vectors_product(first_container, second_container)
        vectors_power = self.__vectors_power(first_container, second_container)
        if not vectors_power:
            return 1.0
        return (1.0 - round((1.0 * vectors_product / vectors_power), 3))


class NewJaccardCoefficientCalculator:
    def __init__(self, containers):
        self.containers = containers

    def __vectors_product(self, first_container, second_container):
        variables = set(first_container.atoms + second_container.atoms)

        result = sum([
            len(list(filter(lambda x: x == var, first_container.atoms))) * len(
                list(filter(lambda x: x == var, second_container.atoms)))
            for var in variables
        ])
        return result

    def distance(self, first_container, second_container):
        vectors_product = self.__vectors_product(first_container, second_container)
        vectors_power = self.__vectors_product(first_container, first_container) + self.__vectors_product(
            second_container, second_container) - vectors_product
        return 1.0 - (1.0 * vectors_product / vectors_power)


class NPCosineSimilarityCalculator:
    def __init__(self, containers):
        self.containers = containers
        self.vocab = self.__make_vocab(self.containers)

    def __container_vocab(self, container):
        return Counter(container.atoms)

    def __container_to_vec(self, container, vocab):
        vec = np.zeros(len(vocab), dtype=np.int)
        dict1 = self.__container_vocab(container)
        for i, term in enumerate(vocab):
            vec[i] = dict1.get(term, 0)
        return vec

    def __make_vocab(self, containers):
        all_terms = set()

        for container in containers:
            all_terms.update(self.__container_vocab(container))

        return all_terms

    def fit_transform(self):
        items = []
        for container in self.containers:
            items.append(self.__container_to_vec(container, self.vocab))

        return np.array(items)

    def get_feature_names(self):
        return np.array(list(self.vocab))

    def get_document_names(self):
        return np.array(list([c.name for c in self.containers]))

    def distance(self, first_container, second_container):
        vocab = self.vocab

        v1 = self.__container_to_vec(first_container, vocab)
        v2 = self.__container_to_vec(second_container, vocab)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if not norm1 or not norm2:
            return 1.0
        return 1.0 - round((1.0 * np.dot(v1, v2) / (norm1 * norm2)), 3)


class NPNewCosineSimilarityCalculator:
    def __init__(self, containers):
        self.containers = containers
        self.vocab = self.__make_vocab(self.containers)
        self.matrix = self.fit_transform()
        self.dists = self.__calc_dists(self.matrix)
        self.documents = self.get_document_names()
        self.features = self.get_feature_names()

    def __calc_dists(self, matrix):
        dists = 1 - cosine_similarity(matrix)
        return dists

    def __container_vocab(self, container):
        return Counter(container.atoms)

    def __container_to_vec(self, container, vocab):
        vec = np.zeros(len(vocab), dtype=np.int)
        dict1 = self.__container_vocab(container)
        for i, term in enumerate(vocab):
            vec[i] = dict1.get(term, 0)
        return vec

    def __make_vocab(self, containers):
        all_terms = set()

        for container in containers:
            all_terms.update(self.__container_vocab(container))

        return all_terms

    def fit_transform(self):
        items = []
        for container in self.containers:
            items.append(self.__container_to_vec(container, self.vocab))

        return np.array(items)

    def get_feature_names(self):
        return np.array(list(self.vocab))

    def get_document_names(self):
        return np.array(list([c.name for c in self.containers]))

    def distance(self, first_container, second_container):
        documents = self.documents

        orig_doc = first_container.name
        orig_doc_indexes = np.where(documents == orig_doc)
        if not len(orig_doc_indexes) > 0 or not len(orig_doc_indexes[0]) > 0:
            return 2.0
        orig_doc_index = orig_doc_indexes[0][0]

        dest_doc = second_container.name
        dest_doc_indexes = np.where(documents == dest_doc)
        if not len(dest_doc_indexes) > 0 or not len(dest_doc_indexes[0]) > 0:
            return 2.0
        dest_doc_index = dest_doc_indexes[0][0]

        return self.dists[orig_doc_index][dest_doc_index]
