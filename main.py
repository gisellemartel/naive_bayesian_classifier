
# -------------------------------------------------------
# Assignment 2
# Written by Giselle Martel 26352936
# For COMP 472 Section JX Summer 2020
# --------------------------------------------------------

import csv
import difflib

import numpy as np
import re
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
if not nltk:
    nltk.download()

from nltk.tokenize import RegexpTokenizer

def debug_print_csv():
    with open("./data/hns_2018_2019.csv",'r') as f:
        rowReader = csv.reader(f, delimiter=',')
        lines = []
        for values in rowReader:
            if values[9] == '2018':
                lines.append(f'{values[3]}\t\t{values[2]}')
        lines.sort()

        for l in lines:
            print(l)

BAYESIAN_SMOOTHING_VALUE = 0.5
DATASET_MODEL_YEAR = '2018'
DATASET_TEST_YEAR = '2019'


def print_data_to_file(data, filename):
    file = open(f'./generated-data/{filename}', "w+")
    for item in data:
        file.write(f'{item}\n')
    file.close()

class Dataset:

    def __init__(self):
        self.vocabulary = []
        self.classifiers = {}

    def display_vocabulary(self):
        for word in self.vocabulary:
            print(word)

    def display_classifiers(self):
        for classifier in self.classifiers:
            print(classifier, end=', ')

    def display_word_frequencies(self):
        for category in self.classifiers:
            print(category)
            word_freq_map = self.classifiers[category]
            for word in word_freq_map:
                print(f'{word}: {word_freq_map[word]}')
            print()

    def generate_frequency_maps_base(self):
        return self.generate_frequency_maps_base()

class ModelDataSet(Dataset):
    def __init__(self):
        super().__init__()
        self.total_word_count_foreach_class = {}
        self.conditional_probabilities = {}
        self.probabilities_classes = {}
        self.total_documents = 0
        self.num_docs_per_classifier = {}
        self.parent_ref = [self.classifiers, self.vocabulary]

    def calculate_conditional_probability(self, word, classifier):
        # P(word | classifier) = freq of word in classifier / total words in classifier
        if word in self.classifiers[classifier]:
            word_count = self.classifiers[classifier][word]
        else:
            word_count = 0

        total_word_count = self.total_word_count_foreach_class[classifier]

        # return cond probability with smoothing
        return (word_count + BAYESIAN_SMOOTHING_VALUE) / (total_word_count + len(self.vocabulary))

    def calculate_classifier_probability(self, classifier):
        return self.num_docs_per_classifier[classifier] / self.total_documents

    def train_dataset_model(self):
        for classifier in self.classifiers:
            for word in self.vocabulary:
                self.conditional_probabilities[word] \
                    = self.calculate_conditional_probability(word, classifier)
            self.probabilities_classes[classifier] = self.calculate_classifier_probability(classifier)

    def display_conditional_probabilities(self):
        for p in self.conditional_probabilities:
            print(p, end=': ')
            print(f'{round(self.conditional_probabilities[p] * 100, 6)} %')

    def display_probabilities_classes(self):
        for p in self.probabilities_classes:
            print(p, end=': ')
            print(f'{round(self.probabilities_classes[p] * 100, 6)} %')

    def generate_frequency_maps(self, classifier, words):
        # add the classifier to the model if does not exist yet
        if classifier not in self.classifiers:
            self.classifiers[classifier] = {}

        if classifier not in self.total_word_count_foreach_class:
            self.total_word_count_foreach_class[classifier] = 0

        for word in words:
            #  add the word to the vocabulary for the model
            if word not in self.vocabulary:
                self.vocabulary.append(word)

            # generate the frequency of each word for each classifier
            if word not in self.classifiers[classifier]:
                self.classifiers[classifier][word] = 1
            else:
                current_freq = self.classifiers[classifier][word]
                self.classifiers[classifier][word] = current_freq + 1

            total_word_count = self.total_word_count_foreach_class[classifier]
            self.total_word_count_foreach_class[classifier] = total_word_count + 1

        self.vocabulary.sort()

    def write_dataset_model_to_file(self):
        file = open(f'./generated-data/model-{DATASET_MODEL_YEAR}.txt', "w+")

        classifier_order = [
            'story',
            'ask_hn',
            'show_hn',
            'poll'
        ]

        delim = '  '
        doc_lines_to_print = []

        for word in self.vocabulary:
            line = word
            classifiers = self.parent_ref[0]
            for classifier in classifier_order:
                if classifier in classifiers and word in classifiers[classifier]:
                    frequency = classifiers[classifier][word]
                    cond_probability = round(self.conditional_probabilities[word], 8)
                    # add data for word and current class to line
                    line += (f'{delim}{frequency}{delim}{cond_probability}')
                elif classifier in classifiers and word not in classifiers[classifier]:
                    frequency = 0
                    cond_probability = round(self.conditional_probabilities[word], 8)
                    # add data for word and current class to line
                    line += (f'{delim}{frequency}{delim}{cond_probability}')
                else:
                    frequency = 0
                    cond_probability = 0.0
                    # add data for word and current class to line
                    line += (f'{delim}{frequency}{delim}{cond_probability}')
            # store line to be written later to file
            doc_lines_to_print.append(line)

        # sort entries alphabetically
        doc_lines_to_print.sort()

        # prepend with line #
        for i, line in enumerate(doc_lines_to_print):
            doc_lines_to_print[i] = f'{i+1}{delim}{line}\n'

        # write data to file
        for l in doc_lines_to_print:
            file.write(l)

        file.close()

class TestDataSet(Dataset):

    def __init__(self):
        super().__init__()

    def generate_frequency_maps(self, classifier, words):
        # add the classifier to the model if does not exist yet
        if classifier not in self.classifiers:
            self.classifiers[classifier] = {}

        # if classifier not in self.total_word_count_foreach_class:
        #     self.total_word_count_foreach_class[classifier] = 0

        for word in words:
            #  add the word to the vocabulary for the model
            if word not in self.vocabulary:
                self.vocabulary.append(word)

            # generate the frequency of each word for each classifier
            if word not in self.classifiers[classifier]:
                self.classifiers[classifier][word] = 1
            else:
                current_freq = self.classifiers[classifier][word]
                self.classifiers[classifier][word] = current_freq + 1
            #
            # total_word_count = self.total_word_count_foreach_class[classifier]
            # self.total_word_count_foreach_class[classifier] = total_word_count + 1


class NaiveBayesianClassifier:

    def __init__(self, csv_file_name):
        self.csv_file_name = csv_file_name
        self.dataset_model = ModelDataSet()
        self.dataset_test = TestDataSet()

    def read_csv_data(self):
        # map to store which columns the desired data categories are contained
        data_categories_indices = {
            'title': -1,
            'year': -1,
            'class': - 1
        }

        with open(self.csv_file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            current_row = 0
            all_rejected_words = []

            debug_print_titles = []

            for row in csv_reader:
                # first row, parse the col categories
                if current_row == 0:
                    for i, category in enumerate(row):
                        # cleanup the string into readable format
                        category = ''.join(category.split()).lower()

                        if category == 'title':
                            data_categories_indices['title'] = i
                        elif category == 'posttype':
                            data_categories_indices['class'] = i
                        elif category == 'year':
                            data_categories_indices['year'] = i

                    current_row += 1
                else:
                    year = row[data_categories_indices['year']]
                    classifier = row[data_categories_indices['class']]
                    title_string = row[data_categories_indices['title']].lower()
                    raw_words_chars = list(title_string)

                    debug_print_titles.append(title_string)

                    # generate the sanitized words from the current line
                    regex = r'(\w+\,\w+)|(\w+\'\w+)|(\w+\’\w+)|(\w+-\w+(-*\w*)+)|(?!((\w+\,\w+)|(\w+\'\w+)|(\w+\’\w+)|(\w+-\w+(-*\w*)+)))(\w+)'
                    tokenizer = RegexpTokenizer(regex)
                    sanitized_words_tuples = tokenizer.tokenize(title_string)
                    sanitized_words = []
                    rejected_words = []

                    for entry in sanitized_words_tuples:
                        matches = [e for e in entry if len(e) > 0]
                        if len(matches) > 1:
                            print('Something went wrong with the tokenization')
                        for match in matches:
                            if len(match) > 0:
                                sanitized_words.append(match)

                    sanitized_words_chars = list(' '.join(w for w in sanitized_words))

                    rejected_chars = list((nltk.Counter(raw_words_chars) - nltk.Counter(sanitized_words_chars)).elements())

                    for c in rejected_chars:
                        rejected_words.append(c)

                    if len(rejected_words) > 0:
                        rejected_words_line = '\t'.join(w for w in rejected_words)
                    else:
                        rejected_words_line=""

                    all_rejected_words.append(rejected_words_line)

                    # determine the frequency of each word for each classifier
                    if year == DATASET_MODEL_YEAR:
                        if classifier not in self.dataset_model.num_docs_per_classifier:
                            self.dataset_model.num_docs_per_classifier[classifier] = 1
                        else:
                            current_num_docs = self.dataset_model.num_docs_per_classifier[classifier]
                            self.dataset_model.num_docs_per_classifier[classifier] = current_num_docs + 1

                        self.dataset_model.generate_frequency_maps(classifier, sanitized_words)

                    elif year == DATASET_TEST_YEAR:
                        self.dataset_test.generate_frequency_maps(classifier, sanitized_words)

                    current_row += 1

            print_data_to_file(debug_print_titles, 'debug_titles.txt')

            # get the total num of documents
            self.dataset_model.total_documents = current_row

            # write rejected words to file
            print_data_to_file(all_rejected_words, 'remove_word.txt')
            # write the vocabulary to file
            print_data_to_file(self.dataset_model.vocabulary, 'vocabulary.txt')


def main():
    debug_print_csv()
    # TODO: prompt user for name of dataset file, model year, and training year
    naive_bayesian_classifier = NaiveBayesianClassifier('./data/hns_2018_2019.csv')
    naive_bayesian_classifier.read_csv_data()

    naive_bayesian_classifier.dataset_model.train_dataset_model()
    naive_bayesian_classifier.dataset_model.write_dataset_model_to_file()

    # debug
    # naive_bayesian_classifier.dataset_model.display_conditional_probabilities()
    # naive_bayesian_classifier.dataset_model.display_probabilities_classes()


if __name__ == '__main__':
    main()
