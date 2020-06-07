
# -------------------------------------------------------
# Assignment 2
# Written by Giselle Martel 26352936
# For COMP 472 Section JX Summer 2020
# --------------------------------------------------------

import csv
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

# parse the files in the training set and build a vocabulary with all the words it contains in Title for 2018. 
# Then for each word, compute their frequencies and the probabilities of each
# classes are: (story, ask_hn, show_hn and poll).
# 
# Extract the data from 2019 as the testing dataset.

# To process the texts, fold the Title to lowercase, then tokenize and use the set of resulting word as your vocabulary.
# For each word wi in the training set, save its frequency and its conditional probability for each Post Type class:
#  P(wi|story), P(wi|ask_hn), P(wi|show_hn) and P(wi|poll).
# 
# These probabilities must be smoothed with 0.5.

# Save your model in the text file named model-2018.txt. The format of this file must be the following:
# 1. A line counter i, followed by 2 spaces.
# 2. The word wi, followed by 2 spaces.
# 3. The frequency of wi in the class story, followed by 2 spaces.
# 4. The smoothed conditional probability of wi in story - P(wi|story), followed by 2 spaces.
# 5. The frequency of wi in the class ask_hn, followed by 2 spaces.
# 6. The smoothed conditional probability of wi in ask_hn - P(wi|ask_hn), followed by 2 spaces.
# 7. The frequency of wi in the class show_hn, followed by 2 spaces.
# 8. The smoothed conditional probability of wi in show_hn - P(wi|show_hn), followed by 2 spaces.
# 9. The frequency of wi in the class poll, followed by 2 spaces.
# 10. The smoothed conditional probability of wi in poll - P(wi|poll), followed by a carriage return.

# Your file must be sorted alphabetically. 
# For the four different Post Type class, ask_hn and show_hn should be considered as two words (ask_hn and show_hn) 
# not 4 words in your vocabulary. For example, your files model-2018.txt could look like the following:
# 1 block 3 0.003 40 0.4 10 0.014 4 0.04
# 2 ask-hn 3 0.003 40 0.4 40 0.034 40 0.0024 
# 3 query 40 0.4 50 0.03 20 0.00014 15 0.4
# 4 show-hn 0.7 0.003 0 0.000001 30 0.4 2 0.4

# Please note:
# 1. The values presented above is an example not based on the real data.
# 2. Your program should be able to output a file named "vocabulary.txt", which should contain all the words in your vocabulary.
# f the title includes some words, which you think is not useful for classification, you can remove it from you vocabulary, 
# but you need put all the removed words in an independent file named "remove_word.txt".

BAYESIAN_SMOOTHING_VALUE = 0.5
DATASET_MODEL_YEAR = '2018'
DATASET_TEST_YEAR = '2019'

class Dataset:

    def __init__(self):
        self.vocabulary = []
        self.classifiers = {}
        self.total_word_count_foreach_class = {}
        self.conditional_probabilities = {}
        self.probabilities_classes = {}
        self.total_documents = 0
        self.num_docs_per_classifier = {}

    def display_vocabulary(self):
        for word in self.vocabulary:
            print(word)

    def display_classifiers(self):
        for classifier in self.classifiers:
            print(classifier, end=', ')

    def display_conditional_probabilities(self):
        for p in self.conditional_probabilities:
            print(p, end=': ')
            print(self.conditional_probabilities[p], end=', ')

    def display_probabilities_classes(self):
        for p in self.probabilities_classes:
            print(p, end=': ')
            print(self.probabilities_classes[p], end=', ')


class NaiveBayesianClassifier:

    def __init__(self, csv_file_name):
        self.csv_file_name = csv_file_name
        self.dataset_model = Dataset()
        self.dataset_test = Dataset()

    def calculate_conditional_probability(self, word, classifier):
        # P(word | classifier) = freq of word in classifier / total words in classifier
        word_count = self.dataset_model.classifiers[classifier][word]
        total_word_count = self.dataset_model.total_word_count_foreach_class[classifier]

        # return cond probability with smoothing
        return (word_count + BAYESIAN_SMOOTHING_VALUE) / (total_word_count + len(self.dataset_model.vocabulary))

    def calculate_classifier_probability(self, classifier):
        return self.dataset_model.num_docs_per_classifier[classifier] / self.dataset_model.total_documents


    def train_dataset_model(self):
        for classifier in self.dataset_model.classifiers:
            for word in self.dataset_model.vocabulary:
                self.dataset_model.conditional_probabilities[word] \
                    = self.calculate_conditional_probability(word, classifier)
            self.dataset_model.probabilities_classes[classifier] = self.calculate_classifier_probability(classifier)


    def generate_frequency_maps(self, dataset, classifier, words):
        # add the classifier to the model if does not exist yet
        if classifier not in dataset.classifiers:
            dataset.classifiers[classifier] = {}

        if classifier not in dataset.total_word_count_foreach_class:
            dataset.total_word_count_foreach_class[classifier] = 0
        
        for word in words:
            #  add the word to the vocabulary for the model
            if word not in dataset.vocabulary:
                dataset.vocabulary.append(word)
            
            # generate the frequency of each word for each classifier
            if word not in dataset.classifiers[classifier]:
                dataset.classifiers[classifier][word] = 1
            else:
                current_freq = dataset.classifiers[classifier][word]
                dataset.classifiers[classifier][word] = current_freq + 1

            total_word_count = dataset.total_word_count_foreach_class[classifier]
            dataset.total_word_count_foreach_class[classifier] = total_word_count + 1

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

                    # get all the words in the current line
                    tokenizer = RegexpTokenizer(r'\w+')
                    words = tokenizer.tokenize(title_string)

                    # determine the frequency of each word for each classifier
                    if year == DATASET_MODEL_YEAR:
                        if classifier not in self.dataset_model.num_docs_per_classifier:
                            self.dataset_model.num_docs_per_classifier[classifier] = 1
                        else:
                            current_num_docs = self.dataset_model.num_docs_per_classifier[classifier]
                            self.dataset_model.num_docs_per_classifier[classifier] = current_num_docs + 1

                        self.generate_frequency_maps(self.dataset_model, classifier, words)

                    elif year == DATASET_TEST_YEAR:
                        self.generate_frequency_maps(self.dataset_test, classifier, words)

                    current_row += 1

            self.dataset_model.total_documents = current_row


def main():
    naive_bayesian_classifier = NaiveBayesianClassifier('./data/hns_2018_2019.csv')
    naive_bayesian_classifier.read_csv_data()

    # print(len(naive_bayesian_classifier.dataset_model.vocabulary))
    # naive_bayesian_classifier.dataset_model.display_classifiers()
    # print()
    # print(len(naive_bayesian_classifier.dataset_test.vocabulary))
    # naive_bayesian_classifier.dataset_test.display_classifiers()

    for category in naive_bayesian_classifier.dataset_model.classifiers:
        print(category)
        word_freq_map = naive_bayesian_classifier.dataset_model.classifiers[category]
        for word in word_freq_map:
            print(f'{word}: {word_freq_map[word]}')
        print()


if __name__ == '__main__':
    main()
