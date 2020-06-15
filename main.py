
# -------------------------------------------------------
# Assignment 2
# Written by Giselle Martel 26352936
# For COMP 472 Section JX Summer 2020
# --------------------------------------------------------

import csv
import math
import os
import ssl
import nltk
from enum import Enum

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
if not nltk:
    nltk.download()

from nltk.tokenize import RegexpTokenizer

BAYESIAN_SMOOTHING_VALUE = 0.5

def csv_to_array(file_name):
    lines = []
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lines.append(row)

    return lines

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

def print_data_to_file(data, filename):
    file = open(f'./generated-data/{filename}', "w+")
    for item in data:
        file.write(f'{item}\n')
    file.close()

def prompt_user_dataset_file_name():
    filename = './data/'
    while not os.path.isfile(filename):
        filename = input("Enter the name of the dataset file (must be located in ./data): ")
        filename = f'./data/{filename}'
        if not os.path.isfile(filename):
            print('Did not enter a valid file name! Try again')
    return filename

def prompt_user_year(type):
    year = input(f'Enter the year of the {type} data: ')
    return year

def prompt_user_experiment_type():
    experiment_type = None
    while type(experiment_type) != int or experiment_type < 1 or experiment_type > 4:

        display_str = 'Enter the number corresponding to the type of experiment you would like to conduct\n' \
                      '1 - Baseline\n' \
                      '2 - Stop-word filtering\n' \
                      '3 - Word-length filtering\n' \
                      '4 - Infrequent word filtering\n'
        experiment_type = int(input(display_str))

        if type(experiment_type) != int or experiment_type < 1 or experiment_type > 4:
            print('invalid selection! Try again.')

    return experiment_type

def get_experiment_filename(experiment_type):
    if experiment_type == ExperimentType.BASELINE:
        return 'baseline'
    if experiment_type == ExperimentType.STOP_WORD:
        return 'stopword'
    if experiment_type == ExperimentType.WORD_LEN:
        return 'wordlength'

class ExperimentType(Enum):
    BASELINE = 1
    STOP_WORD = 2
    WORD_LEN = 3
    INFREQ_WORD = 4

class Dataset:
    def __init__(self):
        self.experiment_type = ExperimentType.BASELINE
        self.classifiers = {}

    def display_classifiers(self):
        for classifier in self.classifiers:
            print(classifier, end=', ')

class ModelDocument:
    def __init__(self, raw_title, sanitized_title, rejected_words, classifier):
        self.raw_title = raw_title
        self.sanitized_title = sanitized_title
        self.rejected_words = rejected_words
        self.classifier = classifier

class ModelDataSet(Dataset):
    def __init__(self, year):
        super().__init__()
        self.total_word_count_foreach_class = {}
        self.conditional_probabilities = {}
        self.probabilities_classes = {}
        self.total_documents = 0
        self.num_docs_per_classifier = {}
        self.vocabulary = []
        # 2D array of the words rejected for each document
        self.rejected_words = []
        self.parent_ref = [self.classifiers, self.experiment_type]
        self.year = year

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
        self.conditional_probabilities = {}
        self.probabilities_classes = {}
        for classifier in self.classifiers:
            for word in self.vocabulary:
                if word not in self.conditional_probabilities:
                    self.conditional_probabilities[word] = {}
                # determine the probability of having the curr word given the curr class
                self.conditional_probabilities[word][classifier] = self.calculate_conditional_probability(word, classifier)

            # determine the probability of the current class within the model
            self.probabilities_classes[classifier] = self.calculate_classifier_probability(classifier)

    def display_conditional_probabilities(self):
        for p in self.conditional_probabilities:
            print(p, end=': ')
            print(f'{round(self.conditional_probabilities[p] * 100, 6)} %')

    def display_probabilities_classes(self):
        for p in self.probabilities_classes:
            print(p, end=': ')
            print(f'{round(self.probabilities_classes[p] * 100, 6)} %')

    def display_vocabulary(self):
        for word in self.vocabulary:
            print(word)

    def display_word_frequencies(self):
        for category in self.classifiers:
            print(category)
            word_freq_map = self.classifiers[category]
            for word in word_freq_map:
                print(f'{word}: {word_freq_map[word]}')
            print()

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

            self.total_word_count_foreach_class[classifier] += 1

    def write_dataset_model_to_file(self, experiment_type):
        if experiment_type != ExperimentType.INFREQ_WORD:
            global filename
            if experiment_type == ExperimentType.BASELINE:
                filename = f'./generated-data/model-{self.year}.txt'
            elif experiment_type == ExperimentType.STOP_WORD:
                filename = f'./generated-data/stopword-model.txt'
            elif experiment_type == ExperimentType.WORD_LEN:
                filename = f'./generated-data/wordlength-model.txt'

            file = open(filename, "w+")
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
                        cond_probability = round(self.conditional_probabilities[word][classifier], 8)
                        # add data for word and current class to line
                        line += (f'{delim}{frequency}{delim}{cond_probability}')
                    elif classifier in classifiers and word not in classifiers[classifier]:
                        frequency = 0
                        cond_probability = round(self.conditional_probabilities[word][classifier], 8)
                        # add data for word and current class to line
                        line += (f'{delim}{frequency}{delim}{cond_probability}')
                    else:
                        frequency = 0
                        cond_probability = 0.0
                        # add data for word and current class to line
                        line += (f'{delim}{frequency}{delim}{cond_probability}')
                # store line to be written later to file
                doc_lines_to_print.append(line)

            # prepend with line #
            for i, line in enumerate(doc_lines_to_print):
                doc_lines_to_print[i] = f'{i+1}{delim}{line}\n'

            # write data to file
            for l in doc_lines_to_print:
                file.write(l)

            file.close()

class TestDocument:
    def __init__(self, title, classifier):
        self.title = title
        self.true_class = classifier
        self.generated_class = ''
        self.class_scores = {}
        self.is_prediction_correct = False

class TestDataSet(Dataset):

    def __init__(self, year):
        super().__init__()
        self.documents = []
        self.year = year

    def write_test_results_to_file(self, experiment_type):
        if experiment_type != ExperimentType.INFREQ_WORD:
            filename = get_experiment_filename(experiment_type)
            file = open(f'./generated-data/{filename}-result.txt', "w+")

            classifier_order = [
                'story',
                'ask_hn',
                'show_hn',
                'poll'
            ]

            delim = '  '
            doc_lines_to_print = []
            line_ctr = 1

            for document in self.documents:
                line = f'{line_ctr}{delim}{document.title}{delim}{document.generated_class}{delim}'
                for classifier in classifier_order:
                    if classifier in document.class_scores:
                        line += f'{document.class_scores[classifier]}{delim}'
                    else:
                        line += f'0{delim}'

                global label
                if document.is_prediction_correct:
                    label = 'right'
                else:
                    label = 'wrong'

                line += f'{document.true_class}{delim}{label}\n'
                line_ctr += 1
                # store line to be written later to file
                doc_lines_to_print.append(line)

            # write data to file
            for l in doc_lines_to_print:
                file.write(l)

            file.close()

class GraphData():
    def __init__(self):
        self.vocabulary_sizes = []
        self.classifier_scores = []

class NaiveBayesianClassifier:

    def __init__(self, data,  model_year, test_year):
        self.raw_data = data
        self.sanitized_model_documents = []
        self.sanitized_test_documents = []
        self.dataset_model = ModelDataSet(model_year)
        self.dataset_test = TestDataSet(test_year)
        self.stop_words = []
        self.word_filtering_graph_data_1 = GraphData()
        self.word_filtering_graph_data_2 = GraphData()

    def calc_success_rate(self):
        num_incorrect_classifications = 0

        for document in self.dataset_test.documents:
            if not document.is_prediction_correct:
                num_incorrect_classifications += 1

        return (1 - num_incorrect_classifications / len(self.dataset_test.documents)) * 100

    def display_test_result(self):
            num_incorrect_classifications = 0

            for document in self.dataset_test.documents:
                if not document.is_prediction_correct:
                    num_incorrect_classifications += 1

            if self.dataset_test.experiment_type == ExperimentType.BASELINE:
                label = 'Baseline'
            if self.dataset_test.experiment_type == ExperimentType.STOP_WORD:
                label = 'Stop-word'
            if self.dataset_test.experiment_type == ExperimentType.WORD_LEN:
                label = 'Word-length'
            if self.dataset_test.experiment_type == ExperimentType.INFREQ_WORD:
                label = 'Infrequent word filtering'

            success_rate = ( 1 - num_incorrect_classifications/len(self.dataset_test.documents)) * 100

            print(f'* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n\n'
                  f'{label} experiment yielded a classification success rate of: {success_rate}%\n')
            print(f'Model Statistics:\n'
                  f'# words in model vocabulary: {len(self.dataset_model.vocabulary)}\n'
                  f'# documents in test dataset: {len(self.dataset_test.documents)}\n'
                  f'# correctly classified documents: {len(self.dataset_test.documents) - num_incorrect_classifications}\n'
                  f'# incorrectly classified documents: {num_incorrect_classifications}')

    def sanitize_document(self, document):
        # generate the sanitized words from the current line based on the regex
        regex = r'(\w+\,\w+)|(\w+\'\w+)|(\w+\’\w+)|(\w+-\w+(-*\w*)+)|(?!(\w+\,\w+)|(\w+\'\w+)|(\w+\’\w+)|(\w+-\w+(-*\w*)+))(\w+)'
        tokenizer = RegexpTokenizer(regex)
        sanitized_words_tuples = tokenizer.tokenize(document)
        sanitized_words = []
        for entry in sanitized_words_tuples:
            matches = [e for e in entry if len(e) > 0]
            if len(matches) > 1:
                print('Something went wrong with the tokenization')
            for match in matches:
                if len(match) > 0:
                    sanitized_words.append(match)
        return ' '.join(w for w in sanitized_words)

    def parse_stop_words_from_file(self):
        with open('./data/stopwords.txt') as file:
            lines = file.read().splitlines()
        for word in lines:
            self.stop_words.append(word)

    def parse_data_baseline(self):
        # map to store which columns the desired data categories are contained
        data_categories_indices = {
            'title': -1,
            'year': -1,
            'class': - 1
        }

        model_doc_ctr = 0

        for i, data in enumerate(self.raw_data):
            # first row, parse the col categories
            if i == 0:
                for i, category in enumerate(data):
                    # cleanup the string into readable format
                    category = ''.join(category.split()).lower()

                    if category == 'title':
                        data_categories_indices['title'] = i
                    elif category == 'posttype':
                        data_categories_indices['class'] = i
                    elif category == 'year':
                        data_categories_indices['year'] = i
            else:
                year = data[data_categories_indices['year']]
                classifier = data[data_categories_indices['class']]
                document = data[data_categories_indices['title']].lower()

                # sanitize the current document (title from csv file)
                sanitized_document = self.sanitize_document(document)

                # store the sanitized document to be used to develop model for experiments later
                if year == self.dataset_model.year:
                    # determine which char are rejected based on set difference between the raw doc and sanitized doc
                    sanitized_words_chars = list(sanitized_document)
                    raw_document_chars = list(document)
                    rejected_chars = list((nltk.Counter(raw_document_chars) - nltk.Counter(sanitized_words_chars)).elements())
                    # parse the rejected chars back into words (for printing later)
                    rejected_words = [c for c in rejected_chars]

                    self.sanitized_model_documents.append(
                        ModelDocument(document, sanitized_document, rejected_words, classifier))

                    # determine the frequency of each word for each classifier for model
                    if classifier not in self.dataset_model.num_docs_per_classifier:
                        self.dataset_model.num_docs_per_classifier[classifier] = 1
                    else:
                        current_num_docs = self.dataset_model.num_docs_per_classifier[classifier]
                        self.dataset_model.num_docs_per_classifier[classifier] = current_num_docs + 1
                    model_doc_ctr += 1
                else:
                    self.sanitized_test_documents.append(TestDocument(sanitized_document, classifier))

        # set the total num of documents for the model (to be used later to calc probability of each class)
        self.dataset_model.total_documents = model_doc_ctr

    def generate_vocabulary(self, experiment_type):
        self.dataset_model.experiment_type = experiment_type
        self.dataset_test.experiment_type = experiment_type
        if experiment_type != ExperimentType.INFREQ_WORD:
            # reset the vocabulary
            self.dataset_model.vocabulary = []
            if experiment_type == ExperimentType.STOP_WORD:
                self.parse_stop_words_from_file()

            for i, doc in enumerate(self.sanitized_model_documents):
                words_in_doc = doc.sanitized_title.split()

                if experiment_type == ExperimentType.BASELINE:
                    self.dataset_model.generate_frequency_maps(doc.classifier, words_in_doc)

                elif experiment_type == ExperimentType.STOP_WORD:
                    for word in words_in_doc:
                        if word in self.stop_words:
                            words_in_doc.remove(word)
                            doc.rejected_words.append(word)

                    self.dataset_model.generate_frequency_maps(doc.classifier, words_in_doc)

                elif experiment_type == ExperimentType.WORD_LEN:
                    for word in words_in_doc:
                        if len(word) <= 2 or len(word) >= 9:
                            words_in_doc.remove(word)
                            doc.rejected_words.append(word)

                    self.dataset_model.generate_frequency_maps(doc.classifier, words_in_doc)

                if len(doc.rejected_words) > 0:
                    rejected_words_line = '\t'.join(w for w in doc.rejected_words)
                else:
                    rejected_words_line = ""

                self.dataset_model.rejected_words.append(rejected_words_line)
            # write rejected words to file
            print_data_to_file(self.dataset_model.rejected_words, 'remove_word.txt')
            # write the vocabulary to file
            self.dataset_model.vocabulary.sort()
            print_data_to_file(self.dataset_model.vocabulary, 'vocabulary.txt')

    def generate_testing_data(self):
        self.dataset_test.documents = self.sanitized_test_documents
        # if self.dataset_test.experiment_type == ExperimentType.BASELINE:
        #     self.dataset_test.documents = self.sanitized_test_documents
        # else:
        #     for i, doc in enumerate(self.sanitized_test_documents):
        #         words_in_doc = doc.title.split()
        #
        #         if self.dataset_test.experiment_type == ExperimentType.STOP_WORD:
        #             for word in words_in_doc:
        #                 if word in self.stop_words:
        #                     words_in_doc.remove(word)
        #
        #         elif self.dataset_test.experiment_type == ExperimentType.WORD_LEN:
        #             for word in words_in_doc:
        #                 if len(word) <= 2 or len(word) >= 9:
        #                     words_in_doc.remove(word)
        #
        #         elif self.dataset_test.experiment_type  == ExperimentType.INFREQ_WORD:
        #             # TODO: infreq experiement
        #             pass
        #
        #         # add the documents for the testing dataset (including their true classification)
        #         # classification to be approximated by naive bayesian classifier
        #         self.dataset_test.documents.append(doc)

    def generate_least_frequent_word_filtering(self):
        frequency_thresholds = [
            1, 5, 10, 15, 20
        ]

        for freq in frequency_thresholds:
            # reset the vocabulary
            self.dataset_model.vocabulary = []
            for doc in self.sanitized_model_documents:
                words_in_doc = doc.sanitized_title.split()
                self.dataset_model.generate_frequency_maps(doc.classifier, words_in_doc)

                if len(doc.rejected_words) > 0:
                    rejected_words_line = '\t'.join(w for w in doc.rejected_words)
                else:
                    rejected_words_line = ""
                self.dataset_model.rejected_words.append(rejected_words_line)

            for classifier in self.dataset_model.classifiers:
                for word in self.dataset_model.classifiers[classifier]:
                    frequency = self.dataset_model.classifiers[classifier][word]
                    if frequency <= freq and frequency > 0 and word in self.dataset_model.vocabulary:
                        self.dataset_model.vocabulary.remove(word)
                        self.dataset_model.classifiers[classifier][word] = 0
                        self.dataset_model.total_word_count_foreach_class[classifier] -= 1

            # TODO: parse the vocab size as x values, and the score as y-values
            self.word_filtering_graph_data_1.vocabulary_sizes.append(len(self.dataset_model.vocabulary))

            self.generate_testing_data()
            self.dataset_model.train_dataset_model()
            self.classify_test_dataset()
            print(self.calc_success_rate())

    def generate_most_frequent_word_filtering(self):
        frequency_thresholds = [
            0.05, 0.10, 0.15, 0.20, 0.25
        ]

        for freq in frequency_thresholds:
            # reset the vocabulary
            self.dataset_model.vocabulary = []
            for i, doc in enumerate(self.sanitized_model_documents):
                pass

            # TODO: parse the vocab size as x values, and the score as y-values
            self.word_filtering_graph_data_2.vocabulary_sizes.append(len(self.dataset_model.vocabulary))

        self.generate_testing_data()
        self.dataset_model.train_dataset_model()
        self.classify_test_dataset()

    def classify_test_dataset(self):
        for document in self.dataset_test.documents:
            self.classify(document)

    def classify(self, document):
        max_score = -math.inf
        for classifier in self.dataset_model.classifiers:
            score = self.dataset_model.probabilities_classes[classifier]
            for word in document.title.split():
                if word in self.dataset_model.vocabulary:
                    score = score + math.log10(self.dataset_model.conditional_probabilities[word][classifier])
                    document.class_scores[classifier] = score

            if score > max_score:
                max_score = score
                document.generated_class = classifier

        if document.generated_class == document.true_class:
            document.is_prediction_correct = True

    def do_experiment(self, experiment_type):
        if experiment_type != ExperimentType.INFREQ_WORD:
            self.generate_vocabulary(experiment_type)
            self.generate_testing_data()
            self.dataset_model.train_dataset_model()
            self.dataset_model.write_dataset_model_to_file(experiment_type)
            # run the Naive Bayes Classifier
            self.classify_test_dataset()
            self.dataset_test.write_test_results_to_file(experiment_type)
            self.display_test_result()
        else:
            self.generate_least_frequent_word_filtering()
            # TODO: finish this
            # self.generate_most_frequent_word_filtering()




def main():
    # debug_print_csv()

    # file_name = prompt_user_dataset_file_name()
    # model_year = prompt_user_year('model')
    # test_year = prompt_user_year('test')
    # experiment_type = prompt_user_experiment_type()

    file_name = './data/hns_2018_2019.csv'
    model_year = '2018'
    test_year = '2019'

    data = csv_to_array(file_name)

    classifier = NaiveBayesianClassifier(data, model_year, test_year)
    classifier.parse_data_baseline()

    # classifier.do_experiment(ExperimentType.BASELINE)
    # classifier.do_experiment(ExperimentType.STOP_WORD)
    # classifier.do_experiment(ExperimentType.WORD_LEN)
    classifier.do_experiment(ExperimentType.INFREQ_WORD)


if __name__ == '__main__':
    main()
