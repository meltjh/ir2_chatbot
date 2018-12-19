import json
from collections import Counter, OrderedDict
from ChatsDataset import ChatsDataset
from torch.utils.data import Dataset, DataLoader
from word2id import Word2Id
import matplotlib.pyplot as plt
import time
import nltk
import re
import numpy as np

def read(filename, word2id, add_new_words, min_occurence, glove_vocab = None):
    """
    Returns: template_data, user_data. Both as id's
    template_data: {chat_id:[template1,...]}
    user_data: [{"id":chat_id, "in":sentence_in, "out":sentence_out}, ...]
    """
    all_data = []
    all_words = []
    all_data2id = []

    print("Reading the file")
    start = time.time()
    with open(filename) as file:
        json_data = json.load(file)
        data = json_data["data"]
        for line in data:
            paragraph = line["paragraphs"][0] # list only consists of one paragraph
            context = paragraph["context"] #.lower()
            qas = paragraph["qas"][0] # list only consists of one Q&A
            question = qas["question"] #.lower()
            answer_info = qas["answers"][0] # list only consists of one answer
            answer = answer_info["text"] #.lower()
            answer_start = answer_info["answer_start"]
            qa_id = qas["id"]

            question = process_tokens(word_tokenize(question))
            answer = process_tokens(word_tokenize(answer))

            context = context.replace("''", '" ').replace("``", '" ')
            resources = list(map(word_tokenize, nltk.sent_tokenize(context)))
            resources = [process_tokens(tokens) for tokens in resources]

            all_words.extend(question)
            all_words.extend(answer)
            for sentence in resources:
                all_words.extend(sentence)

            # if training, have to go through all the data first
            if add_new_words:
                all_data.append({"qa_id": qa_id, "question": question, "answer": answer,
                                 "answer_start": answer_start, "resources": resources})

            # if testing or validation, can already convert to ids
            else:
                question2id, answer2id, resources2id = word2id.datapoint2id(question, answer, resources)
                all_data2id.append({"qa_id": qa_id, "question": question2id, "answer": answer2id,
                                    "answer_start": answer_start, "resources": resources2id})

    end = time.time()
    print("-- Finished reading the file")
    print("It took {:.2f} seconds\n".format(end-start))
    start = time.time()
    # if training, have to get the most common words first
    if add_new_words:
        print("Adding words to the word2id")

        # get the most common words that occur in the GLoVE set and convert them to ids
        counter = Counter(all_words)
#        plot_word_counts(counter)
        
        # TODO Om de frequencies ff te plotten
#        most_common = counter.most_common()

        word2id.frequent_words2id(counter, glove_vocab, min_occurence)
        print("Getting the training set\n")
        # convert all the data to ids
        for datapoint in all_data:
            qa_id = datapoint["qa_id"]
            question = datapoint["question"]
            answer = datapoint["answer"]
            resources = datapoint["resources"]
            answer_start = datapoint["answer_start"]

            question2id, answer2id, resources2id = word2id.datapoint2id(question, answer, resources)

            all_data2id.append({"qa_id": qa_id, "question": question2id,
                                "answer": answer2id, "answer_start": answer_start,
                                "resources": resources2id})
        end = time.time()
        print("Finished adding words to the word2id")
        print("It took {:.2f} seconds\n".format(end-start))

    return all_data2id
#
#def plot_word_counts(counter):
#    freqs = Counter([val for _, val in counter.items()])
#    sorted_freqs = OrderedDict(sorted(freqs.items()))
#    
#    max_freq = list(sorted_freqs.keys())[-1]
#    
#    x = np.arange(max_freq)
#    
#    vals = [val2 for _, val2 in sorted_freqs.items()]
#    idx = np.arange(len(sorted_freqs))
#    
#    bar_width = 0.35
#
#    plt.bar(idx, vals)
#    plt.
#
#    # add labels
#    plt.xticks(idx + bar_width, sorted_freqs.keys())
#    plt.show()
#    exit

def print_counts(template_data, user_data):
    # TODO: nog goed maken.
    template_lengths = []
    for key, templates in template_data.items():
        for template in templates:
            template_lengths.append(len(template))

    plt.title("template")
    plt.yscale('log')
    plt.hist(template_lengths, bins=max(template_lengths), cumulative=True, alpha=0.25)
    plt.hist(template_lengths, bins=max(template_lengths))
    plt.show()

    in_lengths = []
    out_lengths = []
    for data in user_data:
        sentence_in = data["in"]
        sentence_out = data["out"]

        in_lengths.append(len(sentence_in))
        out_lengths.append(len(sentence_out))

    plt.title("sentence in")
    plt.yscale('log')
    plt.hist(in_lengths, bins=max(in_lengths), cumulative=True, alpha=0.25)
    plt.hist(in_lengths, bins=max(in_lengths))
    plt.show()

    plt.title("sentence out")
    plt.yscale('log')
    plt.hist(out_lengths, bins=max(out_lengths), cumulative=True, alpha=0.25)
    plt.hist(out_lengths, bins=max(out_lengths))
    plt.show()

def get_single_dataset(filename, word2id, batch_size, is_train, min_occurence, shuffle, glove_vocab = None, print_freqs = False):
    """
    Returns a single dataset as dataloader object in batches.
    """
    print("Processing file {}".format(filename))
    data = read(filename, word2id, is_train, min_occurence, glove_vocab)

    # Print frequencies for data analysis.
    # TODO: fix print_counts for new version
#    if print_freqs:
#        print_counts(template_data, user_data)

    dataset = ChatsDataset(data)
    dataloader = DataLoader(dataset, batch_size, collate_fn=dataset.collate, shuffle=shuffle)
    print("-- Finished processing file {}\n".format(filename))
    return dataloader

def get_datasets(path, batch_size, min_occurence, glove_vocab = None, print_freqs = False, only_train = False):
    """
    Returns all three datasets and the word2id object.
    """
    print("==== Getting the datasets ====\n")
    word2id = Word2Id()
    train_data = get_single_dataset(path + "train-v1.1.json", word2id, batch_size, True, min_occurence, True, glove_vocab, print_freqs)
    if only_train:
      return train_data, _, _, word2id
    
    dev_data = get_single_dataset(path + "dev-v1.1.json", word2id, batch_size, False, min_occurence, False, print_freqs)
    test_data = get_single_dataset(path + "test-v1.1.json", word2id, batch_size, False, min_occurence, False, print_freqs)
    print("==== Finished getting the datasets ====\n")

    return train_data, dev_data, test_data, word2id

# Copied from https://github.com/nikitacs16/d_bi_att_flow
def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        token = token.lower()
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens

# Copied from https://github.com/nikitacs16/d_bi_att_flow
def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
