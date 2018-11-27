import json
from collections import Counter
from ChatsDataset import ChatsDataset
from torch.utils.data import Dataset, DataLoader
from word2id import Word2Id
import matplotlib.pyplot as plt
import time

def read(filename, word2id, add_new_words, glove_vocab = None):
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
            context = paragraph["context"]
            qas = paragraph["qas"][0] # list only consists of one Q&A
            question = qas["question"].lower()
            answer_info = qas["answers"][0] # list only consists of one answer
            answer = answer_info["text"].lower()
            answer_start = answer_info["answer_start"]
            qa_id = qas["id"]
            
            resources = [x.lstrip().rstrip().lower() for x in context.split(".") if len(x) > 0]
            resources = [sentence + "." for sentence in resources]
                        
            all_words.extend(question.split())
            all_words.extend(answer.split())
            for sentence in resources:
                all_words.extend(sentence.split())
            
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
        most_common = counter.most_common()
        word2id.most_common2id(most_common, glove_vocab, 20000)
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

def get_single_dataset(filename, word2id, batch_size, is_train, glove_vocab = None, print_freqs = False):
    """
    Returns a single dataset as dataloader object in batches.
    """
    print("Processing file {}".format(filename))
    data = read(filename, word2id, is_train, glove_vocab)

    # Print frequencies for data analysis.
    # TODO: fix print_counts for new version
#    if print_freqs:
#        print_counts(template_data, user_data)

    dataset = ChatsDataset(data)
    dataloader = DataLoader(dataset, batch_size, collate_fn=dataset.collate)
    print("-- Finished processing file {}\n".format(filename))
    return dataloader

def get_datasets(path, batch_size, glove_vocab = None, print_freqs = False):
    """
    Returns all three datasets and the word2id object.
    """
    print("==== Getting the datasets ====\n")
    word2id = Word2Id()
    train_data = get_single_dataset(path + "train-v1.1.json", word2id, batch_size, True, glove_vocab, print_freqs)
    dev_data = get_single_dataset(path + "/dev-v1.1.json", word2id, batch_size, False, print_freqs)
    test_data = get_single_dataset(path + "/test-v1.1.json", word2id, batch_size, False, print_freqs)
    print("==== Finished getting the datasets ====\n")

    return train_data, dev_data, test_data, word2id