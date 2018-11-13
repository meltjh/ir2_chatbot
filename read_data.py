import json
from collections import Counter
from ChatsDataset import ChatsDataset
from torch.utils.data import Dataset, DataLoader
from word2id import Word2Id
import matplotlib.pyplot as plt

def read(filename, word2id, add_new_words):
    """ 
    Returns: template_data, user_data. Both as id's
    template_data: {chat_id:[template1,...]}
    user_data: [{"id":chat_id, "in":sentence_in, "out":sentence_out}, ...] 
    """
    
    template_data = {}
    user_data = []
    
    with open(filename) as file:
        json_data = json.load(file)
        for line in json_data:
            # Extracting            
            chat = line["chat"]
            documents = line["documents"]  
            chat_id = line["chat_id"]
            plot = documents["plot"]
            review = documents["review"]
            comments = documents["comments"]            
            facts = documents["fact_table"]
            fact_boxoffice = facts["box_office"] if ("box_office" in facts and str(facts["box_office"]).lower() != "nan") else []
            fact_awards = facts["awards"] if "awards" in facts else []
            fact_taglines = facts["taglines"] if "taglines" in facts else []
            fact_similar_movies = facts["similar_movies"] if "similar_movies" in facts else []
            
            # Split the sentences into a list of sentences splitted by the .
            plot = [x.lstrip().rstrip() for x in plot.split(".")]
            review = [x.lstrip().rstrip() for x in review.split(".")]
            
            # Formatting templates data
            templates = []
            for data in [plot, 
                         review, 
                         comments, 
                         fact_boxoffice, 
                         fact_awards, 
                         fact_taglines, 
                         fact_similar_movies]:
                if not isinstance(data, list):
                    data = [data]
                for sentence in data:
                    # Obtain ids and add them to the template.
                    if len(sentence) > 0:
                        templates.append(word2id.string2id(sentence, add_new_words))
            template_data[chat_id] = templates
                
            # Loop to have pairs of q (even) and a (uneven). 
            # Some chats have uneven amount of messages where the last one is then removed (hence the -1),
            # since it is supposed to be the human and thus no answer should be learned.
            for i in range(0, len(chat)-1, 2):
                # Obtain the ids and add them to the template.
                sentence_in = word2id.string2id(chat[i])
                sentence_out = word2id.string2id(chat[i+1])
                data = {"id":chat_id, "in":sentence_in, "out":sentence_out}
                user_data.append(data)
           
    return template_data, user_data

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

def get_single_dataset(filename, word2id, batch_size, is_train, print_freqs = False):
    """
    Returns a single dataset as dataloader object in batches.
    """
    template_data, user_data = read(filename, word2id, is_train)

    # Print frequencies for data analysis.
    if print_freqs:
        print_counts(template_data, user_data)

    dataset = ChatsDataset(template_data, user_data)
    dataloader = DataLoader(dataset, batch_size, collate_fn=dataset.collate)
    return dataloader

def get_datasets(path, batch_size, print_freqs = False):
    """
    Returns all three datasets and the word2id object.
    """
    word2id = Word2Id()
    train_data = get_single_dataset(path + "/train_data.json", word2id, batch_size, True, print_freqs)
#    dev_data = get_single_dataset(path + "/dev_data.json", word2id, batch_size, False, print_freqs)
#    test_data = get_single_dataset(path + "/test_data.json", word2id, batch_size, False, print_freqs)
    
    return train_data , word2id #, dev_data, test_data, word2id
    
    
#train_data, dev_data, test_data, word2id = get_datasets("data/main_data", 5, False)
