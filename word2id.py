from collections import defaultdict
import time


class Word2Id:
  def __init__(self):
    self.TAG_UNK = "<UNK>"
    self.TAG_BOS = "<BOS>"
    self.TAG_EOS = "<EOS>"
    self.TAG_PAD = "<PAD>"
    
    self.w2id = defaultdict(lambda:len(self.w2id))
    self.id2w = []
     
    # Initialize the default tags.
    # This part should be hardcoded since it is used within string2id.
    self.tag_id_unk = 0
    self.tag_id_bos = 1
    self.tag_id_eos = 2
    self.tag_id_pad = 3
    # This part is to ensure it is all correctly represented in the mappings.
    self.string2id([self.TAG_UNK], True) # Give UNK id 0.
    self.string2id([self.TAG_BOS], True) # Give BOS id 1.
    self.string2id([self.TAG_EOS], True) # Give EOS id 2.
    self.string2id([self.TAG_PAD], True) # Give PAD id 3.

  def frequent_words2id(self, counter, glove_embeddings, freq_threshold):
      """
      Add the frequent words that occurs in the GLoVE set to Word2Id.
      Input: the list of tuples with all the words from the training set and 
      their frequencies.
      """
      print("Start getting the words that occur more than {} times".format(freq_threshold))
      start = time.time()      
      print("Size of the training dataset: {} words".format(len(counter.keys())))
      print("Size of GLoVE: {} words".format(len(glove_embeddings)))

      # Keep the words that appear more than the threshold.
      frequent_words = [word for word, freq in counter.items() if freq > freq_threshold]
      
      keys = counter.keys()
      intersect_no_threshold = list(set(keys).intersection(glove_embeddings))
      print("Without a threshold, the number of words in the intersection is: {}".format(len(intersect_no_threshold)))
      
      # Only get the words that occur in both the dataset and the glove embeddings,
      # so that it is not needed to loop through the whole glove set.
      intersection = list(set(frequent_words).intersection(glove_embeddings))

      # Add the words to word2id.
      for word in intersection:
          self._process_single_string(word, True)

      end = time.time()
      print("-- Finished getting the {} words that occur more than {} times".format(len(intersection), freq_threshold))
      print("It took {:.2f} seconds\n".format(end-start))
          
  def datapoint2id(self, question, answer, resources):
      """
      Convert the question, answer and resources to their ids.
      Input: question and answer as strings and resources as list of strings.
      Output: question and answer as list of ids and resources as list of lists of ids.
      """
      question2id = self.string2id(question, False)
      answer2id = self.string2id(answer, False)
      resources2id = []
      for resource in resources:
          resources2id.append(self.string2id(resource, False))
      
      return question2id, answer2id, resources2id
      
  def string2id(self, data, add_new_words):
    """
    Input: data as a single word or a sentence.
    Output: the data as a list of ids and updates the w2id.
    Note that the bos and eos tags are added as well.
    """
    # Instead of data.split(), the re is used to split special characters as individual words, too.
    list_ids = [self.tag_id_bos] # Add begin of sentence tag.
    for word in data:
      wid = self._process_single_string(word, add_new_words)
      list_ids.append(wid)
    list_ids.append(self.tag_id_eos) # Add end of sentence tag.
        
    return list_ids
          
  def _process_single_string(self, word, add_new_words):
    """
    It returns an id and updates w2id. The input word will be processed as well.
    Input: a single word
    Output: a single id.
    """
    # processing word
    word = word.lower()
    if add_new_words:
      # Add the new words if not existing already
      wid = self.w2id[word]
      self._add_id2w(wid, word)
    else:
      # Do not add if not existing already. Return UNK in this case.
      if word in self.w2id:
        wid = self.w2id[word]
      else:
        wid = self.tag_id_unk
    return wid
  
  def _add_id2w(self, wid, word):
    """
    Increases the id2w with the word if it does not exist yet.
    Input: word_id and word_str.
    """
    if len(self.id2w) == wid:
      self.id2w.append(word)
  
  def id2string(self, data):
    """
    Input: data as a list of ids.
    Output: a single string.
    """
    string = ""
    for wid in data:
      # Add each word and a space.
      # Check if the word exist or not.
      if len(self.id2w) > wid:
        string += self.id2w[wid] + " "
      else:
        string += self.id2w[self.tag_id_unk] + " " 
        # self.id2w[self.tag_id_unk] seems stupid, but it is to ensure 
        # that the unk tag is consistent after formatting etc.
    
    # Remove the last space.
    string = string[:-1]
    return string
