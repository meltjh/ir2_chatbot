from collections import defaultdict

class Word2Id:
    def __init__(self):
        """
        
        """
        self.UNK_TAG = "UNK"
        self.w2id = defaultdict(lambda:len(self.w2id))
        self.id2w = []
        self.string2id(self.UNK_TAG) # Give UNK id 0.
       
    def string2id(self, data):
        """
        Input: data as a single word or a sentence.
        Output: the data as a list of ids and updates the w2id.
        """
        list_words = data.split()
        list_ids = []
        for word in list_words:
            id = self._process_single_string(word)
            list_ids.append(id)
            
        return list_ids
            
    def _process_single_string(self, word):
        """
        It returns an id and updates w2id. The input word will be processed as well.
        Input: a slingle word
        Output: a single id.
        """
        # processing word
        word = word.lower()
        wid = self.w2id[word]
        
        self._add_id2w(wid, word)
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
        Input: data as a list of id's.
        Output: a single string.
        """
        string = ""
        for wid in data:
            # Add each word and a space.
            string += self.id2w[wid] + " "
        
        # Remove the last space.
        string = string[:-1]
        return string
            