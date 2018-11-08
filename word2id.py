from collections import defaultdict
import re

class Word2Id:
    def __init__(self):
        """
        
        """
        # TODO: een begin en end tag?
        self.UNK_TAG = "UNK"
        self.w2id = defaultdict(lambda:len(self.w2id))
        self.id2w = []
        self.unk_id = self.string2id(self.UNK_TAG)[0] # Give UNK id 0.
       
    def string2id(self, data, add_new_words = True):
        """
        Input: data as a single word or a sentence.
        Output: the data as a list of ids and updates the w2id.
        """
        # Instead of data.split(), the re is used to split spcial characters as individual words, too.
        list_words = re.findall(r"[\w']+|[.,!?;()\"]", data) 
        list_ids = []
        for word in list_words:
            wid = self._process_single_string(word, add_new_words)
            list_ids.append(wid)
            
        return list_ids
            
    def _process_single_string(self, word, add_new_words):
        """
        It returns an id and updates w2id. The input word will be processed as well.
        Input: a slingle word
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
                wid = self.unk_id
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
            # Check if the word exist or not.
            if len(self.id2w) > wid:
                string += self.id2w[wid] + " "
            else:
                string += self.id2w[self.unk_id] + " " 
                # self.id2w[self.unk_id] seems stupid, but it is to ensure 
                # that the unk tag is consistent after formatting etc.
        
        # Remove the last space.
        string = string[:-1]
        return string