from torch.utils.data import Dataset, DataLoader
import numpy as np

class ChatsDataset(Dataset):

    def __init__(self, data):
        question_list, answer_list, resources_list, qa_ids = self.split_data(data)
        self.question_list = question_list
        self.answer_list = answer_list
        self.resources_list = resources_list
        self.qa_ids = qa_ids

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, i):
        """
        Return current item.
        """
        question = self.question_list[i]
        answer = self.answer_list[i]
        resources = self.resources_list[i]
        qa_id = self.qa_ids[i]
        return {"question": question, "answer": answer, "resources": resources, "qa_id": qa_id}
    
    def split_data(self, data):
        """
        Split the data into lists of input, output, the corresponding template and chat ids.
        """
        question_list = []
        answer_list = []
        resources_list = []
        qa_ids = []

        for datapoint in data:
            question_list.append(datapoint["question"])
            answer_list.append(datapoint["answer"])
            qa_ids.append(datapoint["qa_id"])
            resources_list.append(datapoint["resources"])

        return question_list, answer_list, resources_list, qa_ids

    def collate(self, batch):
        """
        Returns a single batch.
        """
        question_list = []
        answer_list = []
        resources_list = []
        qa_ids = []
                
        for item in batch:
            question_list.append(item["question"])
            answer_list.append(item["answer"])
            resources_list.append(item["resource"])
            qa_ids.append(item["qa_id"])

        return question_list, answer_list, resources_list, qa_ids