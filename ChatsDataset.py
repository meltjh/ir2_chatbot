from torch.utils.data import Dataset, DataLoader

class ChatsDataset(Dataset):

    def __init__(self, templates, chats):
        in_list, out_list, template_list, chat_ids = self.split_data(chats, templates)
        self.in_list = in_list
        self.out_list = out_list
        self.template_list = template_list
        self.chat_ids = chat_ids

    def __len__(self):
        return len(self.in_list)

    def __getitem__(self, i):
        """
        Return current item.
        """
        input = self.in_list[i]
        output = self.out_list[i]
        template = self.template_list[i]
        chat_id = self.chat_ids[i]
        return {"in": input, "out": output, "template": template, "chat_id": chat_id}
    
    def split_data(self, chats, templates):
        """
        Split the data into lists of input, output, the corresponding template and chat ids.
        """
        in_list = []
        out_list = []
        template_list = []
        chat_ids = []
        for chat in chats:
            in_list.append(chat["in"])
            out_list.append(chat["out"])
            chat_id = chat["id"]
            chat_ids.append(chat_id)
            template_list.append(templates[chat_id])
        return in_list, out_list, template_list, chat_ids
    
    def collate(self, batch):
        """
        Returns the batch.
        """
        inputs = []
        outputs = []
        templates = []
        chat_ids = []
        
        for item in batch:
            inputs.append(item["in"])
            outputs.append(item["out"])
            templates.append(item["template"])
            chat_ids.append(item["chat_id"])

        return inputs, outputs, templates, chat_ids