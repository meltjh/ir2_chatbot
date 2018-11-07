from torch.utils.data import Dataset, DataLoader

class ChatsDataset(Dataset):

    def __init__(self, templates, chats):
        in_list, out_list, template_list = self.split_data(chats, templates)
        self.in_list = in_list
        self.out_list = out_list
        self.template_list = template_list

    def __len__(self):
        return len(self.in_list)

    def __getitem__(self, i):
        """
        Return current item.
        """
        input = self.in_list[i]
        output = self.out_list[i]
        template = self.template_list[i]
        
        return {"in": input, "out": output, "template": template}
    
    def split_data(self, chats, templates):
        """
        Split the data into lists of input, output and the corresponding template.
        """
        in_list = []
        out_list = []
        template_list = []
        for chat in chats:
            in_list.append(chat["in"])
            out_list.append(chat["out"])
            template_id = chat["id"]
            template_list.append(templates[template_id])
        return in_list, out_list, template_list
    
    def collate(self, batch):
        """
        Returns the batch.
        """
        inputs = []
        outputs = []
        templates = []
        
        for item in batch:
            inputs.append(item["in"])
            outputs.append(item["out"])
            templates.append(item["template"])
        
        return inputs, outputs, templates