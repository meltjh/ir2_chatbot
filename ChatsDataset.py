from torch.utils.data import Dataset, DataLoader

class ChatsDataset(Dataset):

    def __init__(self, templates, chats, merge_type):
        in_list, out_list, template_list, chat_ids = self.split_data(chats, templates, merge_type)
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
    
    def split_data(self, chats, templates, merge_type):
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
            span = chat["span"]
            chat_templates = self.preprocess_templates(templates[chat_id], span, merge_type)
            template_list.append(chat_templates)

        return in_list, out_list, template_list, chat_ids
    
    def preprocess_templates(self, chat_templates, span, merge_type):
        """
        Create the templates based on the merge type.
        - all: simply append all the sentences from each source.
        - oracle: only append the sentences from the span's source.
        - ms: mixed-short
        """
        
        assert merge_type in ('all', 'oracle', 'ms')
        
        template_list = []
        if merge_type == "all":
            for source in chat_templates.keys():
                template = chat_templates[source]
                if len(template) > 0:
                    template_list.extend(template)

        elif merge_type == "oracle":
            for source in chat_templates.keys():
                template = chat_templates[source]
                for sentence in template:
                    if all(elem in sentence for elem in span):
                        template_list = (template)
                        return template_list
                    
        elif merge_type == "ms":
            # Get the length of the sources
            num_words = {}
            for source in chat_templates.keys():
                num_words[source] = sum([len(x) for x in chat_templates[source]])
            total_words = sum(num_words.keys())
            
            # Compute their proportions
            ratios = num_words.copy()
            for source in chat_templates.keys():
                if chat_templates[source] > 0:
                    ratios[source] /= total_words
            
            # TODO: juiste aantal woorden pakken van elke source

        return template_list
    
    def collate(self, batch):
        """
        Returns a single batch.
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