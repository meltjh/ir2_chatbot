import json

def read(filename):
    
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
            fact_boxoffice = facts["box_office"] if ("box_office" in facts and facts["box_office"] != "NaN") else []
            fact_awards = facts["awards"] if "awards" in facts else []
            fact_taglines = facts["taglines"] if "taglines" in facts else []
            fact_similar_movies = facts["similar_movies"] if "similar_movies" in facts else []
            
            # Formatting templates data
            templates = []
            for data in [plot, 
                         review, 
                         comments, 
                         fact_boxoffice, 
                         fact_awards, 
                         fact_taglines, 
                         fact_similar_movies]:
                if isinstance(data, list):
                    templates += data
                else:
                    templates.append(data)
            template_data[chat_id] = templates
                
            # Loop to have pairs of q (even) and a (uneven). 
            # Some chats have uneven amount of messages where the last one is then removed (hence the -1),
            # since it is supposed to be the human and thus no answer should be learned.
            for i in range(0, len(chat)-1, 2):
                sentence_in = chat[i]
                sentence_out = chat[i+1]
                data = {"id":chat_id, "in":sentence_in, "out":sentence_out}
                user_data.append(data)
           
    return template_data, user_data

template_data, user_data = read("data/main_data/test_data.json")
print("Done")

