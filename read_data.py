import json

def read(filename):
    with open(filename) as file:
        data = json.load(file)
        for line in data:
            imdb_id = line["imdb_id"]
            chat = line["chat"]
            documents = line["documents"]
            break

read("data/main_data/train_data.json")