#%%

# imports
from os import listdir
from os.path import join

#%% md

# 1. Check if these keywords exist in the data, if so, how often?

#%%

# 1.1 Get keywords


def get_keywords(path):
    with open(path, "r") as keywords_file:
        keywords_data = keywords_file.readlines()

    category_keywords = {}
    for i in range(len(keywords_data)):
        category, keywords = keywords_data[i].split(":")
        keywords = keywords.split(",")
        category_keywords[category] = keywords

    return category_keywords


entity_names_path = "/Users/akhil/code/lexical_lab/companies/tempus/entity_names.txt"
category_keywords = get_keywords(entity_names_path)

#%%

# 1.2 For each keyword, find how many times it is repeated

text_path = "/Users/akhil/code/lexical_lab/companies/tempus/Data"

keyword_counts = {
    category: {keyword: 0 for keyword in category_keywords[category]}
    for category in category_keywords
}

file_names = listdir(text_path)

for file_name in file_names:
    file_path = join(text_path, file_name)

    with open(file_path, "r") as text_file:
        text = text_file.read()


#%%
