import re
from os import listdir
from os.path import join
from typing import Dict
from typing import List


def preprocess_text(text):
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = re.sub(r"\[[0-9]+]", "", text)  # Removes citations of kind [1]
    return text


def dataset_generator(data_path: str):
    """
    Iterates over the files in the dataset
    """
    file_names = listdir(data_path)

    for file_name in file_names:
        # Validate filename
        if not re.match(r"j[0-9]+\.txt", file_name):
            continue

        with open(join(data_path, file_name), "r") as text_file:
            text = text_file.read()
            yield file_name, text


def get_entity_names(entity_names_path: str) -> Dict[str, List[str]]:
    """
    entity_names_path is the path to the file 'entity_names.txt' that is already provided.
    I could use additional entity names from https://www.cancer.gov/types and my own,
    but keeping it small because it seems to do the job just fine for this assignment.
    """
    with open(entity_names_path, "r") as keywords_file:
        keywords_data = keywords_file.readlines()

    category_keywords = {}
    for i in range(len(keywords_data)):
        category, keywords = keywords_data[i].split(":")
        keywords = keywords.split(",")
        keywords = [keyword.strip() for keyword in keywords]
        category_keywords[category] = keywords

    return category_keywords


if __name__ == "__main__":
    data_path = "/Users/akhil/code/lexical_lab/companies/tempus/Data"
    for file_name, text in dataset_generator(data_path):
        print(file_name)
