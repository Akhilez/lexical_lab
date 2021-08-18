from typing import Dict
import spacy
from os import makedirs
from os.path import join, dirname
from typing import Tuple
from typing import List
import random
from spacy.util import minibatch
from spacy.lang.en import English
from spacy.training import Example
from spacy import displacy
from dataset import preprocess_text, get_entity_names, dataset_generator


def _get_entities_data_for_sentence(
    sentence: str, category_keywords: Dict[str, List[str]]
):
    """
    This function checks for all the words in the given sentence if they are one of the entity names.
    If such entity exists, then it's start and end indices are recorded.
    """
    data: List[Tuple[int, int, str]] = []

    for category in category_keywords:
        for keyword in category_keywords[category]:
            pieces = keyword.split(" ")
            words = sentence.split(" ")
            current_index = 0
            for word in words:
                # This is a brute force search.
                # If the first word matches, then check for the rest of the words.
                if pieces[0].lower() == word.lower():
                    end_index = current_index + len(keyword)
                    if sentence[current_index:end_index].lower() == keyword.lower():
                        data.append((current_index, end_index, category))
                current_index += len(word) + 1
    return data


def _prepare_training_data(
    data_path: str, category_keywords: Dict[str, List[str]], training_nlp
) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
    """
    The whole dataset is parsed and any sentence containing a named entity is recorded as a training example.
    Example:
    ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]})
    """
    print("Preparing training data")

    training_data: List[Tuple[str, dict]] = []

    sentencizer = English()  # just the language with no pipeline
    sentencizer.add_pipe("sentencizer")

    for _, text in dataset_generator(data_path):
        text = preprocess_text(text)
        for sentence in sentencizer(text).sents:
            data = _get_entities_data_for_sentence(sentence.text, category_keywords)
            if data:
                doc = training_nlp.make_doc(sentence.text)
                label = {"entities": data}
                example = Example.from_dict(doc, label)
                training_data.append(example)

                # print(f"\nText: {sentence.text}\nLabel: {label}")

    return training_data


def _run_predictions(nlp, data_path: str, output_path: str):
    """
    Writes into tsv file in the format of `<file_name> <entity_label> <entity_text>`
    """
    docs = []
    with open(join(output_path, "output_named_entities.tsv"), "w") as output_file:
        for file_name, text in dataset_generator(data_path):
            text = preprocess_text(text)
            doc = nlp(text)
            for entity in doc.ents:
                output_file.write(f"{file_name}\t{entity.label_}\t{entity.text}\n")

            docs.append(doc)
    displacy.serve(docs, style="ent", page=True)


def _save_model(nlp, output_path, model_name="model"):
    output_model_path = join(output_path, model_name)
    makedirs(output_model_path, exist_ok=True)
    nlp.to_disk(output_model_path)


def main():
    """
    # 0. Add custom NER pipe
    # 1. Prepare training data
    # 2. Train a model
    # 3. Save the model
    # 4. Run prediction on the dataset. (generate output tsv)
    """

    model_type = "en_core_web_lg"
    epochs = 50
    batch_size = 4

    # -----------------

    base_path = join(dirname(__file__))

    data_path = join(base_path, "Data")
    output_path = join(base_path, "output")

    # Get all the given entity names from 'entity_names.txt'
    entity_names: Dict[str, List[str]] = get_entity_names(
        join(data_path, "entity_names.txt")
    )

    # Customize the model to learn custom named entities.
    nlp = spacy.load(model_type)
    ner = nlp.get_pipe("ner")
    for category in entity_names:
        ner.add_label(category)

    # Disable pipeline components you dont need to change
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec", "transformer"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Prepare the training data
    training_data = _prepare_training_data(data_path, entity_names, nlp)

    # ------------- Training the model --------------
    print("Training the model")

    with nlp.disable_pipes(*unaffected_pipes):
        for epoch in range(epochs):
            losses = {}
            epoch_loss, n_batches = 0, 0

            # shuffling examples  before every iteration
            random.shuffle(training_data)

            # batch up the examples using spaCy's minibatch
            batches = minibatch(training_data, size=batch_size)

            for batch in batches:
                nlp.update(
                    batch,  # Batch of examples
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )

                epoch_loss += losses["ner"]
                n_batches += 1
            epoch_loss /= n_batches
            print(f"Epoch: {epoch}\tLoss: {epoch_loss}")

    _save_model(nlp, output_path, model_type)
    _run_predictions(nlp, data_path, output_path)


if __name__ == "__main__":
    main()
