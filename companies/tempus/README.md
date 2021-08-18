# Tempus NLP Assignment by Akhil Devarashetti


## File structure

- `named_entity_extractor.py` contains training and output tsv generation.
- `dataset.py` contains few helper functions related to data reading.
- `experiments.ipynb` contains some code experiments to see how the data looks.
- `Data` dir contains the given data
- `output` dir contains output files and saved models.


## Getting Started
Run the following commands to install spacy and set it up.

```shell
pip install -r requirements.txt
spacy download en_core_web_lg
```

Once you're in the root dir, run `named_entity_extractor.py` 
to get the output tsv file as well as a `displaCy.html` file that visualizes the results.

## Improvements:

If you look at the `output/displaCy.html` file in a browser,
you'll find some right predictions and some wrong ones.
The wrong ones are for example:

- née Brooke: cancertype
- Covid-19: medication

As future improvements, I would do the following:

- Train transformers using `en_core_web_trf`
- Using more entity names in training data from internet.
