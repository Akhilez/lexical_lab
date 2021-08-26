from os import listdir

from torch.utils.data import Dataset


class CornellMovieGenreDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        files = listdir(self.data_path)
        print(files)
        assert "movie_titles_metadata.txt" in files

    def __getitem__(self, index: int):
        pass


if __name__ == "__main__":
    path = "/Users/akhil/code/lexical_lab/companies/ginger/data/cornell"
    dataset = CornellMovieGenreDataset(path)
