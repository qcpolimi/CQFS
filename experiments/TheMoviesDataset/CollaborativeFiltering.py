from data.DataLoader import TheMoviesDatasetLoader
from experiments.train_CF import train_CF


def main():
    data_loader = TheMoviesDatasetLoader()
    train_CF(data_loader)


if __name__ == '__main__':
    main()
