from collections import Counter
from json import load

from numpy import unique


def main():
    caminho_dataset = '/home/anderson/PycharmProjects/mmsegmentation/data/insetos/subimagens/10x10/treino.json'

    dataset = load(caminho_dataset)
    labels = [label['category_id'] for label in dataset['annotations']]


if __name__ == '__main__':
    main()
