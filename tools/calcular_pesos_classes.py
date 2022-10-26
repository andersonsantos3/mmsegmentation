from argparse import ArgumentParser
from os import listdir
from os.path import join
from PIL.Image import open

from numpy import array, unique
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    parser = ArgumentParser(
        description='Script que calcula pesos para as classes de um conjunto de dados'
    )

    parser.add_argument('diretorio_anotacoes', help='Diretório que contém os arquivos com as anotações (máscaras)')
    parser.add_argument(
        'conjuntos',
        help='Conjunto(s) a ser(em) utilizado(s). Por exemplo: treino validacao teste',
        nargs='+'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    labels = list()
    for conjunto in args.conjuntos:
        diretorio_conjunto = join(args.diretorio_anotacoes, conjunto)
        imagens = [imagem for imagem in listdir(diretorio_conjunto) if imagem.endswith('.png')]
        for imagem in imagens:
            caminho_imagem = join(diretorio_conjunto, imagem)
            im = open(caminho_imagem)
            im_array = array(im)
            labels += im_array.flatten().tolist()

    class_weights = compute_class_weight(class_weight='balanced', classes=unique(labels), y=labels)
    print(class_weights)


if __name__ == '__main__':
    main()
