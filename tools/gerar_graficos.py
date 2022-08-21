from argparse import ArgumentParser
from collections import defaultdict
from json import loads
from os.path import basename, dirname

from matplotlib import pyplot as plt
from numpy import mean


def parse_args():
    parser = ArgumentParser(description='Gráficos ')

    parser.add_argument('--arquivos', help='Caminho para cada arquivo a ser lido', nargs='+')
    parser.add_argument('--modo', help='Ou train ou val')
    parser.add_argument('--atributo', help='Valor a ser plotado. Ex.: loss, aAcc')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dados_arquivos = defaultdict(list)
    for arquivo in args.arquivos:
        with open(arquivo, 'r', encoding='utf_8') as a:
            linhas = a.readlines()

        valores = defaultdict(list)
        for linha in linhas:
            dado = loads(linha)
            if dado.get('mode') == args.modo and dado.get(args.atributo):
                valores[dado['epoch']].append(dado[args.atributo])

        for epoca in valores:
            dados_arquivos[arquivo].append(mean(valores[epoca]))

        plt.plot(dados_arquivos[arquivo])
    plt.title(' '.join((args.atributo, args.modo)))
    plt.legend([basename(dirname(arquivo)) for arquivo in args.arquivos])
    plt.show()


if __name__ == '__main__':
    main()
