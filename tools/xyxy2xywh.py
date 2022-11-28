from argparse import ArgumentParser
from json import dump, load
from pathlib import Path
from typing import Union


def parse_args():
    parser = ArgumentParser(
        description='Argumentos para a conversão de um dataset COCO no padrão xyxy para o padrão xywh'
    )

    parser.add_argument('caminho_anotacoes', help='Caminho do arquivo .json com as anotações')
    parser.add_argument('caminho_saida', help='Caminho do arquivo .json de saída', type=Path)

    args = parser.parse_args()
    return args


def carregar_json(caminho_json: str) -> dict:
    with open(caminho_json, 'r', encoding='utf_8') as arquivo:
        dados = load(arquivo)
    return dados


def converter_anotacoes(
        anotacoes: dict[str, list[dict[str, Union[int, str, list[int]]]]]
) -> dict[str, list[dict[str, Union[int, str, list[int]]]]]:
    for anotacao in anotacoes['annotations']:
        xmin, ymin, xmax, ymax = anotacao['bbox']
        anotacao['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
    return anotacoes


def salvar(anotacoes: dict[str, list[dict[str, Union[int, str, list[int]]]]], caminho_saida: Path) -> None:
    caminho_saida.parent.mkdir(exist_ok=True)
    with open(caminho_saida, 'w', encoding='utf_8') as arquivo:
        dump(anotacoes, arquivo, indent=4)


def main():
    args = parse_args()

    anotacoes = carregar_json(args.caminho_anotacoes)
    anotacoes_convertidas = converter_anotacoes(anotacoes)
    salvar(anotacoes_convertidas, args.caminho_saida)


if __name__ == '__main__':
    main()
