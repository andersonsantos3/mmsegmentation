from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from json import dump, load as load_json
from pathlib import Path
from pickle import load as load_pickle
from typing import Optional, Union

from numpy import ndarray
from skimage.measure import label, regionprops
from skimage.morphology import closing, square


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Argumentos para gerar bounding boxes a partir de máscaras de segmentação')

    parser.add_argument('caminho_predicoes', help='Caminho para o arquivo .pkl com as predições')
    parser.add_argument('caminho_txt', help='Caminho para o arquivo .txt que possui o nome das imagens')
    parser.add_argument('caminho_saida', help='Caminho para o arquivo .json de saída', type=Path)

    parser.add_argument('--limiar_bbox', help='Área mínima (em pixels) para considerar uma bbox', type=int, default=1)

    args = parser.parse_args()
    return args


def gerar_boxes_imagens(
        nomes_imagens: list[str],
        predicoes: ndarray,
        limiar_bbox: int
) -> defaultdict[str, dict[str, list[Union[int, float, list[int]]]]]:
    """Código adaptado de: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_label.html"""

    assert len(nomes_imagens) == len(predicoes), 'Subimagens e predições devem ter o mesmo tamanho'

    imagens = defaultdict(dict)
    for nome_imagem, predicao in zip(nomes_imagens, predicoes):
        nome = nome_imagem.split('_')[0]
        if nome not in imagens:
            imagens[nome]['boxes'] = list()
            imagens[nome]['labels'] = list()
            imagens[nome]['scores'] = list()

        xmin_, ymin_, _, _ = map(int, nome_imagem.split('_')[1:])
        bw = closing(predicao > 0, square(3))
        label_image = label(bw)
        for region in regionprops(label_image):
            if region.area_bbox >= limiar_bbox:
                ymin, xmin, ymax, xmax = region.bbox
                categoria, score = obter_categoria_score(predicao, xmin, ymin, xmax, ymax)

                imagens[nome]['boxes'].append([xmin_ + xmin, ymin_ + ymin, xmin_ + xmax, ymin_ + ymax])
                imagens[nome]['labels'].append(categoria)
                imagens[nome]['scores'].append(score)
    return imagens


def obter_categoria_score(predicao, xmin, ymin, xmax, ymax) -> Optional[tuple[int, float]]:
    bbox = predicao[ymin:ymax, xmin:xmax]
    quantidade_pixels = Counter(bbox.flatten())
    del quantidade_pixels[0]
    categoria = int(quantidade_pixels.most_common(1)[0][0])
    score = quantidade_pixels[categoria] / sum(quantidade_pixels.values())
    return categoria, score


def obter_nomes_imagens_ordenados(caminho_txt: str) -> list[str]:
    with open(caminho_txt, 'r', encoding='utf_8') as arquivo:
        nomes_imagens = [linha.removesuffix('\n') for linha in arquivo.readlines()]
    nomes_imagens.sort()
    return nomes_imagens


def obter_predicoes(caminho_predicoes: str) -> ndarray:
    with open(caminho_predicoes, 'rb') as arquivo:
        predicoes = load_pickle(arquivo)
    return predicoes


def salvar_predicoes_unidas(predicoes_unidas: defaultdict, caminho_saida: Path) -> None:
    caminho_saida.parent.mkdir(exist_ok=True)
    with caminho_saida.open('w', encoding='utf_8') as arquivo:
        dump(predicoes_unidas, arquivo, indent=4)


def main():
    args = parse_args()

    predicoes = obter_predicoes(args.caminho_predicoes)
    nomes_imagens = obter_nomes_imagens_ordenados(args.caminho_txt)
    predicoes_unidas = gerar_boxes_imagens(nomes_imagens, predicoes, args.limiar_bbox)
    salvar_predicoes_unidas(predicoes_unidas, args.caminho_saida)


if __name__ == '__main__':
    main()
