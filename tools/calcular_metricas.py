from argparse import ArgumentParser

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = ArgumentParser(description='Argumentos para o cálculo de métricas de datasets no padrão COCO')

    parser.add_argument('caminho_anotacoes', help='Arquivo .json contendo as anotações', type=str)
    parser.add_argument('caminho_predicoes', help='Arquivo .json contendo as predições', type=str)
    parser.add_argument('caminho_saida', help='Arquivo para salvar os resultados', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    coco_gt = COCO(args.caminho_anotacoes)
    coco_dt = coco_gt.loadRes(args.caminho_predicoes)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
