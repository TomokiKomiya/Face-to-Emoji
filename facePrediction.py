from image2TrainAndTest import image2TrainAndTest
import network

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers
from chainer.datasets import tuple_dataset
from chainer import Chain, Variable, optimizers
from chainer import training
from chainer.training import extensions

def faceDetectionFromPath(path, size):
    img = Image.open(path)
    img = img.resize((size, size))
    r,g,b = img.split()
    rImgData = np.asarray(np.float32(r)/255.0)
    gImgData = np.asarray(np.float32(g)/255.0)
    bImgData = np.asarray(np.float32(b)/255.0)
    imgData = np.asarray([rImgData, gImgData, bImgData])
    print(imgData.shape)
    return imgData

def main():
    parse = argparse.ArgumentParser(description='face detection')
    parse.add_argument('--gpu','-g',type=int, default=-1,
                    help='GPU ID(negative value indicates CPU')
    parse.add_argument('--model','-m', default='GoogLeNet_output_100_4.model')
    parse.add_argument('--size', '-s', type=int, default=224)
    parse.add_argument('--testpath', '-p', default="./data/test/IMG_7945.jpg")
    args = parse.parse_args()

    if args.model == '':
        sys.stderr.write("Tom's Error occurred! ")
        sys.stderr.write("You have to designate the path to model")
        return

    outNumStr = args.model.split(".")[0].split("_")
    outnum = int(outNumStr[ len(outNumStr)-1 ])

    # ネットワークを作成
    predictor = network.MLP()
    print("今回使用する学習済みモデル : {}".format(args.model))

    chainer.serializers.load_npz(args.model, predictor, strict=False)

    ident = [""] * outnum
    for line in open("face.txt", "r"):
        dirname = line.split(",")[0]
        label = line.split(",")[1]
        ident[int(label)] = dirname

    # fetch value data to predict
    faceImgs = faceDetectionFromPath(args.testpath, args.size)

    faceImgs = faceImgs[None, ...]

    gpu_id = args.gpu
    if gpu_id >= 0:
        predictor.to_gpu(gpu_id)

    # ネットワークと同じデバイス上にデータを送る
    x = predictor.xp.asarray(faceImgs)

    # モデルのforward関数に渡す
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = predictor(faceImgs)

    # Variable形式で出てくるので中身を取り出す
    y = y.array

    # 結果をCPUに送る
    # y = to_cpu(y)

    # 予測確率の最大値のインデックスを見る
    pred_label = y.argmax(axis=1)

    print('ネットワークの予測:', ident[pred_label[0]])


if __name__ == '__main__':
    main()
