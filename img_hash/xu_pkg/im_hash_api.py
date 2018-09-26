# coding=utf-8
import os

import cv2

from img_hash.xu_pkg.meipai_infer import MeipaiRetrival
from root_dir import ROOT_DIR


class ImageHash(object):
    def __init__(self):
        gpus = '0'
        model_root_path = os.path.join(ROOT_DIR, 'img_hash', 'xu_pkg', 'models')

        self.hashing_infer = MeipaiRetrival()
        self.hashing_infer.load_model(model_root_path, device_id=gpus)

    def predict(self, image_path):
        img0 = cv2.imread(image_path)
        return self.hashing_infer.predict([img0])


def main():
    img_path = os.path.join(ROOT_DIR, 'img_hash', 'xu_pkg', 'have_zi', 'test', 'test0.jpg')
    ih = ImageHash()
    res = ih.predict(img_path)
    print(res)


if __name__ == '__main__':
    main()
