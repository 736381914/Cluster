import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, pickle


def visulize_attention_ratio(img_path, attention_mask, ratio=1, cmap="jet", save_path=None):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵,要求是（H，W）
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    print("load image from: ", img_path)
    # load the image
    img = Image.open(img_path, mode='r')
    # 现象：如果图像位深度为24则正常的HWC，为8则读出来只有两维HW，没有C了。
    # 解决：只需要convert成RGB即可解决：
    img = img.convert('RGB')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')
    plt.savefig("img.png", dpi=300)

    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    plt.savefig(save_path, dpi=300)

if __name__ == '__main__':
    category = "grid"
    base_dir = "./Heat_Map/" + category + "/"
    for num in ["001", "003", "004", "005", "006", "007", "009", "0010"]:
        # img_name = "007.png"
        # atten_name = "007_our.png"
        # heat_name = "007_our_heat.png"

        img_name = num + ".png"
        atten_name = num + "_our.png"
        heat_name = num + "_our_heat.png"

        img_path = base_dir + img_name
        print(img_path)
        atten_path = base_dir + atten_name
        print(atten_path)
        save_path = base_dir + heat_name
        print(save_path)

        atten = cv2.imread(atten_path)
        atten = cv2.cvtColor(atten, cv2.COLOR_BGR2GRAY)
        print(atten.shape)
        visulize_attention_ratio(img_path, atten, save_path=save_path)
