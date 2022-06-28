from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# import cv2

def show_image_with_mask(image, mask):
    img = Image.open(image)
    mask = Image.open(mask)
    print('Image Size', np.asarray(img).shape)
    print('Mask Size', np.asarray(mask).shape)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.6)
    plt.title('Picture with Mask Appplied')
    plt.savefig()

if __name__ == '__main__':
    image_p = 'dataset2/images/2021年09月17日14時10分17秒391_result.tif'
    mask_p = 'dataset2/pre_encoded/2021年09月17日14時10分17秒391_result.png'
    show_image_with_mask(image_p, mask_p)
