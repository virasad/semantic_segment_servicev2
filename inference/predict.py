import os
from datetime import datetime

import albumentations as A
import cv2
# import matplotlib
import numpy as np
import torch
from preimutils.segmentations.voc.utils import utils as preutils

from model import model
from utils import postprocess

# matplotlib.use('TkAgg')


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing_predict(preprocessing_fn, h, w):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.augmentations.geometric.resize.Resize(h, w),
        A.Lambda(image=to_tensor),
    ]
    return A.Compose(_transform)


# from flash.image
class InferenceSeg:
    def __init__(self, encoder, encoder_weights, h=1024, w=1024, model_path=None, num_classes=100):
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print('Using Device: ' + str(self.device))
        self.model_path = model_path
        self.preprocessing_fn = model.prep(encoder, encoder_weights)
        self.preprocessing = get_preprocessing_predict(self.preprocessing_fn, h, w)
        self.colors = self.generate_random_colors(num_classes)

    def set_model(self, model_path):
        self.best_model = torch.load(model_path, map_location=self.device)
        self.best_model.to(self.device)
        self.best_model.eval()

    def set_classes(self, n_classes):
        self.colors = self.generate_random_colors(n_classes)

    def set_size(self, h, w):
        self.preprocessing = get_preprocessing_predict(self.preprocessing_fn, h, w)

    def predict(self, image):
        x_tensor = torch.from_numpy(image['image']).to(
            self.device).unsqueeze(0)
        pr_mask = self.best_model.predict(x_tensor)
        return pr_mask.squeeze().cpu()

    def predict_data(self, image_p, return_image=False, return_coco=False, save=False, output_dir=None):
        """
        Predict the segmentation of an image using the model.
        :param image_p: cv2 image or path to image
        :param return_image: if True, return the image
        :param output_dir: directory to save the image
        :return: cv2 mask of the segmentation
        """
        if isinstance(image_p, str):
            # check if the image is a path
            if not os.path.isfile(image_p):
                raise ValueError('Image path does not exist')
            image = cv2.imread(image_p)
        elif isinstance(image_p, np.ndarray):
            image = image_p
        else:
            raise ValueError('Image must be a path or numpy array')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_prep = self.preprocessing(image=image)
        mask = self.predict(image_prep)
        labels = torch.argmax(mask, dim=-3)  # HxW

        if return_image:
            labels = labels.cpu().numpy()
            rgb_label = preutils.decode_segmap(labels, self.colors, False)
            alpha = 0.7
            cv2.resize(image, (rgb_label.shape[1], rgb_label.shape[0]))
            overlay = cv2.addWeighted(image, alpha, rgb_label, 1 - alpha, 0, rgb_label)
            _, encoded_img = cv2.imencode('.JPG', overlay)
            # save_dst = os.path.join(output_dir, 'result_{}.png'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
            # encoded_img_b64 = base64.b64encode(encoded_img)
            # cv2.imwrite(save_dst, overlay)
            return encoded_img
        elif return_coco:
            labels = labels.cpu().numpy()
            res = postprocess.get_segmentation_dict(labels, self.colors)
            print(res)
            return res


        elif save:
            labels = labels.cpu().numpy()
            rgb_label = preutils.decode_segmap(labels, self.colors, False)
            alpha = 0.7
            cv2.resize(image, (rgb_label.shape[1], rgb_label.shape[0]))
            overlay = cv2.addWeighted(image, alpha, rgb_label, 1 - alpha, 0, rgb_label)
            # _, encoded_img = cv2.imencode('.JPG', overlay)
            save_dst = os.path.join(output_dir, 'result_{}.png'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
            # encoded_img_b64 = base64.b64encode(encoded_img)
            cv2.imwrite(save_dst, overlay)
        else:
            labels = labels.cpu().numpy()
            return labels

    @staticmethod
    def generate_random_colors(n):
        colors = []
        colors.append((0, 0, 0))
        for i in range(1, n):
            colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        return colors


if __name__ == '__main__':
    import click
    import glob


    @click.command()
    @click.option('--images_dir', default='/home/james/Documents/data/images', help='Path to images directory')
    @click.option('--model_path', default='/home/james/Documents/data/best_model.pth', help='Path to model')
    @click.option('--size', type=(int, int),help='Size of image w h')
    @click.option('--save', default=False, is_flag=True, help='Save output')
    @click.option('--output_dir', default='/home/james/Documents/data/output', help='Path to output directory')
    @click.option('--num_classes', default=19, help='Number of classes')
    def main(images_dir, model_path, size, save, output_dir, num_classes):
        inference = InferenceSeg('mobilenet_v2', 'imagenet', size[0], size[1], model_path, num_classes)
        inference.set_model(model_path)
        for image_path in glob.glob(os.path.join(images_dir, '*')):
            inference.predict_data(image_path, save=save, output_dir=output_dir)


    main()
