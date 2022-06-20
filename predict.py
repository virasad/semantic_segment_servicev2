import cv2
import albumentations as A
import torch
import os
from model import model
import matplotlib
from preimutils.segmentations.voc.utils import utils


matplotlib.use('TkAgg')
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing_predict(preprocessing_fn, SIZE):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.augmentations.geometric.resize.Resize(SIZE[0], SIZE[1]),
        A.Lambda(image=to_tensor),
    ]
    return A.Compose(_transform)


# from flash.image
class InferenceSeg:
    def __init__(self, encoder, encoder_weights, size, model_path=None):
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print('Using Device: ' + str(self.device))
        self.model_path = model_path
        self.size = size
        preprocessing_fn = model.prep(encoder, encoder_weights)
        self.preprocessing = get_preprocessing_predict(preprocessing_fn, size)
        if model_path:
            self.best_model = torch.load(model_path, map_location=self.device)
        else:
            self.best_model = torch.load(
                './best_model.pth', map_location=self.device)
        self.best_model.to(self.device)

    def predict(self, image):
        x_tensor = torch.from_numpy(image['image']).to(
            self.device).unsqueeze(0)
        pr_mask = self.best_model.predict(x_tensor)
        return pr_mask.squeeze().cpu()

    def predict_data(self, image_path, save= False, output_dir=None):
        print('isss',type(self.size[0]))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_prep = self.preprocessing(image=image)
        mask = self.predict(image_prep)
        labels = torch.argmax(mask, dim=-3)  # HxW
        labels = labels.cpu().numpy()
        rgb_label = utils.decode_segmap(labels, [(0, 0, 0),(255, 0, 0), (0, 0, 255)], False)
        # print(np.where(labels == 1))
        # mask_im = color.label2rgb(labels, image, colors=[(
        #     255, 0, 0), (0, 0, 255)], bg_label=0, bg_color=(0, 0, 0))

        # plt.show()
        if save:
            save_dst = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
            alpha = 0.7
            overlay = cv2.addWeighted(image, alpha, rgb_label, 1 - alpha, 0, rgb_label)
            cv2.imwrite(save_dst, overlay)
        return labels
        


if __name__ == '__main__':
    import click
    import glob

    @click.command()
    @click.option('--images_dir', default='/home/james/Documents/data/images', help='Path to images directory')
    @click.option('--model_path', default='/home/james/Documents/data/best_model.pth', help='Path to model')
    @click.option('--size', type=(int, int), help='Size of image w h')
    @click.option('--save', default=False, is_flag=True, help='Save output')
    @click.option('--output_dir', default='/home/james/Documents/data/output', help='Path to output directory')
    def main(images_dir, model_path, size, save, output_dir):
        inference = InferenceSeg('mobilenet_v2', 'imagenet', size, model_path)
        for image_path in glob.glob(images_dir + '/*'):
            inference.predict_data(image_path, save, output_dir )

    main()
