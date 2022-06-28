from preimutils.segmentations.voc import Dataset, LabelMap, SegmentationAug
import os
import shutil
import glob
import random


def check_data(dataset_path, images_extention='jpg'):
    dataset = Dataset(dataset_path, images_extention)

    # To Check if the dataset2 is complete(there is a mask for each image)
    paths = dataset.check_valid_dataset()
    return paths


def augment_data(input_images_path, mask_images_path, labelmap_path, count, images_extention='jpg', resize=False, width=0, height=0):
    """
    Augmentation for each picture depending on the statistics of the objects existing in your dataset2
    Saves the augmented image in your dataset2 path with the following pattern aug_[num].jpg

    Args:
        input_images_path: Images path
        mask_images_path: Masks path
        count(int): How much of each label you want to have(counting the images you already have)
        images_extention:(optional) Your input image's extention(Default is 'jpg')
        resize:(bool : optional)-> defult False ... resize your augmented images
        width:(int : optional) width for resized ... if resize True you should set this argument
        height:(int : optional) height for resized... if resize True you should set this argument
    Returns:
        No return
    """

    augmentation = SegmentationAug(
        labelmap_path, mask_images_path, input_images_path, images_extention)
    augmentation.auto_augmentation(count, resize, width, height)


def mask_to_raw(input_images_path, mask_images_path, labelmap_path, images_extention='jpg'):
    '''
    Creates a raw mask for each mask image to use in the training process
    (H, W, 3) --> (H, W, 1)
    Each pixel's value will be the class number of it's class (0, 1, 2, ...) insted of the actual color value

    This function will create a directory 'pre_encoded' in the mask data directory, writing all raw mask images there
    '''
    labelmap = LabelMap(labelmap_path)
    augmentation = SegmentationAug(
        labelmap_path, mask_images_path, input_images_path, images_extention)

    class_colors = labelmap.color_list()
    augmentation.encode_mask_dataset(class_colors)

    print('All mask images converted to raw masks(H, W, 1) loacted in SegmentationClass/pre_encoded directory')


def train_val_test_split(dataset_path, val_size=0.2, test_size=None, shuffle=False, images_extention='jpg'):
    '''
    Creates .txt files with names of images without their extentions, seprated in train/val/test
    The .txt files are located at 'dataset2/ImageSets/Segmentation'
    The files will be as following:
    train.txt -> Training images names
    val.txt -> Validation images names
    test.txt -> Test images names (If test_size specified)
    trainval.txt -> All images names

    You can use shuffle=True to shuffle your images before splitting(Recommended)
    '''
    dataset = Dataset(dataset_path, images_extention)
    dataset.seprate_dataset(valid_persent=val_size,
                            test_persent=test_size, shuffle=shuffle)

    if test_size > 0:
        sets = ['train', 'val', 'test']
    else:
        sets = ['train', 'val']

    for data in sets:
        dir = dataset.images_dir + '/' + data
        if not os.path.exists(dir):
            os.mkdir(dir)

        dir = dataset.masks_dir + '/' + data
        if not os.path.exists(dir):
            os.mkdir(dir)

        image_names = open(os.path.join(
            dataset_path, 'ImageSets', 'Segmentation', data + '.txt'), 'r')
        names = image_names.read().splitlines()
        image_names.close()

        for name in names:
            if data == 'train':
                shutil.copy(os.path.join(dataset.images_dir, name + '.' +
                            images_extention), os.path.join(dataset.images_dir, data))
                shutil.copy(os.path.join(dataset.masks_dir, name +
                            '.png'), os.path.join(dataset.masks_dir, data))
            else:
                shutil.copy(os.path.join(dataset.images_dir, name + '.' +
                            images_extention), os.path.join(dataset.images_dir, data))
                shutil.copy(os.path.join(dataset.masks_dir, 'pre_encoded',
                            name + '.png'), os.path.join(dataset.masks_dir, data))


def preprocess(val_test_split, dataset, image_extention, prep, augment_count, resize):
    if val_test_split[0]:

        paths = check_data(
            dataset, images_extention=image_extention)

        images_dir, masks_dir, label_map_path = paths
        mask_to_raw(images_dir, masks_dir,
                    label_map_path, image_extention)

        train_val_test_split(
            dataset, val_test_split[1], val_test_split[2], shuffle=True, images_extention=image_extention)

    if prep:
        paths = check_data(
            dataset, images_extention=image_extention)
        images_dir, masks_dir, label_map_path = paths

        if augment_count > 0:
            resize, width, height = resize
            augment_data(images_dir + '/train', masks_dir + '/train', label_map_path,
                         augment_count, image_extention, resize, width, height)

        mask_to_raw(images_dir + '/train', masks_dir + '/train',
                    label_map_path, image_extention)

        total_masks_path = glob.glob(os.path.join(
            masks_dir, 'train', 'pre_encoded', '*.png'))
        total_masks_path = map(
            lambda x: os.path.basename(x[:-4]), total_masks_path)
        total_masks_path = list(total_masks_path)

        random.shuffle(total_masks_path)
        train_ds = total_masks_path[:]

        with open(os.path.join(dataset, 'ImageSets', 'Segmentation', 'train.txt'), 'w') as f:
            f.write('\n'.join(train_ds))

if __name__ == '__main__':
    images_path = '/home/amir/Projects/semantic_segmentation_service/new_semantic_segmentation/dataset/images'
    masks_path = '/home/amir/Projects/semantic_segmentation_service/new_semantic_segmentation/dataset/labels'
    label_map_path = '/home/amir/Projects/semantic_segmentation_service/new_semantic_segmentation/dataset/labelmap.txt'
    mask_to_raw(images_path, masks_path, label_map_path, 'tif')