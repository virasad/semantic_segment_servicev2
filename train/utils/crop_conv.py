import cv2
import os
from tqdm import tqdm


def crop_conv(image_name, mask_name, size, strides, dst_image_dir, dst_mask_dir):
    image = cv2.imread(image_name)
    mask = cv2.imread(mask_name)
    name = os.path.basename(image_name)
    name = name.split('.')[0]
    f_counter = 0
    xKernShape = size[0]
    yKernShape = size[1]

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        out = image[x: x + xKernShape, y: y + yKernShape]
                        mask_out = mask[x: x + xKernShape, y: y + yKernShape]
                        if (out.shape[0] == size[0] and out.shape[1] == size[1]):
                            cv2.imwrite(os.path.join(dst_image_dir, f"{name}_{f_counter}.png"), out)
                            cv2.imwrite(os.path.join(dst_mask_dir, f"{name}_{f_counter}.png"), mask_out)
                            f_counter += 1

                except Exception as e:
                    print(e)
                    break

if __name__ == '__main__':
    from pathlib import Path
    # data_dir = '/home/sfm3/Amir/semantic_segmentation_service/train/new_semantic_segmentation/dataset2/data_dataset_voc/images'
    # dst_dir = '/home/sfm3/Amir/semantic_segmentation_service/train/new_semantic_segmentation/dataset2/data_dataset_voc/crop_image'
    dataset_dir = '/home/sfm3/Amir/semantic_segmentation_service/train/new_semantic_segmentation/dataset2/data_dataset_voc'
    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'masks')
    image_list = os.listdir(images_dir)
    size = (1024, 1024)
    for image_p in tqdm(image_list):
        mask_p = os.path.splitext(image_p.replace('images', 'masks'))[0] + '.png'
        
        if os.path.exists(os.path.join(masks_dir, mask_p)):
            crop_conv(os.path.join(images_dir, image_p), os.path.join(masks_dir, mask_p), size, strides=1024,
                        dst_image_dir=os.path.join(dataset_dir , 'crop_image'), dst_mask_dir=os.path.join(dataset_dir , 'crop_mask'))
        else:
            print(f'{mask_p} not found')