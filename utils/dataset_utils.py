from sklearn.model_selection import train_test_split
import os
from glob import glob


def split_train_test(images_path, masks_path, validation_split):
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))
    all_images_list = os.listdir(images_path)
    all_masks_list = os.listdir(masks_path)
    all_images_base_name = [os.path.splitext(os.path.basename(x))[0] for x in all_images_list]
    all_masks_base_name = [os.path.splitext(os.path.basename(x))[0] for x in all_masks_list]
    intersection_of_masks_list = intersection(all_images_base_name, all_masks_base_name)
    intersection_of_masks_list.sort()
    all_masks_list = []
    all_images_list = []
    for file in intersection_of_masks_list:
        image = list(glob(os.path.join(images_path, file + '.*')))[0]
        mask = list(glob(os.path.join(masks_path, file + '.*')))[0]
        all_masks_list.append(mask)
        all_images_list.append(image)
    # all_masks_list = all_masks_list[:min_length]
    # all_images_list = all_images_list[:min_length]

    # all_images_list = [os.path.join(images_path, x)
    #                     for x in all_images_list]
    # all_masks_list = [os.path.join(masks_path, x) for x in all_masks_list]

    train_images, test_images, train_masks, test_masks = train_test_split(all_images_list, all_masks_list,
                                                                            test_size=validation_split, shuffle=True,
                                                                            random_state=19)
    print('Train Size   : ', len(train_images))
    print('Val Size     : ', len(test_images))
    return train_images, train_masks, test_images, test_masks


        