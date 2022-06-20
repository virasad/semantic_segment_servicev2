import cv2


image = cv2.imread('/home/amir/Projects/semantic_segmentation_service/new_semantic_segmentation/dataset/pre_encoded/2021年09月16日16時30分39秒825_result.png', 0)
print(image.shape)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
