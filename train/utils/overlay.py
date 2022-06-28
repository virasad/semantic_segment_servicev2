import cv2
import click


@click.command()
@click.option('--image-path')
@click.option('--mask-path')
def main(image_path, mask_path, w, h):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    image = cv2.resize(image, (w, h))
    mask = cv2.resize(mask, (w, h))
    alpha = 0.7
    overlay = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0, mask)
    cv2.imwrite('overlayed.png', overlay)


if __name__ == '__main__':
    main()
