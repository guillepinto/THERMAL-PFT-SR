import argparse
import cv2
import os

from imresize import imresize


def main(args):
    # assuming root dir has only images
    images = os.listdir(args.root_dir)  

    # create output directory 
    os.makedirs(args.out_dir, exist_ok=True)

    for image_path in images:
        image = cv2.imread(os.path.join(args.root_dir, image_path), cv2.IMREAD_UNCHANGED)
        image = imresize(image, output_shape=(args.width, args.height))

        image_path = f'{os.path.splitext(image_path)[0]}.png'

        cv2.imwrite(os.path.join(args.out_dir, image_path), image)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Downscale images for SR training.')
    parser.add_argument('--root_dir', required=True, help='Path to the dataset folder.')
    parser.add_argument('--out_dir', required=True, help='Folder target path.')
    parser.add_argument('--width', type=int, default=64, help='Width of the downscaled image.')
    parser.add_argument('--height', type=int, default=64, help='Height of the downscaled image.')
    args = parser.parse_args()
    main(args)
