import os
import numpy as np
import pandas as pd
import argparse
import colorsys
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


def load(im_path):
    """
    Given an image path, returns a PIL image
    """
    pil_image = Image.open(im_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)#[:, :, [2, 1, 0]]
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    # plt.imshow(img[:,:,[2,1,0]])
    # plt.axis("off")
    # plt.show()
    cv2.imshow("temp", img)
    cv2.waitKey(0)


def generate_colors(num_colors):
    color_arrays = []
    for i in range(num_colors):
        # Generate a random HSV bright color
        random_hsv = [np.random.uniform(low=0.0, high=1),
                      np.random.uniform(low=0.2, high=1),
                      np.random.uniform(low=0.9, high=1)]
        # Convert the random HSV to RGB
        random_rgb = colorsys.hsv_to_rgb(random_hsv[0],
                                         random_hsv[1],random_hsv[2])
        color_arrays.append(random_rgb)
    return color_arrays

def overlay_boxes(image, boxes, labels):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        colors = generate_colors(len(labels))

        for box, color in zip(boxes, colors):
            top_left, bottom_right = box[:2], box[2:]
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2
            )
            x = top_left[0] + 10
            y = top_left[1] + 20
            cv2.putText(
                image, labels, (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
            )

        try:
            image = image.get()
        except AttributeError:
            pass

        return image

def get_annotations(anno_file):
    """
    open Annotation .csv file and return pandas DataFrame
    """
    annot = pd.read_csv(anno_file, index_col=None)
    return annot


if __name__ == "__main__":

        parser = argparse.ArgumentParser(description="Age and Gender training dataset visualization tool")
        parser.add_argument(
             "--data-path",
            # default="./datasets",
            metavar="FILE",
            help="base path to the dataset",
        )
        parser.add_argument(
            "--annot-file",
            # default="combined_age_group_dataset.csv",
            metavar="FILE",
            help="annotation .csv file",
        )
        parser.add_argument('-n', '--n_samples', default=50, type=int, metavar='N',
                    help='number of data samples to test (default: 10)')

        args = parser.parse_args()
        root_data_dir = args.data_path
        annot_file_path = os.path.join(
            root_data_dir,
            args.annot_file
        )
        annotations = get_annotations(annot_file_path)
        sample_idx = np.random.choice(len(annotations), size=args.n_samples)
        for idx in sample_idx:
            annotation = annotations.iloc[idx]
            img_hash = annotation.image_hash
            bbox = [[annotation.x1,
                    annotation.y1,
                    annotation.x2,
                    annotation.y2]]
            gender = annotation.gender
            age = annotation.age
            category = annotation.category
            label = "/".join([category, gender, age])
            img = load(os.path.join(
                root_data_dir,
                img_hash + ".jpg"
            ))
            img = overlay_boxes(img, bbox, label)
            imshow(img)

