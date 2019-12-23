import os
import sys
sys.path.insert(0, "age_gender")
import pandas as pd
import numpy as np
import itertools
import PIL
import PIL.Image as Image
import colorsys
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.metrics import confusion_matrix


import torch
import torchvision
import torchvision.transforms as transforms
from model import resnet
from model.MobileNetV2 import MobileNetV2

def initialize_model(model_arch="resnet101", model_path="model_best.pth.tar"):

    model = resnet.__dict__[model_arch](pretrained=False, num_gender=2, num_age_groups=3)
    # model = MobileNetV2(num_gender=2, num_age_groups=3)
    model.eval()
    model = model.cuda()
    param = torch.load(model_path)
    state_dict = {}
    for layer, val in param['state_dict'].items():
        layer = layer.replace("module.", "")
        state_dict[layer] = val

    model.load_state_dict(state_dict)

    return model

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
            x = top_left[0] + 5
            y = top_left[1] + 5
            cv2.putText(
                image, labels, (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
            )

        try:
            image = image.get()
        except AttributeError:
            pass

        return image
def draw_text(image, gender, age):
    labels = gender + "_" + age
    h, w, _ = image.shape
    # x = w // 2
    # y = h // 2
    x = 100
    y = 100
    cv2.putText(
                image, labels, (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
    )
    return image

def imshow(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":

    DEBUG = True
    model = initialize_model()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ])

    root_dir = "datasets"
    annot_file = "annotations/test.csv"


    if not os.path.isdir(os.path.join(root_dir, "accurate")):
        os.mkdir(os.path.join(root_dir, "accurate"))
    if not os.path.isdir(os.path.join(root_dir, "inaccurate")):
        os.mkdir(os.path.join(root_dir, "inaccurate"))

    annots = pd.read_csv(os.path.join(root_dir, annot_file), index_col=None)
    n_imgs = len(annots)

    gender2Idx = {"male": 0, "female": 1, "other":2}
    Idx2gender = {0: "male", 1: "female", 2:"other"}
    age2Idx = {"children": 0, "adult": 1, "toddler":2}
    Idx2Age = {0: "children", 1: "adult", 2: "toddler"}
    gender_preds = []
    age_preds = []
    images = []
    gender_gts = []
    age_gts = []


    for i in range(n_imgs):
        if i % 100 == 0:
            print("{}/{} images processed".format(i, n_imgs))

        annot = annots.iloc[i]
        image_hash = annot.image_hash
        gender = annot.gender
        age = annot.age
        human_bbox = [
            annot.x1,
            annot.y1,
            annot.x2,
            annot.y2
        ]
        image_path = os.path.join(root_dir,
                                image_hash +
                                  ".jpg"
                                )

        image = Image.open(image_path).convert("RGB")
        image_crop = image.crop(human_bbox)
        # crop human_bbox:
        # image_bbox = overlay_boxes(np.array(image_crop, dtype=np.uint8), [human_bbox], gender)
        # imshow(image_bbox)
        # continue
        # image_crop.show()

        im = transform(image).unsqueeze(0)
        im = transform(image).unsqueeze(0).cuda()
        gender_gts.append(gender2Idx[gender])
        age_gts.append(age2Idx[age])

        with torch.no_grad():
            gender_output, age_output = model(im)
            gender_pred = gender_output.argmax().cpu().item()
            age_pred = age_output.argmax().cpu().item()
            gender_preds.append(gender_pred)
            age_preds.append(age_pred)

            if DEBUG:
                image_bbox = draw_text(np.array(image_crop, dtype=np.uint8), Idx2gender[gender_pred], Idx2Age[age_pred])
                cv2.imshow("temp", image_bbox[:,:,::-1])
                cv2.waitKey(0)
                #plt.imsave(os.path.join(root_dir, "incorrect_predictions",gender, image_hash.split("/")[-1] + ".jpg"), image_bbox)



    with open("predictions.pkl", "wb") as fid:
        pickle.dump({
            "gender_preds" : gender_preds,
            "age_preds" : age_preds,
            "gender_gts" : gender_gts,
            "age_gts" : age_gts,
            "images" : images,
        }, fid)

    cnf_matrix = confusion_matrix(gender_gts, gender_preds)
    cnf_matrix = confusion_matrix(age_gts, age_preds)
    np.set_printoptions(precision=4)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list(age2Idx.keys()),
                      title='Confusion matrix, with normalization', normalize=True)
    plt.show()



