import cv2
import face_recognition
import pandas as pd
import numpy as np
from PIL import Image

# PATHS
ORIGINAL_IMAGES_PATH = "../Datasets/IMDBWIKI/Original/Images/"
PREPROCESSED_IMAGES_PATH = "../Datasets/IMDBWIKI/Preprocessed/Images/"
ORIGINAL_CSV_PATH = "../Datasets/IMDBWIKI/Original/CSVs/"
PREPROCESSED_CSV_PATH = "../Datasets/IMDBWIKI/Preprocessed/CSVs/"
IMDB_PATH = "imdb_crop/"
WIKI_PATH = "wiki_crop/"


# PREPROCESSING VARIABLES
nTimesToUpsample = 1 # for low quality images
margin = 0 # percentage of the original bounding square width to be added

# Read the filtered imdb and wiki CSVs and create final dataframe that will combine both
    imdb_df = pd.read_csv(ORIGINAL_CSV_PATH + "imdb.csv")
wiki_df = pd.read_csv(ORIGINAL_CSV_PATH + "wiki.csv")
preprocessed_df = pd.DataFrame(columns=["genders", "ages", "img_paths"])


def preprocessImage(img, nTimesToUpsample, margin):
    detectedFaces = face_recognition.face_locations(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                    number_of_times_to_upsample=nTimesToUpsample, model="cnn")
    # Filter images with multiple/no detected faces
    if len(detectedFaces) != 1:
        return detectedFaces, None
    # Modify face bounding rectangle
    img_h, img_w, _ = img.shape
    y1, x2, y2, x1 = detectedFaces[0]
    width = x2 - x1 - 1
    height = y2 - y1 - 1
    diff = height - width
    shift = int(margin * max(width, height))
    if (diff > 0):
        if diff % 2 == 0:  # symmetric
            top = max(y1 - shift, 0)
            bottom = min(y2 + shift, img_h - 1)
            left = max(x1 - shift - int(diff / 2), 0)
            right = min(x2 + shift + int(diff / 2), img_w - 1)
        else:
            top = max(y1 - shift, 0)
            bottom = min(y2 + shift, img_h - 1)
            left = max(x1 - shift - int((diff - 1) / 2), 0)
            right = min(x2 + shift + int((diff + 1) / 2), img_w - 1)
    elif (diff <= 0):
        if diff % 2 == 0:  # symmetric
            top = max(y1 - shift + int(diff / 2), 0)
            bottom = min(y2 + shift - int(diff / 2), img_h - 1)
            left = max(x1 - shift, 0)
            right = min(x2 + shift, img_w - 1)
        else:
            top = max(y1 - shift + int((diff - 1) / 2), 0)
            bottom = min(y2 + shift - int((diff + 1) / 2), img_h - 1)
            left = max(x1 - shift, 0)
            right = min(x2 + shift, img_w - 1)

    face = img[top:bottom, left:right, :]
    face = np.array(Image.fromarray(np.uint8(face)).resize((120, 120), Image.ANTIALIAS))

    return detectedFaces, face


# Preprocess imdb
for index, row in imdb_df.iterrows():
    # Read image and detect faces
    img = cv2.imread(ORIGINAL_IMAGES_PATH + IMDB_PATH + row["img_paths"])
    # Preprocess image
    detectedFaces, face = preprocessImage(img, nTimesToUpsample, margin)
    # Filter images with multiple/no detected faces
    if len(detectedFaces) != 1:
        continue
    # Append row to preprocessed_df
    row["img_paths"] = row["img_paths"][3:]  # remove the internal folder from the image path
    preprocessed_df = preprocessed_df.append(row)
    # Write preprocessed image to Preprocessed Dataset
    cv2.imwrite(PREPROCESSED_IMAGES_PATH + row["img_paths"], face)

    print("IMDB: ", index, end='\r')

# Preprocess wiki
for index, row in wiki_df.iterrows():
    # Read image and detect faces
    img = cv2.imread(ORIGINAL_IMAGES_PATH + WIKI_PATH + row["img_paths"])
    # Preprocess image
    detectedFaces, face = preprocessImage(img, nTimesToUpsample, margin)
    # Filter images with multiple/no detected faces
    if len(detectedFaces) != 1:
        continue
    # Append row to preprocessed_df
    row["img_paths"] = row["img_paths"][3:]  # remove the internal folder from the image path
    preprocessed_df = preprocessed_df.append(row)
    # Write preprocessed image to Preprocessed Dataset
    cv2.imwrite(PREPROCESSED_IMAGES_PATH + row["img_paths"], face)

    print("WIKI: ", index, end='\r')

preprocessed_df.to_csv(PREPROCESSED_CSV_PATH + "preprocessedDataset.csv")

# For some reason running on jupyter causes duplicates in preprocessed_df
# preprocessed_df[["genders", "ages", "img_paths"]].drop_duplicates().to_csv("./preprocessedDataset2.csv")
# duprows = testdf[testdf.duplicated(["img_paths"])]
# duprows = duprows.sort_values("img_paths")
# duprows.to_csv("./test.csv")