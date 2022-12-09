import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")
    """
    1. I iterate over the files in directories, dataPath + "/face" and dataPath + "/non-face", 
       and use cv2.imread(directory, 0) to read in the image and 
       store it as a 2D numpy array of shape (m, n) in img.
    2. I loaded img and its classification (1 for face and 0 for non-face) in a tuple and 
       appended it to dataset.
    """
    dataset = []
    for file in os.listdir(dataPath + "/face"):
      img = cv2.imread(os.path.join(dataPath + "/face", file), 0)
      dataset.append((img, 1))

    for file in os.listdir(dataPath + "/non-face"):
      img = cv2.imread(os.path.join(dataPath + "/non-face", file), 0)
      dataset.append((img, 0))
    # End your code (Part 1)

    return dataset

