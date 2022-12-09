from configparser import Interpolation
import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
    1. Read in the txt file in the given dataPath
    2. For every image, store its file name and the number of faces to be detected
    3. Read in the coordinates of the boxes
    4. Use cv2 to read the image, crop the image with respect to the boxes' coordinate,
       resize the cropped image to 19x19 and then convert them to gray scale
    5. Throw those cropped images into the classifier and draw red or green rectangles 
       with respect to its output
    """
    file = open(dataPath, 'r')
    while True:
      info = file.readline().split(" ")
      if info == ['']: break

      boxes = []
      for i in range(int(info[1])):
        cdt = file.readline().split(" ")
        tmp = [int(cdt[0]), int(cdt[1]), int(cdt[2]), int(cdt[3])]
        boxes.append(tmp)

      images = []
      img = cv2.imread(os.path.join('data/detect/', info[0]))
      for i in boxes:
        cropped = img[i[1]:i[1]+i[3], i[0]:i[0]+i[2]]
        resized = cv2.resize(cropped, (19,19), interpolation=cv2.INTER_NEAREST)
        images.append(cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY))
      
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      for i in boxes:
        if clf.classify(images[boxes.index(i)]) == 1:
          cv2.rectangle(img, (i[0],i[1]), (i[0]+i[2],i[1]+i[3]), (0,255,0), 3)
        else:
          cv2.rectangle(img, (i[0],i[1]), (i[0]+i[2],i[1]+i[3]), (255,0,0), 3)

      plt.imshow(img)
      plt.show()

    file.close()
    # End your code (Part 4)

