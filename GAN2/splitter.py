import cv2,glob
import numpy as np
files = glob.glob ("*.png")
i=0
import os
directory="temp"
if not os.path.exists(directory):
    os.makedirs(directory)
for file in files:
    res = []
    image = cv2.imread (file,cv2.IMREAD_UNCHANGED)
    data = np.split(image, image.shape[0]/64)
    bg_color = image[0][0]
    for arr in data:
        temp=np.split(arr,arr.shape[1]/64, axis=1)
        for el in temp:
            
            cv2.imwrite("temp/"+str(i)+".png",el)
            i+=1
    
    i+=1

