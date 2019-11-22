import json
import cv2
import numpy as np
import os
path_of_the_folder=(input("Give the directory to save your files:"))
a=os.chdir(path_of_the_folder)
path_of_the_file=(input("Give the path of json file : "))
with open(path_of_the_file) as json_file:
    data = json.load(json_file)
def getList(dict): 
    return dict.keys()
frames=list(getList(data))
for i in frames:
    region=data[i]['regions']
    j = 0
    while j < len(region):
        region1=region[j]
        regionx=region1['shape_attributes']['all_points_x']
        regiony=region1['shape_attributes']['all_points_y']
        mapped= zip(regionx,regiony)
        mapped = list(mapped)
        print(mapped)
        mask = np.zeros([720, 1280, 3])
        regionid=region1['region_attributes']['Category_Id']
        if regionid == "37":
            cv2.fillConvexPoly(mask, np.array([mapped]), (0, 255, 255))
        else:
            cv2.fillConvexPoly(mask, np.array([mapped]), (0, 0, 255))
        cv2.imwrite(str(i.replace('.','_'))+str(j)+'.jpg' ,mask)
        j += 1
#------------------code for bitwise----------------------------         
frames=list(data.keys())
for i in range(0,len(frames)):
    while(i < len(frames)):
        region=data[frames[i]]['regions']
        y = len(region)
        b = 0
        if(b == 0):
            y1=frames[i]+str(b)
            y1=y1.replace('.','_')
            n1 = (path_of_the_folder + str(y1)+".jpg")
            image1 = cv2.imread(n1)
            b = 1
        if(b == 1):
            y1=frames[i]+str(b)
            y1=y1.replace('.','_')
            n2 = (path_of_the_folder + str(y1)+".jpg")
            image2 = cv2.imread(n2)
            b = 1
            dest_xor = cv2.bitwise_xor(image1, image2, mask = None)
            output='output'+str(i)+str(b)+'.jpg'
            cv2.imwrite('output'+str(frames[i].replace('.','_'))+str(b)+'.jpg', dest_xor)
        b=2
        if(b>=2):
            while b <len(region):
                y1=frames[i]+str(b)
                y1=y1.replace('.','_')
                n = (path_of_the_folder + str(y1)+".jpg")
                image = cv2.imread(n)
                f = b-1
                output0='output'+str(frames[i].replace('.','_'))+str(f)+'.jpg'
                output1 = cv2.imread(output0)
                dest_xor = cv2.bitwise_xor(image, output1, mask = None)
                cv2.imwrite('output'+str(frames[i].replace('.','_'))+str(b)+'.jpg', dest_xor)
                b +=1
            else:
                break
for i in range(0,len(frames)):
    a=frames[i]
    region=data[frames[i]]['regions']
    y=len(region)-1
    a=a.replace('.','_')
    path_of_input=('output'+str(a)+str(y)+'.jpg')
    img = cv2.imread(path_of_input)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_range = np.array([20, 100, 100]) 
    upper_range = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    cv2.imwrite('road'+str(a)+str(y)+'.jpg', mask)