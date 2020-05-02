import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import warnings
warnings.filterwarnings('ignore')

os.listdir()

fold_0 = pd.read_csv('folds/fold_0.txt',
                     sep = '\t')
fold_1 = pd.read_csv('folds/fold_1.txt',
                     sep = '\t')
fold_2 = pd.read_csv('folds/fold_2.txt',
                    sep = '\t')
fold_3 = pd.read_csv('folds/fold_3.txt',
                    sep = '\t')
fold_4 = pd.read_csv('folds/fold_4.txt',
                    sep = '\t')

def updateAgeInterval(folds):
    age_intervals = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    for i in folds:
        for x in range(i.shape[0]):
            try : 
                dataPoint = i['age'][x]
                if  int(dataPoint) >= 0 and int(dataPoint) < 2:
                    i['age'][x] = '(0, 2)'
                    
                elif int(dataPoint) >= 4 and int(dataPoint) < 6  :
                    i['age'][x] = '(4, 6)'
                    
                elif int(dataPoint) >= 8 and int(dataPoint) < 12 : 
                    i['age'][x] = '(8, 12)'
                    
                elif int(dataPoint) >= 13 and int(dataPoint) <20 :
                    i['age'][x] = '(15, 20)'
                    
                elif  int(dataPoint) >= 23 and int(dataPoint) < 32:
                    i['age'][x] = '(25, 32)'
                
                elif int(dataPoint) >= 35 and int(dataPoint) < 43:
                    i['age'][x] = '(38, 43)'
                    
                elif int(dataPoint) >= 45 and int(dataPoint) < 53:
                     i['age'][x] = '(48, 53)'
                
                elif int(dataPoint) >= 60:
                    i['age'][x] = '(60, 100)'
                
                else : 
                    i['age'][x] = 'NaN'
            except : 
                if dataPoint not in age_intervals:
                    i['age'][x] = 'NaN'

def dropAgeNan(folds):
    for i in folds :
        i.drop(i[i['age'] == 'NaN'].index,inplace = True)
        
        
def dropUnnecesaryColumns(folds):
    for i in folds:
        i.drop(['x', 'y', 'dx',
       'dy', 'tilt_ang', 'fiducial_yaw_angle', 'fiducial_score'],axis = 1,inplace = True)
    
def faceidToString(folds):
    for i in folds:
        i['face_id'] = i['face_id'].astype(str)     
    
def createPathColumn(folds):
    for i in folds:
        i['path'] = 'faces/' + i['user_id']+ "/" + "coarse_tilt_aligned_face."+ i['face_id']+ '.' + i['original_image'] 
        
def addArrayImage(folds):
    tx1 = time.time()
    print('Reading images has started !!')
    images = []
    for i in folds:
        for x in i['path']:
            image = cv2.imread(x)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,(200,200)) #Resize for bounding
            image = image.astype(np.uint8)
            images.append(image)
    images = np.array(images)
    tx2 = time.time()
    print('Images read in {} seconds '.format(int(tx2-tx1)))
    return images / 255 # scaling images
    
def genderToInt(folds):
    for i in folds:
        i['gender'] = [0 if x == 'f' else 1 if x == 'm' else 2 for x in i['gender']]    
    
def createData():   
    folds = (fold_0,fold_1,fold_2,fold_3,fold_4)
    updateAgeInterval(folds)
    dropAgeNan(folds)
    dropUnnecesaryColumns(folds)
    faceidToString(folds)
    createPathColumn(folds)
    genderToInt(folds)
    images = addArrayImage(folds)
    data = pd.concat(folds,axis=0,ignore_index=True)
    return data,images
    