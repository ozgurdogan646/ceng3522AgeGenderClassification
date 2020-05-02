#%% import libraries
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#%%Load models
modelAge = load_model('models/ModelAgePrediction.h5')
modelGender = load_model('models/ModelGenderPrediction.h5')
#Load Encoder
le = LabelEncoder()
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
le.classes_ = np.load('ageLabelEncoder/AgeClasses.npy')
np.load = np_load_old

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):
    global face_cascade
    face_img = img.copy()
    face_rectangles = face_cascade.detectMultiScale(face_img)
    predictionAge = None
    predictionGender = None
    for (x,y,w,h) in face_rectangles:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        try:
            predictionAge,predictionGender = predictAgeGender(face_img[x:x+w,y:y+h])
        except:
            return face_img,'Bekleniyor',''
    if predictionAge and predictionGender :
        return face_img,predictionAge,predictionGender
    else :
        return face_img,' Bekleniyor',''
    
def predictAgeGender(image):
    image = cv2.resize(image,(200,200))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  
    image = image.astype(np.uint8)
    image= image.reshape(-1,200,200,1)
    predictionAge = modelAge.predict_classes(image)
    predictGender = modelGender.predict_classes(image)
    if predictGender == 0 :
        image_gender = "Female"
    elif predictGender == 1:
        image_gender = "Male"
    elif predictGender == 2:
        image_gender = "Undefined"
    image_age = le.inverse_transform(predictionAge)
    return image_age,image_gender

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame,resultAge,resultGender = detect_face(frame)
    cv2.putText(frame,str(resultAge)[1:-1]+" "+str(resultGender),
                (50,50),cv2.FONT_HERSHEY_SIMPLEX,
                1,(255, 0, 0),2)
    cv2.imshow('Video Age Recognition',frame)
    
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()