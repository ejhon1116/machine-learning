import cv2
import numpy as np
import imutils
from neural_network.train_sign_lang import WarmupCosineDecay
from tensorflow.keras import models
import os

# model initialization
model_path = os.path.join('neural_network', 'sign_lang_models', 'max_accuracy.model.keras')
model = models.load_model(model_path, custom_objects={'WarmupCosineDecay': WarmupCosineDecay})

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    frame = imutils.resize(frame, height=28)
    frame = frame[frame.shape[0]//2-14:frame.shape[0]//2+14, frame.shape[1]//2-14:frame.shape[1]//2+14]
else:
    rval = False

while rval:
    
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    frame = imutils.resize(frame, height=28)
    frame = frame[frame.shape[0]//2-14:frame.shape[0]//2+14, frame.shape[1]//2-14:frame.shape[1]//2+14]

    model_input = (cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) / 255.0).reshape(1, 28, 28) 
    #print(model_input)
    predictions = model.predict(model_input)[0]
    #print(predictions)
    print(np.argmax(predictions))

    key = cv2.waitKey(100)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")