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
img = cv2.imread("neural_network/a_sign.png")
img = imutils.resize(img, width=29, height=29)
img = img[0:28, 0:28]
print(img.shape)

cv2.imshow("preview", img)

model_input = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0).reshape(1, 28, 28) 
predictions = model.predict(model_input)[0]
print(np.argmax(predictions))

# Wait for the user to press a key
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()