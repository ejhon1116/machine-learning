from tensorflow.keras import models
from kaggle_data.paths import *
from neural_network.train_sign_lang import WarmupCosineDecay
import pandas
import os
model_path = os.path.join('neural_network', 'sign_lang_models', 'max_accuracy.model.keras')


if __name__ == '__main__':
    model = models.load_model(model_path, custom_objects={'WarmupCosineDecay': WarmupCosineDecay})

    data = pandas.read_csv(os.path.join(sign_lang_path, "sign_mnist_test.csv")).values
    test_labels = data[:, 0]
    test_images = (data[:, 1:] / 255.0).reshape(7172, 28, 28)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)