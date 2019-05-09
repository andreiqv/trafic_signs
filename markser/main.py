from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def predict(model, img):
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def plot_preds(img, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
        preds: list of predicted labels and their probabilities
    """
    labels = ("cat", "dog")
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    plt.figure(figsize=(8, 8))
    plt.subplot(gs[0])
    plt.imshow(np.asarray(img))
    plt.subplot(gs[1])
    plt.barh([0, 1], preds, alpha=0.5)
    plt.yticks([0, 1], labels)
    plt.xlabel("Probability")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

model = load_model("model2.h5")

new_size = (299, 299)

img = load_img("./cat.jpg", target_size=new_size)

preds = predict(model, img)
plot_preds(np.asarray(img), preds)
print(preds)



