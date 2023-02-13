import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

width = 28
nskip = 35

def unite(list_2d):
    rev=[]
    for list_1d in list_2d:
        for item in list_1d:
           rev.append(item)

    return np.array(rev)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=[unite(X_train[i]) for i in range(len(X_train))]
X_test=[unite(X_test[i]) for i in range(len(X_test))]

X_train=X_train[::nskip]
y_train=y_train[::nskip]

x_embedded = TSNE(n_components=2).fit_transform(X_train)



fig, ax = plt.subplots()
height,width=28,28
mnist_int = np.asarray(y_train, dtype=int)

img = np.array([0]*height*width).reshape((height, width))
imagebox = OffsetImage(img, zoom=1.0)
imagebox.image.axes = ax
cmap = plt.cm.RdYlGn

sc = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=mnist_int/10.0, cmap=cmap, s=3)
annot = AnnotationBbox(imagebox, xy=(0,0), xybox=(width,width),
                    xycoords="data", boxcoords="offset points", pad=0.5,
                    arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
annot.set_visible(False)
ax.add_artist(annot)

def update_annot(ind):
    i = ind["ind"][0]
    pos = sc.get_offsets()[i]
    annot.xy = (pos[0], pos[1])

    img = X_train[i][:].reshape((width, width))
    imagebox.set_data(img)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()