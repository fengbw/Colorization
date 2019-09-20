import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np

def image_read():
    image_string = tf.read_file('girl.jpg')
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # red * 0.21 + green * 0.72 + blue * 0.07
    gravity = np.array([0.21, 0.72, 0.07])
    with tf.Session() as sess:
        # convert Tensor to array
        image_date_rgb = image.eval()
    # convert RGB to Gray
    image_date_gray = np.dot(image_date_rgb, gravity)
    print()
    plt.imshow(image_date_gray, cmap = "gray")
    plt.show()

#image_read()
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict = unpickle("data_batch_1")
print(dict[])
