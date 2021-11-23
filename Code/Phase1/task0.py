from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
import numpy as np

face_data = fetch_olivetti_faces()
for key in face_data:
    print(key)

images = face_data.images
target = face_data.target

print(images[0])
print(images.shape)
print(target)
print(target.shape)

for i in range(len(images)):
    im = Image.fromarray(np.uint8(images[i]*255), "L")
    im.save("images/image_" + str(target[i]) + "_" + str(i) + ".png")
