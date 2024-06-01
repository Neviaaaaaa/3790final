import pyocr
import pyocr.builders
from PIL import Image
from keras.applications import VGG16, VGG19, DenseNet201
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

base_model = VGG16(include_top=False, input_shape=(64, 64, 1))
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
output = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1,
                                   height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('path_to_train_data', target_size=(64, 64),
                                                    color_mode='grayscale', batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory('path_to_validation_data', target_size=(64, 64),
                                                        color_mode='grayscale', batch_size=32, class_mode='categorical')

history = model.fit(train_generator, epochs=25, validation_data=validation_generator)

results = model.evaluate(validation_generator)
print("Accuracy:", results[1])

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    exit(1)
tool = tools[0]

img = Image.open('path_to_grayscale_image.png')

txt = tool.image_to_string(
    img,
    lang='ind',
    builder=pyocr.builders.TextBuilder()
)

print(txt)

import pytesseract
from PIL import Image

img = Image.open('path_to_grayscale_image.png')

config = '--psm 6'  # Set Page Segmentation Mode
text = pytesseract.image_to_string(img, lang='ind', config=config)

print(text)

from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

image = cv2.imread('path_to_image', cv2.IMREAD_GRAYSCALE)

resized_image = cv2.resize(image, (64, 64))

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_array = np.expand_dims(resized_image, axis=0)
image_array = np.expand_dims(image_array, axis=3)  # Adding channel dimension

augmented_images = datagen.flow(image_array, batch_size=1)

for i in range(5):  # Generate 5 augmented images
    augmented_image = next(augmented_images)[0].astype(np.uint8)
    cv2.imshow('Augmented Image', augmented_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

import cv2
import numpy as np
import os


def compute_similarity(img1_path, img2_path):
    if not os.path.exists(img1_path):
        print(f"文件不存在: {img1_path}")
        return float('inf')  # 返回一个很大的值

    if not os.path.exists(img2_path):
        print(f"文件不存在: {img2_path}")
        return float('inf')  # 返回一个很大的值

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        print(f"无法读取图像: {img1_path}")
        return float('inf')  # 返回一个很大的值

    if img2 is None:
        print(f"无法读取图像: {img2_path}")
        return float('inf')  # 返回一个很大的值

    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))

    diff = cv2.absdiff(img1, img2)
    similarity = np.sum(diff) / (224 * 224)

    return similarity

first_image_path = '/mnt/data/file-X4l8XrihKbAWKN7nro5awfeM/image.png'
other_images_paths = [
    '/mnt/data/file-WHZNjsGB80xdZ30NtFXB21oQ/image.png',
    '/mnt/data/file-D8v7ULEs4RaPh80Sumb0c5ku/image.png',
    '/mnt/data/file-OUf1vBlHHIwMDs6voOpi3usH/image.png',
    '/mnt/data/file-bLBIH0DHLFEdVE9KJwaUHMqU/image.png',
    '/mnt/data/file-CAjch98hBOaDHHqi4P49Wr5g/image.png'
]

similarities = []

for img_path in other_images_paths:
    similarity = compute_similarity(first_image_path, img_path)
    similarities.append((img_path, similarity))

most_similar_image = min(similarities, key=lambda x: x[1])
most_similar_image
