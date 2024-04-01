import tensorflow as tf
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='Path to saved model')
ap.add_argument('-d', '--test_data', required=True,
                help='Path to test data dir')
ap.add_argument('-p', '--predictions', required=True,
                help='Path to save predictions')
args = vars(ap.parse_args())

model = tf.keras.models.load_model(args['model'])

test_data = tf.keras.preprocessing.image.ImageDataGenerator(
).flow_from_directory(directory=args['test_data'],
                      target_size=(224, 224),
                      shuffle=True)

model.evaluate(test_data)

images, labels = next(test_data)

class_names = os.listdir(args['test_data'])

for i, image in enumerate(images):
    pred_prob = model.predict(tf.expand_dims(image, axis=0))
    pred_label = class_names[tf.argmax(pred_prob[0])]
    cvImg = image.astype(np.uint8)
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    cv2.putText(cvImg, pred_label, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    pred_path = os.path.sep.join([args['predictions'], f'{i}.png'])
    cv2.imwrite(pred_path, cvImg)

# RUN THIS SCRIPT LIKE THIS
# python predict.py --model C:/Users/User/Documents/Machine/Plants-Classification-Project/src/Crop_Disease_Detection_Model/Outputs/models/CDD/Model_1 --test_data C:/Users/User/Documents/Machine/Plants-Classification-Project/src/Crop_Disease_Detection_Model/Data/PlantDiseasesDataset/Apple/valid --predictions C:/Users/User/Documents/Machine/Plants-Classification-Project/src/Crop_Disease_Detection_Model/Outputs/predictions/
