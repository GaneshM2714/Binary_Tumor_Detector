from cProfile import label

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import  imutils
import numpy as np
import os

from tensorflow.python.ops.special_math_ops import lbeta

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'TumorDetector.keras')
model = load_model(model_path)


def crop_brain_contour(image):
    print(f"Image type: {type(image)}, Image shape: {image.shape if isinstance(image, np.ndarray) else 'N/A'}")

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)


    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return new_image

def preprocess_image(image):
    """
    Read images, resize and normalize them.
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    # load the image
    # crop the brain and ignore the unnecessary rest part of the image
    image = crop_brain_contour(image)
    # resize image
    image = cv2.resize(image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    # normalize values
    image = image / 255.
    # convert image to numpy array and append it to X
    X.append(image)
    # append a value of 1 to the target array if the image
    # is in the folder named 'yes', otherwise append 0.

    X = np.array(X)

    return X


def predict(image):

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make a prediction
    result = model.predict(preprocessed_image)

    # Convert prediction to a human-readable format
    output = "Yes" if result[0][0] > 0.5 else "No"
    return output


def read_image(uploaded_file):
    """
    Converts the uploaded file to an OpenCV image.

    Args:
        uploaded_file: The uploaded image file from Streamlit.

    Returns:
        image: The OpenCV-compatible image as a NumPy array.
    """
    # Convert the file to bytes and then to a NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(file_bytes, 1)  # 1 means read as a color image (BGR)

    return image

def train_model(c_model, c_label,c_image):
  """Trains the model on new data.

  Args:
      model: The TensorFlow model to train.
      new_data: A tuple containing the input features (X_train)
                and target labels (y_train) for the new data.

  Returns:
      The trained model.
  """
  # Preprocess the image
  np_img = preprocess_image(c_image)

  label_array = np.array([c_label])  # Shape should be (1,)

  c_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # Train the model on the new image and label
  history = c_model.fit(np_img, label_array, epochs=1, batch_size=1, verbose=2)

  # Save the trained model
  model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'TumorDetector.keras')
  c_model.save(model_path)

  return model

def main():
    st.title("Tumor Prediction App")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image")

    if uploaded_image is not None:
        # Read the image using the custom function
        image = read_image(uploaded_image)
        n_image = image

        # Display the uploaded image
        st.image(image, channels="BGR")

        # Make a prediction
        result = predict(image)

        # Display the result
        st.write("Prediction:", result)
        # st.image(n_image)
        st.write("FEEDBACK: ")
        no_trigger = st.checkbox("Was the Output Not Correct?")
        ans = True
        if no_trigger:
            st.write("No!")
            ans = False
        else:
            st.write("Yes!")

        if not ans:
            label = 0
            if result== "No":
                label = 1
            train_model(model, label,n_image)


if __name__ == "__main__":
    main()
