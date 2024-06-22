# Emotion Classification with Convolutional Neural Networks

This project aims to build an emotion classification model using Convolutional Neural Networks (CNNs) to detect and classify facial expressions into six categories: Angry, Fear, Happy, Neutral, Sad, and Surprise. The model is trained on a facial recognition dataset and achieves an accuracy of 57% on the testing data.

## Dataset
The dataset used for training and testing the emotion classification model is obtained from [Kaggle: Facial Recognition Dataset](https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset). It consists of facial images labeled with corresponding emotions.

## Project Structure
The project is organized into the following files:

- `TrainingModel.py`: Contains the `TrainingModel` class responsible for building, training, and saving the emotion classification model.
- `LiveDetection.py`: Contains the `LiveDetection` class that performs live emotion detection using the trained model and a webcam.
- `EvaluationModel.py`: Contains the `EvaluationModel` class for evaluating the trained model on testing data and single images.

## Model Architecture
The emotion classification model is built using a Convolutional Neural Network (CNN) architecture. The architecture consists of the following layers:

1. Three convolutional layers with ReLU activation, batch normalization, max pooling, and dropout regularization.
2. Flatten layer to convert the 2D feature maps into a 1D feature vector.
3. Two fully connected layers with ReLU activation and dropout regularization.
4. Output layer with softmax activation for multi-class classification.

The model is compiled with the Adam optimizer and categorical cross-entropy loss function.

## Training
The `TrainingModel` class in `TrainingModel.py` is responsible for training the emotion classification model. It applies data augmentation techniques to the training data, such as rotation, shifting, shearing, zooming, and horizontal flipping. The model is trained for 40 epochs with early stopping, model checkpointing, and learning rate reduction callbacks.

## Live Detection
The `LiveDetection` class in `LiveDetection.py` performs live emotion detection using the trained model and a webcam. It captures video frames, detects faces using a Haar cascade classifier, and predicts the emotion for each detected face using the trained model. The predicted emotions are displayed on the video frames in real-time.

## Evaluation
The `EvaluationModel` class in `EvaluationModel.py` evaluates the trained model on the testing data. It generates predictions on the testing data and calculates the accuracy. It also provides a method to evaluate a single image and predict its emotion.

## Results
The trained emotion classification model achieves an accuracy of 57% on the testing data. While this accuracy may seem low, it is important to note that emotion recognition is a challenging task due to the subjectivity and variability of facial expressions across individuals.

## Future Improvements
To improve the performance of the emotion classification model, the following approaches can be explored:

- Collecting a larger and more diverse dataset for training.
- Experimenting with different CNN architectures and hyperparameters.
- Applying advanced data augmentation techniques to generate more training samples.
- Utilizing transfer learning with pre-trained models on larger facial expression datasets.
- Incorporating contextual information and temporal dynamics for more accurate emotion recognition.

## Usage
To use the emotion classification project:

1. Install the required dependencies (OpenCV, TensorFlow, Keras).
2. Run `training_model.py` to train the emotion classification model.
3. Run `live_detection.py` to perform live emotion detection using a webcam.
4. Run `evaluation_model.py` to evaluate the trained model on testing data and single images.

