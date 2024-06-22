import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array


class EvaluationModel:
    """
    Class responsible for evaluating the trained emotion classification model
    """

    def __init__(self, model_path, test_dir):
        """
        Initialize the evaluation model

        Args:
            model_path (str): Path to the trained emotion classification model
            test_dir (str): Directory containing the testing data
        """
        self.model = load_model(model_path)
        self.test_dir = test_dir

    def evaluate(self):
        """
        Evaluate the trained model using the testing data

        Generates predictions on the testing data and calculates accuracy and other metrics
        Prints the evaluation results
        """
        # Create an ImageDataGenerator for testing data
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Generate testing data
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=32,
            class_mode="categorical",
            shuffle=False
        )

        # Get the number of testing samples
        test_samples = test_generator.samples

        # Generate predictions on the testing data
        predictions = self.model.predict(test_generator, steps=test_samples // 32 + 1)

        # Get the true labels of the testing data
        true_labels = test_generator.classes

        # Calculate the accuracy
        accuracy = np.sum(np.argmax(predictions, axis=1) == true_labels) / test_samples

        # Print the evaluation results
        print(f"Test Accuracy: {accuracy:.4f}")

        # Calculate additional metrics if needed
        # ...

    def evaluate_single_image(self, image_path):
        """
        Evaluate a single image using the trained model

        Args:
            image_path (str): Path to the image file

        Returns:
            predicted_class (int): Predicted class index
            predicted_emotion (str): Predicted emotion label
        """
        # Load and preprocess the image
        img = load_img(image_path, target_size=(48, 48), color_mode="grayscale")
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction on the image
        prediction = self.model.predict(img)

        # Get the predicted class and emotion label
        predicted_class = np.argmax(prediction)
        emotions = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        predicted_emotion = emotions[predicted_class]

        return predicted_class, predicted_emotion


# Evaluate the model
model_path = "emotion_model.h5"
test_dir = "Testing"
evaluation = EvaluationModel(model_path, test_dir)
evaluation.evaluate()

# Evaluate a single image
image_path = "Testing/Neutral/Neutral-1008.jpg"
predicted_class, predicted_emotion = evaluation.evaluate_single_image(image_path)
print(f"Predicted emotion for {image_path}: {predicted_emotion}")