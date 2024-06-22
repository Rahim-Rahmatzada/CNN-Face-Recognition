import cv2
import numpy as np
from keras.models import load_model


class LiveDetection:
    """
       Class responsible for performing live emotion detection using a trained model
    """

    def __init__(self, model_path):

        """
        Initialize the face cascade classifier and load the trained emotion detection model

        Args:
            model_path (str): Path to the trained emotion detection model
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = load_model(model_path)
        self.emotions = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def detect_emotion(self, gray, frame):

        """
        Detect faces in the grayscale image and predict emotions using the loaded model

        Args:
            gray (numpy.ndarray): Grayscale image
            frame (numpy.ndarray): Original color frame

        Returns:
                frame (numpy.ndarray): Color frame with emotion labels and bounding boxes drawn on detected faces
        """

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, [1, 48, 48, 1])

            predicted_class = np.argmax(self.model.predict(face))
            emotion = self.emotions[predicted_class]

            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        return frame

    def start(self):

        """
        Start the live emotion detection process using the default camera (camera index 0)

        Capture video frames, detect emotions, and display the results in a window
        Press 'q' to quit the program
        """

        video_capture = cv2.VideoCapture(0)
        while True:
            _, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canvas = self.detect_emotion(gray, frame)
            cv2.imshow('Video', canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


model_path = "emotion_model.h5"
live_detection = LiveDetection(model_path)
live_detection.start()
