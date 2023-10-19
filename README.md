# Sign Language Gesture Recognition

**Sign Language Gesture Recognition** is a Python-based application that utilizes computer vision and machine learning to interpret American Sign Language (ASL) gestures made by users. This program captures live video feed from the webcam, allowing users to perform ASL gestures corresponding to the letters "I," "L," "O," "V," "E," "Y," and "U" (currently it is limited to these alphabets, the user can cuild a dataset for their gestures, train the model on that dataset and run this program with the new model as well). Upon recognizing the gestures, the application accurately predicts the letter and assembles the complete sentence.

## How It Works

1. **Hand Detection and Gesture Recognition:** The application uses the OpenCV library to capture live video feed from the webcam. It detects and tracks the user's hand using the `cvzone.HandTrackingModule` and recognizes specific gestures corresponding to ASL letters.

2. **Gesture Prediction:** The captured hand gesture is processed and classified using a pre-trained deep learning model. The model, loaded from `Model/keras_model.h5`, predicts the corresponding ASL letter based on the gesture.

3. **Interactive User Interface:** The program provides real-time feedback to the user, displaying a bounding box to guide hand placement. If the user places their hand correctly and performs a recognizable ASL gesture, the corresponding letter is identified and displayed on the screen.

4. **Sentence Assembly:** As the user performs multiple gestures, the recognized letters are assembled into a sentence. When the user makes a "thumbs up" gesture, indicating the completion of the sentence, the entire sentence is displayed, and the recognition process resets for a new sentence.

## Features

- **Real-time Gesture Recognition:** The application provides instant feedback on recognized gestures, allowing users to see the recognized letter in real time.

- **Sentence Building:** Users can form complete sentences by performing multiple gestures. The sentence is displayed after the "thumbs up" gesture is detected.

- **User-Friendly Interface:** The program guides users with visual cues, indicating where to place their hand for accurate gesture recognition.

## Usage

1. **Hand Gesture Recognition:** Position your hand within the specified bounding box and perform gestures corresponding to the letters "I," "L," "O," "V," "E," "Y," or "U."

2. **Sentence Formation:** Form words or sentences by performing multiple gestures. The recognized letters are displayed in real time.

3. **Sentence Display:** When you are ready to complete your sentence, make a "thumbs up" gesture. The entire sentence is displayed on the screen.

4. **Restart:** The recognition process resets after displaying the sentence, allowing you to form a new sentence.
