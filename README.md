Sign Language Translator

This is a Python project that translates American Sign Language (ASL) alphabet signs into text and speech. We trained a deep learning model on the ASL Alphabet Dataset and built a small frontend to test it in real time using a webcam.
<img width="893" height="707" alt="Screenshot 2025-08-17 164720" src="https://github.com/user-attachments/assets/187dd632-6f5a-4d05-b6ab-6e43321ef3d2" />

<img width="405" height="746" alt="Screenshot 2025-08-17 165037" src="https://github.com/user-attachments/assets/5157da79-9dd2-4989-8c95-e184a5b579bf" />

Features:

Recognizes ASL alphabet letters.

Shows the current letter, recent letters, and the word being formed.

Displays confidence percentage for each prediction.

Converts detected words into speech.

Simple frontend with live camera feed.

Tech Used:

Python

TensorFlow / Keras (for training the model)

OpenCV (for camera and UI)

gTTS / pyttsx3 (for text to speech)

Dataset:

I used the ASL Alphabet Dataset from Kaggle:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

How It Works:

Train the CNN model on the dataset and save it as model.h5.

Run the Python frontend.

The webcam captures your hand gesture.

The model predicts the letter and updates the word on screen.

Confidence score and recent letters are shown.

Controls:

s → Speak the current word
r → Reset/clear the screen
q → Quit the app
q → Quit the app

How to Train the Model (Optional):

If you want to retrain the model instead of using the pre-trained model.h5:

Download dataset from Kaggle:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Extract the dataset into a folder called asl_alphabet_train/ inside the project.

Run training script:

python model_training.py

This will train the CNN on the dataset and create a new model.h5 file.

Once training is complete, run the app with:

python app.py
