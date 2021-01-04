# Blur and anonymize Faces using OpenCV and DNN
This project uses OpenCV's Face Detection Neural Network to recognize faces and anonymize them by blurring faces of each individual.

### Description
 This project follows 4 steps:
  1. We use the webcam through OpenCV to obtain real time video.
  2. We detect faces of each individual using the OpenCV's Face detection Deep Neural Network.
    [NOTE]I have also added a program(main_HC.py) to do the same using a Haar Cascade Classifier. I found out that there is a 
    difference in face detection accuracy while testing both of them in real time.I have added both the files.You can experiment with both of them.
  3. We have used two different methods to blur the faces. For simple blur, we have used a gaussian blur, and for pixelated blur,
  we have divided the roi(face) into blocks, and blurred each of them using mean BGR values.
  4. After blurring the roi(face), we replace the roi area of the actual webcam feed with the blurred image, and display it.
 
## Setup
### pip3
1. Open the terminal.
2. Clone the repository to your local machine.
3. Navigate inside the folder.
4. Install all dependencies using `pipenv install -r requirements.txt` or `pip install -r requirements.txt`

## Usage
1. Run the main_DNN.py for detecting and blurring faces using OpenCV's Deep Neural Network.
2. Run the main_HC.py for detecting and blurring faces using HaarCascade Classifier.
