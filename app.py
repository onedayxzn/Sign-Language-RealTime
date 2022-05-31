#import flask
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
# opencv
import cv2
import keras
import warnings
import time
import mediapipe as mp
warnings.simplefilter(action='ignore', category=FutureWarning)


# buat instance flask
app = Flask(__name__, template_folder='./template')  # instantiate Flask


word_dict1 = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H',
              17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}


@app.route('/')
def index():
    return render_template('index.html')


def camera_max():
    '''Returns int value of available camera devices connected to the host device'''
    camera = 0
    while True:
        if (cv2.VideoCapture(camera).grab()):
            camera = camera + 1
        else:
            cv2.destroyAllWindows()
            return(max(0, int(camera-1)))


cam_max = camera_max()
camera = cv2.VideoCapture(0)
model = keras.models.load_model("model/mobilenetV2_model2.h5")
cap = cv2.VideoCapture(cam_max, cv2.CAP_DSHOW)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)                 # Make prediction
    # COLOR CONVERSION RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def get_landmark_dist_test(results, x, y):
    global width, height
    hand_array = []
    wrist_pos = results.multi_hand_landmarks[0].landmark[0]
    for result in results.multi_hand_landmarks[0].landmark:
        hand_array.append((result.x-wrist_pos.x) * (width/x))
        hand_array.append((result.y-wrist_pos.y) * (height/y))
    return(hand_array[2:])


def start_model(frame):
    global width, height
    frame = cv2.resize(frame, (width, height))
    image, results = mediapipe_detection(frame, model)
    if results.multi_hand_landmarks[0].landmark:
        hand_array = get_landmark_dist_test(results, width, height)
        return hand_array
    else:
        return None


def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if success:
            frame = start_model(frame)
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
