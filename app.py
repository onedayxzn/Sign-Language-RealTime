#import flask
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
# opencv
import cv2
import numpy as np
import mediapipe as mp


# buat instance flask
app = Flask(__name__)

camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')


def sign_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if success:
            frame
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except:
            pass


@app.route('/video_feed')
def video_feed():
    return Response(sign_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
