#import flask
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
# opencv
import cv2


# buat instance flask
app = Flask(__name__)

camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
