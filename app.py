from flask import Flask,request,jsonify,render_template
import os
from werkzeug.middleware.proxy_fix import ProxyFix
from io import BytesIO
from fastai import *
from fastai.vision import *
from pathlib import Path
import cv2


model_food_not_food_pkl = "foodnofood1.pkl"
path = Path(__file__).parent
model_check_if_food = load_learner(path, model_food_not_food_pkl)
app = Flask(__name__)




@app.route('/')
def home():
    return jsonify("ping!")


@app.route('/check',methods=['POST'])
def check():
    data = request.files["image"]
    img_bytes = (data.read())
    img = open_image(BytesIO(img_bytes))
    prediction = model_check_if_food.predict(img)[0]
    return jsonify({'result': str(prediction)})

@app.after_request
def add_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    return response


if __name__ == "__main__":
    app.run(threaded=True)