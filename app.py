from flask import Flask,request,jsonify,render_template
import os
from werkzeug.middleware.proxy_fix import ProxyFix
from io import BytesIO
from fastai import *
from fastai.vision import *
from pathlib import Path
import cv2


model_food_not_food_pkl = "foodnofood1.pkl"
model_food_classify_pkl = "foodv1.pkl"
path = Path(__file__).parent
model_check_if_food = load_learner(path, model_food_not_food_pkl)
model_food_classify = load_learner(path,model_food_classify_pkl)
app = Flask(__name__)




@app.route('/')
def home():
    return render_template('index.html')


@app.route('/check',methods=['POST'])
def check():
    data = request.form.get('image','')
    img_bytes = (data.read())
    img = open_image(BytesIO(img_bytes))
    prediction = model_check_if_food.predict(img)[0]
    return jsonify({'result': str(prediction)})

@app.route('/classify',methods=['POST'])
def classify():
    data = request.files["image"]
    img_bytes = (data.read())
    img = open_image(BytesIO(img_bytes))
    prediction = model_food_classify.predict(img)[0]
    return jsonify({'result':str(prediction)})




if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0',port=port)
    
