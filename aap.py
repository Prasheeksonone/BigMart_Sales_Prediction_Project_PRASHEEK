from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")
@app.route('/predict',method=['POST','GET'])
def result():
    Item_weight=float(request.form['item_weight'])
    Item_fat_content = float(request.form['item_fat_content'])
    Item_visibility = float(request.form['item_visibility'])
    Item_type = float(request.form['item_type'])
    Item_map= float(request.form['item_map'])
    outlet_establishment_year=float(request.form['outlet_establishment_year	'])
    outlet_size=float(request.form[' outlet_size'])
    outlet_location = float(request.form[' outlet_location'])
    outlet_type = float(request.form[' outlet_type'])

    x=np.array([[Item_weight,Item_fat_content,Item_visibility,Item_type,Item_map,
                 outlet_establishment_year,outlet_size,outlet_location,outlet_type]])

    scaler_path=os.path.join(r'C:\BigMart-Sales-Prediction Project_PRASHEEK\models_done','sc.sav')
    sc=joblib.load(scaler_path)

    x_std=sc.transform(x)

    model_path=r'C:\BigMart-Sales-Prediction Project_PRASHEEK\models_done\lr.sav'

    model=joblib.load(model_path)

    y_pred=model.predict(x_std)

    return jsonify ({'prediction':float(y_pred)})



if __name__ == "__main__":
    app.run(debug=True, port=9457)
