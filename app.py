from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = pickle.load(open("flight_price._xgb.pkl", "rb"))

# Load the LabelEncoders
le_Airline = LabelEncoder()
le_Source = LabelEncoder()
le_Destination = LabelEncoder()
le_Additional_Info=LabelEncoder()


le_Airline.classes_ = np.load('transformation\le_Airline_classes.npy', allow_pickle=True)
le_Source.classes_ = np.load('transformation\le_Source_classes.npy', allow_pickle=True)
le_Destination.classes_ = np.load('transformation\le_Destination_classes.npy', allow_pickle=True)
le_Additional_Info.classes_= np.load('transformation\le_Additional_Info_classes.npy',allow_pickle=True)

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # Extracting features from the form
        date_dep = request.form["Dep_Time"]
        Day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Month = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").month)
        Departure_hour = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").hour)
        Departure_minutes = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").minute)

        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").hour)
        Arrival_minutes = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").minute)

        Duration_hour = abs(Arrival_hour - Departure_hour)
        Duration_minutes = abs(Arrival_minutes - Departure_minutes)
        Duration_total_mins = abs(Duration_hour + Duration_minutes)

        Total_Stops = int(request.form["stops"])
        Airline = request.form['Airline']
        Source = request.form["Source"]
        Destination = request.form["Destination"]
        Additional_Info = request.form["Additional_Info"]

        # Label encode the categorical variables
        encoded_Airline = le_Airline.transform([Airline])
        encoded_Source = le_Source.transform([Source])
        encoded_Destination = le_Destination.transform([Destination])
        encoded_Addition_Info =le_Additional_Info.transform([Additional_Info])
        

        # Make the prediction
        # Make the prediction
        prediction = model.predict([[
        Airline,
        Source,
        Destination,
        Total_Stops,
        Additional_Info,
        Day,
        Month,
        Arrival_hour,
        Arrival_minutes,
        Departure_hour,
        Departure_minutes,
        Duration_minutes,
        Duration_hour,
        Duration_total_mins,
        encoded_Airline,
        encoded_Source,
        encoded_Destination,
        encoded_Addition_Info
]])
  

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text="Your Flight price is Rs. {}".format(output))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
