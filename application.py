from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData


application = Flask(__name__)
app = application


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods = ["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else :
        data = CustomData(
            Delivery_person_Age = float(request.form.get("Delivery_person_Age")),
            Delivery_person_Ratings = float(request.form.get("Delivery_person_Ratings")),
            Weather_conditions = request.form.get("Weather_conditions"),
            Road_traffic_density = request.form.get("Road_traffic_density"),
            Vehicle_condition = int(request.form.get("Vehicle_condition")),
            Type_of_order = request.form.get("Type_of_order"),
            Type_of_vehicle = request.form.get("Type_of_vehicle"),
            multiple_deliveries = float(request.form.get("multiple_deliveries")),
            Festival = request.form.get("Festival"),
            City = request.form.get("City"),
            distance = float(request.form.get("distance"))
        )

        final_new_data = data.get_data_as_dataframe()
        predit_pipeline = PredictPipeline()
        pred = predit_pipeline.predict(final_new_data)

        #print(pred)

        results = round(pred[0],2)

        return render_template("results.html", final_result = results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True, port=5000)