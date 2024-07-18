from flask import Flask, render_template, url_for
from database import data
import pickle
from forms import CustomEfficiencyForm, userPrediction, customUserPrediction
from customModel import efficieny, inhouse, truncate, generate_graph
from functions import userPredFunc
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

graph_cache = {}
@app.route("/custom",  methods=["GET", "POST"])
def custom():
    form = CustomEfficiencyForm()
    predicted = pd.DataFrame() 
    graphs = {}

    if form.validate_on_submit():
        #efficiency
        rate = form.rate.data
        eff = efficieny(rate)

        #inhouse SMH
        inhouseSMH = inhouse(eff)

        #dispatch SMH
        dispatchSMH = pickle.load(open('pkl-files/custom/dispatchPredictValues.pkl', 'rb'))

        #outsourced SMH
        model = pickle.load(open('pkl-files/custom/outsourcedModel.pkl', 'rb'))
        poly = pickle.load(open('pkl-files/custom/outsourcedPolyModel.pkl', 'rb'))
        outsourcedSMH = truncate(model.predict(poly.transform(np.array(inhouseSMH).reshape(6,1))))

        #predicted data
        predicted = pd.DataFrame({
            'financial_year' : dispatchSMH['financial_year'],
            'sales' : dispatchSMH['sales'],
            'man_power' : dispatchSMH['man_power'],
            'efficiency' : eff,
            'dispatch_smh' : dispatchSMH['dispatch_smh'],
            'inhouse_smh' : inhouseSMH,
            'outsourced_smh' : outsourcedSMH
        }).reset_index()

        if rate in graph_cache:
            graphs = graph_cache[rate]
        else:
            graphs['efficiency'] = generate_graph(dispatchSMH['financial_year'], eff, 'Efficiency over Years')
            graphs['inhouse_smh'] = generate_graph(dispatchSMH['financial_year'], inhouseSMH, 'Inhouse SMH over Years')
            graphs['outsourced_smh'] = generate_graph(dispatchSMH['financial_year'], outsourcedSMH, 'Outsourced SMH over Years')
            graphs['dispatch_smh'] = generate_graph(dispatchSMH['financial_year'], dispatchSMH['dispatch_smh'], 'Dispatch SMH over Years')

            graph_cache[rate] = graphs
     
    return render_template("custom.html", title="Custom", historic_data=data, predicted_data=predicted, form=form, graphs=graphs)

@app.route("/custom/userInputPred", methods=["GET", "POST"])
def custom_userInputPred():
    form = customUserPrediction()
    predicted = pd.DataFrame()
    if form.validate_on_submit():
        financial_year = form.financial_year.data
        parts = financial_year.split('-')
        rate = form.rate.data
        sales = form.sales.data
        booked = form.booked.data

        predicted = userPredFunc(parts = parts[1], booked = booked, sales = sales, dispatchpkl = 'custom/dispatchModel.pkl', outsourcedpkl = 'custom/outsourcedModel.pkl', dispatchPolypkl = 'custom/dispatchPolyModel.pkl',  outsourcedPolypkl = 'custom/outsourcedPolyModel.pkl',  effpkl = None, rate = rate)
    return render_template("custom_userPred.html", predicted_data=predicted, form=form)
    
@app.route("/optimized")
def optimized():
    predicted_data = pickle.load(open('pkl-files/optimized/predictedValues.pkl', 'rb'))
    return render_template("optimized.html", title="Optimized", historic_data=data, predicted_data=predicted_data)

@app.route("/optimized/userInputPred", methods=["GET", "POST"])
def optimized_userInputPred():
    form = userPrediction()
    predicted = pd.DataFrame()
    if form.validate_on_submit():
        financial_year = form.financial_year.data
        parts = financial_year.split('-')
        sales = form.sales.data
        booked = form.booked.data
        predicted = userPredFunc(parts = parts[1], booked = booked, sales = sales, dispatchpkl = 'optimized/dispatchModel.pkl', outsourcedpkl = 'optimized/outsourcedModel.pkl', dispatchPolypkl = 'optimized/dispatchPolyModel.pkl',  outsourcedPolypkl = 'optimized/outsourcedPolyModel.pkl',  effpkl = 'optimized/effModel.pkl', rate = None)
    return render_template("optimized_userPred.html", predicted_data=predicted, form=form)

@app.route("/linear")
def linear():
    predicted_data = pickle.load(open('pkl-files/linear/predictedValues.pkl', 'rb'))
    return render_template("linear.html", title="Linear", historic_data=data, predicted_data=predicted_data)

@app.route("/linear/userInputPred", methods=["GET", "POST"])
def linear_userInputPred():
    form = userPrediction()
    predicted = pd.DataFrame()
    if form.validate_on_submit():
        financial_year = form.financial_year.data
        parts = financial_year.split('-')
        sales = form.sales.data
        booked = form.booked.data
        predicted = userPredFunc(parts = parts[1], booked = booked, sales = sales, dispatchpkl = 'linear/dispatchModel.pkl', outsourcedpkl = 'linear/outsourcedModel.pkl', dispatchPolypkl = None,  outsourcedPolypkl = None,  effpkl = 'linear/effModel.pkl', rate = None)
    return render_template("linear_userPred.html", form=form, predicted_data=predicted)

@app.route("/poly")
def poly():
    predicted_data = pickle.load(open('pkl-files/polynomial/predictedValues.pkl', 'rb'))
    return render_template("poly.html", title="Poly", historic_data=data, predicted_data=predicted_data)

@app.route("/poly/userInputPred", methods=["GET", "POST"])
def poly_userInputPred():
    form = userPrediction()
    predicted = pd.DataFrame()
    if form.validate_on_submit():
        financial_year = form.financial_year.data
        parts = financial_year.split('-')
        sales = form.sales.data
        booked = form.booked.data
        predicted = userPredFunc(parts = parts[1], booked = booked, sales = sales, dispatchpkl = 'polynomial/dispatchModel.pkl', outsourcedpkl = 'polynomial/outsourcedModel.pkl', dispatchPolypkl = 'polynomial/dispatchPolyModel.pkl',  outsourcedPolypkl = 'polynomial/outsourcedPolyModel.pkl',  effpkl = 'polynomial/effModel.pkl', rate = None)
    return render_template("poly_userPred.html", form=form, predicted_data=predicted)

@app.route("/svr")
def svr():
    predicted_data = pickle.load(open('pkl-files/SVR/predictedvalues.pkl', 'rb'))
    return render_template("svr.html", title="SVR", historic_data=data, predicted_data=predicted_data)

@app.route("/svr/userInputPred", methods=["GET", "POST"])
def svr_userInputPred():
    form = userPrediction()
    predicted = pd.DataFrame()
    if form.validate_on_submit():
        financial_year = form.financial_year.data
        parts = financial_year.split('-')
        sales = form.sales.data
        booked = form.booked.data
        predicted = userPredFunc(parts = parts[1], booked = booked, sales = sales, dispatchpkl = 'SVR/dispatchModel.pkl', outsourcedpkl = 'SVR/outsourcedModel.pkl', dispatchPolypkl = None,  outsourcedPolypkl = None,  effpkl = 'SVR/effModel.pkl', rate = None)
    return render_template("svr_userPred.html", form=form, predicted_data=predicted)

@app.route("/arima")
def arima():
    predicted_data = pickle.load(open('pkl-files/ARIMA/predictedValues.pkl', 'rb'))
    return render_template("arima.html", title="ARIMA", historic_data=data, predicted_data=predicted_data)

@app.route("/arima/userInputPred", methods=["GET", "POST"])
def arima_userInputPred():
    form = userPrediction()
    predicted = pd.DataFrame()
    if form.validate_on_submit():
        financial_year = form.financial_year.data
        parts = financial_year.split('-')
        sales = form.sales.data
        booked = form.booked.data
        predicted = userPredFunc(parts = parts[1], booked = booked, sales = sales, dispatchpkl = 'ARIMA/dispatchModel.pkl', outsourcedpkl = 'ARIMA/outsourcedModel.pkl', dispatchPolypkl = None,  outsourcedPolypkl = None,  effpkl = 'ARIMA/effModel.pkl', rate = None)
    return render_template("arima_userPred.html", form=form, predicted_data=predicted)

if __name__ == "__main__":
    app.run(debug=True)

