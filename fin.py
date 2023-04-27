from inspect import ArgInfo
from sys import displayhook
from tkinter.ttk import Style

from turtle import color, left
from fastapi import BackgroundTasks, FastAPI, Form
from fastapi.responses import HTMLResponse
import numpy as np
import xgboost as xgb
import joblib

app = FastAPI()
inputs = []

@app.get("/")
async def form_get():
    return HTMLResponse(
        """
        <html>
        <head>
         <style>
    body {
        background-color: red;
    }
    form {
        margin: 0 auto;
        width: 50%;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
    }
    label {
        display: inline-block;
        width: 200px;
        text-align: right;
    }
    input[type="submit"] {
        margin-left: 200px;
    }
</style>
            <title>Life expectancy prediction</title>
            <style>
                label {
                    display: inline-block;
                    width: 100px;
                    text-align: right;
                }
                input[type="submit"] {
                    margin-left: 100px;
                }
            </style>
            <title>Life expectancy prediction</title>
            <style>
                label {
                    display: inline-block;
                    width: 100px;
                    text-align: right;
                }
                input[type="submit"] {
                    margin-left: 100px;
                }
            </style>
        </head>
        <body>
            <h1>Life expectancy prediction</h1>
            <form method="post">
                <label for="status">Status 0 for developing and 1 for developed:</label>
<input type="number" 
id="status" name="status"value="0"><br><br>


<label for="adult_mortality">Adult Mortality:</label>
<input type="number" id="adult_mortality" name="adult_mortality"value="0"><br><br>

<label for="alcohol">Alcohol:</label>
<input type="number" id="alcohol" name="alcohol"value="0"><br><br>

<label for="percentage_expenditure">Percentage Expenditure:</label>
<input type="number" id="percentage_expenditure" name="percentage_expenditure"value="0"><br><br>

<label for="hepatitis_b">Hepatitis B:</label>
<input type="number" id="hepatitis_b" name="hepatitis_b"value="0"><br><br>

<label for="bmi">BMI:</label>
<input type="number" id="bmi" name="bmi"value="0"><br><br>

<label for="under_five_deaths">Under-Five Deaths:</label>
<input type="number" id="under_five_deaths" name="under_five_deaths"value="0"><br><br>

<label for="polio">Polio:</label>
<input type="number" id="polio" name="polio"value="0"><br><br>

<label for="total_expenditure">Total Expenditure:</label>
<input type="number" id="total_expenditure" name="total_expenditure"value="0"><br><br>

<label for="diphtheria">Diphtheria:</label>
<input type="number" id="diphtheria" name="diphtheria"value="0"><br><br>

<label for="hiv_aids">HIV/AIDS:</label>
<input type="number" id="hiv_aids" name="hiv_aids"value="0"><br><br>

<label for="gdp">GDP:</label>
<input type="number" id="gdp" name="gdp"value="0"><br><br>

<label for="thinness_1_19_years">Thinness 1-19 years:</label>
<input type="number" id="thinness_1_19_years" name="thinness_1_19_years"value="0"><br><br>

<label for="thinness_5_9_years">Thinness 5-9 years:</label>
<input type="number" id="thinness_5_9_years" name="thinness_5_9_years"value="0"><br><br>

<label for="income_composition">Income Composition of Resources:</label>
<input type="number" id="income_composition" name="income_composition"value="0"><br><br>

<label for="schooling">Schooling:</label>
<input type="number" id="schooling" name="schooling"value="0"><br><br>
<p>select "rmf" for RandomForestRegressor "dtr" for DecisionTreeRegressor "xgb" for xgboost</p>

                <select id="operation" name="operation">
                    <option value="rfr">Rfr</option>
                    <option value="dtr">Dtr</option>
                    <option value="xgb">Xgb</option>
                </select><br><br>
                <input type="submit" value="Predict">
            </form>
        </body>
        </html>
        """
    )

@app.post("/")
async def form_post(status: int = Form(0),adult_mortality: int = Form(0),alcohol: int = Form(0),percentage_expenditure: int = Form(0),hepatitis_b: int = Form(0),bmi: int = Form(0),under_five_deaths: int = Form(0),polio: int = Form(0),total_expenditure: int = Form(0),diphtheria: int = Form(0),hiv_aids: int = Form(0),gdp: int = Form(0),thinness_1_19_years: int = Form(0),thinness_5_9_years: int = Form(0),income_composition: int = Form(0),schooling: int = Form(0), operation: str = Form(...)):
    inputs=[]
    inputs.append([status,adult_mortality,alcohol,percentage_expenditure,hepatitis_b,bmi,under_five_deaths,polio,total_expenditure,diphtheria,hiv_aids,gdp,thinness_1_19_years,thinness_5_9_years,income_composition,schooling])
    new_data=inputs
    new_data = np.array(new_data)
    new_data = new_data.reshape(1,-1)
    if operation == "rfr":
        model = joblib.load('RandomForestRegressor.pkl')
        x=model.predict(new_data)
        result = x
        r="RandomForestRegressor"
    elif operation == "dtr":
        model = joblib.load('DecisionTreeRegressor.pkl')
        y=model.predict(new_data)
        result = y
        r="DecisionTreeRegressor"
    elif operation == "xgb":
        model = joblib.load('xbboost.pkl')
        y=model.predict(new_data)
        result = y
        r="xgboost"
        
    else:
        result = None

    return HTMLResponse(
        f"""
        <html>
        <head>
            <title>predictions</title>
            
            <style>
            
                label {{
                    display: inline-block;
                    width: 100px;
                    text-align: right;
                }}
                input[type="submit"] {{
                    margin-left: 100px;
                }}
            </style>
            
        </head>
         
        <body>
        
        
            <h1>Predictions</h1>
            <form method="post">
                <select id="operation" name="operation">
                    <option value="rfr"{' selected="selected"' if operation == 'rfr' else ''}>Rfr</option>
                    <option value="dfr"{' selected="selected"' if operation == 'dfr' else ''}>Dfr</option>
                    <option value="xgb"{' selected="selected"' if operation == 'xgb' else ''}>Xgb</option>
                </select><br><br>
                <input type="submit" value="Predict">
            </form>
            <br>
            <p>predicted life expectancy: {result}</p>
            <p>Inputs: {inputs},model used: {r}</p>
        </body>
        </html>
        """
    )
