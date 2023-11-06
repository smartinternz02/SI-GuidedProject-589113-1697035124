from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import pandas as pd

df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
app = Flask(__name__, template_folder='templates')

model = pickle.load(open('Diabetes_Predict4.pickle','rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route("/input", methods=['GET', 'POST']) 
def predictions():
    if request.method == 'POST': 
        Sex = float(request.form['Sex'])
        HighBP = float(request.form['HighBP'])
        HighChol = float(request.form['HighChol'])
        CholCheck = float(request.form['CholCheck'])
        Fruits = float(request.form['Fruits'])
        Veggies = float(request.form['Veggies'])
        Age = float(request.form['Age'])
        BMI = float(request.form['BMI'])
        Smoker = float(request.form['Smoker'])
        Stroke = float(request.form['Stroke'])
        HvyAlcoholConsump = float(request.form['HvyAlcoholConsump'])
        AnyHealthcare = float(request.form['AnyHealthcare'])
        DiffWalk = float(request.form['DiffWalk'])
        HeartDiseaseorAttack = float(request.form['HeartDiseaseorAttack'])
        PhysActivity = float(request.form['PhysActivity'])
        NoDocbcCost = float(request.form['NoDocbcCost'])
        GenHlth = float(request.form['GenHlth'])
        MentHlth = float(request.form['MentHlth'])
        PhysHlth = float(request.form['PhysHlth'])
        Education = float(request.form['Education'])
        Income = float(request.form['Income'])
        input_data = [HighBP, HighChol,	CholCheck, BMI, Smoker,	Stroke,	HeartDiseaseorAttack, PhysActivity,	Fruits,	Veggies, HvyAlcoholConsump,	AnyHealthcare, NoDocbcCost,	GenHlth, MentHlth, PhysHlth, DiffWalk, Sex,	Age, Education,	Income]
        print(input_data)
        print(type(input_data[0]))
        
        X = df.drop(columns=['Diabetes_binary'],axis=1)
        stand = StandardScaler()
        Fit_Transform = stand.fit(X)
        
        input_data_as_nparray = np.array(input_data)
        input_data_reshaped = input_data_as_nparray.reshape(1, -1)
        input_data_reshaped = Fit_Transform.transform(input_data_reshaped)
        print(input_data_reshaped)
        prediction = model.predict(input_data_reshaped)
        print(prediction[0])
        if (prediction[0] == 0.0):
            return render_template('result.html', result = 'Diabetes Absent', txtclr = 'green', name = __name__) 
        elif (prediction[0] == 1.0):
            return render_template('result.html', result='Diabetes Present', txtclr = 'red', name=__name__)
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)