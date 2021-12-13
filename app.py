import pickle
import numpy as np
from flask import Flask, render_template,request

# Global variables
app=Flask(__name__)
loadedModel = pickle.load(open('KNN model.pkl','rb'))

#routes
@app.route('/')
def home():
    return render_template('ad.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    Daily_Internet_Usage = request.form['Daily_Internet_Usage']
    Daily_Time_Spent_on_Site = request.form['Daily_Time_Spent_on_Site']
    Age = request.form['Age']
    Area_Income = request.form['Area_Income']
    
    prediction = loadedModel.predict([[ Daily_Internet_Usage,Daily_Time_Spent_on_Site,Age,Area_Income]])[0]
    probability = np.max(loadedModel.predict_proba([[ Daily_Internet_Usage,Daily_Time_Spent_on_Site,Age,Area_Income]]))
    probability = str(np.round(probability, 2) * 100) + "%"
    
    if prediction == 0:
        prediction= probability + ' chance the user will not click on the ad'
    elif prediction == 1:
        prediction = probability + ' chance the user will click on the ad'

    return render_template('ad.html', api_output=prediction)    

if __name__ == '__main__':
    app.run(debug=True)