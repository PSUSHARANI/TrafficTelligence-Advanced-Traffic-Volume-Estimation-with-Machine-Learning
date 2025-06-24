import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')


model = pickle.load(open('/Users/usharanips/Desktop/trafic project/Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
       
        input_feature = [float(x) for x in request.form.values()]
        feature_values = np.array(input_feature).reshape(1, -1)

        
        column_names = ['holiday', 'temp', 'rain', 'snow', 'weather',
                        'year', 'month', 'day', 'hours', 'minutes', 'seconds']

        
        data = pd.DataFrame(feature_values, columns=column_names)

        
        if hasattr(model, 'feature_names_in_'):
            data = data[model.feature_names_in_]

       
        prediction = model.predict(data)
        result_text =  str(int(prediction[0]))

       
        return render_template("result.html", prediction=result_text)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(port=port, debug=True, use_reloader=False)
