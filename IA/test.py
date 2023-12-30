from flask import Flask, render_template, request
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC

app = Flask(__name__)

# Charger le jeu de données Iris
iris = load_iris()

# Créer et entraîner le modèle SVM
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(iris.data, iris.target)

@app.route('/')
def index():
    return render_template('index.html', feature_names=iris.feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    new_flower_features = [request.form[feature] for feature in iris.feature_names]
    new_flower_features = np.array(new_flower_features).astype(float)
    predicted_species = svm_classifier.predict([new_flower_features])
    result = f"{iris.target_names[predicted_species][0]}"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

