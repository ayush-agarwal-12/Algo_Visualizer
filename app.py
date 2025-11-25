from flask import Flask, render_template, request, jsonify
from algorithms import MLAlgorithms

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_algo', methods=['POST'])
def run_algo():
    req = request.json
    algo = req.get('algo')
    params = req.get('params', {})
    
    result = {}
    
    try:
        if algo == 'linear_regression':
            result = MLAlgorithms.linear_regression_scratch(
                float(params.get('param1', 0.05)), int(params.get('param2', 30)))
        elif algo == 'polynomial_regression':
            result = MLAlgorithms.polynomial_regression_scratch(
                2, float(params.get('param1', 0.0001)), int(params.get('param2', 50)))
        elif algo == 'logistic_regression':
            result = MLAlgorithms.logistic_regression_scratch(
                float(params.get('param1', 0.1)), int(params.get('param2', 30)))
        elif algo == 'svm':
            result = MLAlgorithms.svm_scratch(
                float(params.get('param1', 0.01)), int(params.get('param2', 40)), float(params.get('param3', 1.0)))
        elif algo == 'knn':
            result = MLAlgorithms.knn_scratch(int(params.get('param1', 3)))
        elif algo == 'naive_bayes':
            result = MLAlgorithms.naive_bayes_scratch()
        elif algo == 'decision_tree':
            result = MLAlgorithms.decision_tree_scratch(int(params.get('param1', 3)))
        elif algo == 'kmeans':
            result = MLAlgorithms.kmeans_scratch(
                int(params.get('param1', 3)), int(params.get('param2', 10)))
        elif algo == 'dbscan':
            result = MLAlgorithms.dbscan_scratch(
                float(params.get('param1', 0.5)), int(params.get('param2', 4)))
                
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)