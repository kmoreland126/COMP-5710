from flask import Flask, request, jsonify
import sys

app = Flask(__name__)

# Define the root endpoint for GET requests
@app.route('/', methods=['GET'])
def home():
    return "<h1>Welcome to a Simple Flask API!</h1>"

# Define an endpoint for GET requests
@app.route('/sqa', methods=['GET'])
def greetSQA():
    return "<h1>Welcome to the SQA course!</h1>"

@app.route('/ssp', methods=['GET'])
def greetSSP():
	return "<h1>Secure Software Process<h1>"

@app.route('/vanity', methods=['GET'])
def greetVanity():
	return "<h1>Kate Ella Moreland<h1>"

@app.route('/mypython', methods=['GET'])
def greetmypython():
	return sys.version

@app.route('/csse', methods=['GET'])
def greetCSSE():
	return "<h1>Department of Computer Science and Software Engineering<h1>"

if __name__ == '__main__':
    app.run(debug=True)
