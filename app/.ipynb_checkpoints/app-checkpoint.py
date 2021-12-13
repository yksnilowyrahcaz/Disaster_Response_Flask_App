from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', pageTitle='Spelling Bee')

if __name__ == '__main__':
    app.run(debug=True)