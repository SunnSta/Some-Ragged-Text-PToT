"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from RagWeb import app

@app.route('/')
@app.route('/input')
def home():
    """Renders the home page."""
    return render_template(
        'input.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/output')
def about():
    """Renders the about page."""
    return render_template(
        'output.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
