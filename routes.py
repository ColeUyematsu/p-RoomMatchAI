from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd

routes = Blueprint('main', __name__)

# Homepage route
@routes.route('/')
def home():
    return render_template('index.html')

# Questionnaire route
@routes.route('/questionnaire.html', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'POST':
        # Get responses and save them
        responses = request.form.to_dict()
        df = pd.DataFrame([responses])
        df.to_csv('questionnaire_responses.csv', mode='a', header=False, index=False)
        return redirect(url_for('main.thank_you'))

    # Sample dynamic questions (can load from DB or file)
    questions = [
        {'id': 'q1', 'text': 'What is your preferred study time?'},
        {'id': 'q2', 'text': 'Do you prefer a quiet or active living space?'},
        {'id': 'q3', 'text': 'How clean do you like your room?'}
    ]
    return render_template('questionaire.html', questions=questions)

# Thank you page route
@routes.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')