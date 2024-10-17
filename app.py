from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

# New route to display the questionnaire
@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'POST':
        # Collect questionnaire responses from the form
        responses = request.form.to_dict()
        
        # Store responses in a CSV file
        df = pd.DataFrame([responses])
        df.to_csv('questionnaire_responses.csv', mode='a', header=False, index=False)
        
        return redirect(url_for('thank_you'))

    # Dynamic questions (you can load these from a database or file)
    questions = [
        {'id': 'q1', 'text': 'What is your preferred study time?'},
        {'id': 'q2', 'text': 'Do you prefer a quiet or active living space?'},
        # Add more questions dynamically
    ]
    return render_template('questionnaire.html', questions=questions)

@app.route('/thank_you')
def thank_you():
    return "Thank you for completing the questionnaire!"