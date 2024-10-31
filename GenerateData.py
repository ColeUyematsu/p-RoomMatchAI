import pandas as pd
import numpy as np

questions = [
    "How well do you live together with your roommate?", 
    "How strictly do you follow a schedule?",
    "How comfortable are you with having guests stay over?", 
    "How serious is your romantic relationship?", 
    "How much do you enjoy different music genres?", 
    "How much are you planning to pack for school?", 
    "How important is it to have a TV in your room?",
    "How well do you study in environments with social activity?", 
    "Are you an athlete? Impact on sleep schedule?", 
    "How often do you travel or go off-campus?", 
    "Importance of going to bed at the same time?", 
    "Frequency of watching TV or movies?", 
    "Importance of similar major or classes?", 
    "Opinion on quiet hours?", 
    "Earliest alarm time?", 
    "Frequency of recreational smoking or drinking?", 
    "Frequency of loud appliance use?", 
    "Expected frequency of guests?", 
    "Value of room decor?", 
    "Desired closeness with roommate?", 
    "Expected frequency of family visits?", 
    "Frequency of doing own laundry?", 
    "Prioritization: work, play, cleaning, rest (rank each from 1-10)", 
    "Likelihood of going out socially weekly?", 
    "Weekday bedtime?", 
    "Noise level in room?", 
    "Room presence frequency?", 
    "Ideal room temperature?", 
    "Window opening frequency?", 
    "Flexibility with different schedules?", 
    "Preference for guest curfew?", 
    "Importance of knowing neighbors?", 
    "Frequency of eating in room?", 
    "Directness in handling conflict?"
]

num_samples = 100
data = {question: np.random.randint(1, 11, num_samples) for question in questions}
data['NAME'] = [f"Person_{i+1}" for i in range(num_samples)]

df = pd.DataFrame(data)
df.to_csv("test.csv", index=False)

