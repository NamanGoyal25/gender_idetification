import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor

from tkinter import *
from tkinter.filedialog import *

warnings.filterwarnings("ignore")


class GenderIdentifier:

    def __init__(self, random_files_path, females_model_path, males_model_path):
        self.random_training_path = random_files_path
        self.error = 0
        self.total_sample = 0
        self.features_extractor = FeaturesExtractor()
        # load models
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm = pickle.load(open(males_model_path, 'rb'))

    def process(self, file):
        self.total_sample += 1
        output_text.insert(END, "--> TESTING: " + os.path.basename(file) + "\n")

        vector = self.features_extractor.extract_features(file)
        winner = self.identify_gender(vector)
        expected_gender = file.split("/")[1][:-1]

        output_text.insert(END, "+ EXPECTATION: " + expected_gender + "\n")
        output_text.insert(END, "+ IDENTIFICATION: " + winner + "\n")

        if winner != expected_gender: self.error += 1
        output_text.insert(END, "----------------------------------------------------\n")

    def identify_gender(self, vector):
        # female hypothesis scoring
        is_female_scores = np.array(self.females_gmm.score(vector))
        is_female_log_likelihood = is_female_scores.sum()
        # male hypothesis scoring
        is_male_scores = np.array(self.males_gmm.score(vector))
        is_male_log_likelihood = is_male_scores.sum()

        output_text.insert(END, "+ FEMALE SCORE: " + str(round(is_female_log_likelihood, 3)) + "\n")
        output_text.insert(END, "+ MALE SCORE: " + str(round(is_male_log_likelihood, 3)) + "\n")

        if is_male_log_likelihood > is_female_log_likelihood:
            winner = "male"
        else:
            winner = "female"

        return winner


def browse():
    book = askopenfilename()
    identifier = GenderIdentifier("TestingData/random", "females.gmm", "males.gmm")
    identifier.process(book)


def test_gender():
    gender_identifier = GenderIdentifier("TestingData/random", "females.gmm", "males.gmm")
    gender_identifier.process()


window = Tk()
window.geometry("500x500")

window.title("Gender Identifier")
window.configure(bg='#ADD8E6')

browse_button = Button(text='Browse', command=browse, bg='#008080', fg='white')
browse_button.place(x=220, y=100)

output_text = Text(window, bg='black', fg='white')
output_text.place(x=50, y=200, width=400, height=250)

window.mainloop()
