import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
import streamlit as st
from variety import audio
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

    def process(self):
        files = self.get_file_paths(self.random_training_path)
        # read the test directory and get the list of test audio files
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)
            winner = self.identify_gender(vector)
            #expected_gender = file.split("/")[1][:-1]

            #print("%10s %6s %1s" % ("+ EXPECTATION", ":", expected_gender))
            print("%10s %3s %1s" % ("+ IDENTIFICATION", ":", winner))
            st.write("%10s %3s %1s" % ("+ IDENTIFICATION", ":", winner))
            #if winner != expected_gender: self.error += 1
            print("----------------------------------------------------")

        #print(f"Accuracy: {audio:.2f}%")

    def get_file_paths(self, random_training_path):
        # get file paths
        random = [os.path.join(random_training_path, f) for f in os.listdir(random_training_path)]
        files = random
        return files

    def identify_gender(self, vector):
        # female hypothesis scoring
        is_female_scores = np.array(self.females_gmm.score(vector))
        is_female_log_likelihood = is_female_scores.sum()
        # male hypothesis scoring
        is_male_scores = np.array(self.males_gmm.score(vector))
        is_male_log_likelihood = is_male_scores.sum()

        print("%10s %5s %1s" % ("+ FEMALE SCORE", ":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_log_likelihood, 3))))

        if abs(is_male_log_likelihood) > abs(is_female_log_likelihood):
            winner = "male"
        else:
            winner = "female"
        return winner


if __name__ == "__main__":
    gender_identifier = GenderIdentifier("TestingData/random", "females.gmm", "males.gmm")
    gender_identifier.process()

