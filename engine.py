from . import classification


class ClassificationEngine:

    def trainModel(self, filename):
        accuracy, self.predictions, self.model = classification.doRandomForestClassification(filename)
        self.testError = 1.0 - accuracy
        return self.testError, self.predictions

    def __init__(self):
        return
