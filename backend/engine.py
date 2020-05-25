from classification import doRandomForestClassification

class ClassificationEngine:


    def trainModel(self, filename):
        accuracy, self.predictions, self.model = doRandomForestClassification(filename)
        self.testError = 1.0 - accuracy
        return self.testError, self.predictions

    def __init__(self):
        return
