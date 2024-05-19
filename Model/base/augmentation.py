from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from enum import Enum


class Strategies(Enum):
    OverSampling = RandomOverSampler(random_state=0)
    SMOTESampling = SMOTE(k_neighbors=2)
    ADASYNSampling = ADASYN()


class Augmentation:

    def __init__(self, X, Y):
        """
            X : featuers set
            Y : label set
        """

        self.Orignial_features = X
        self.Orignial_labels = Y

    def generate(self, strategy: Strategies):
        engine = strategy.value
        self.x_resampled, self.y_resampled = engine.fit_resample(
            self.Orignial_features, self.Orignial_labels)
        return (self.x_resampled, self.y_resampled)

    def get_generate_data(self):
        if self.x_resampled is None or self.y_resampled is None:
            self.generate(self, "")

        return (self.x_resampled, self.y_resampled)
