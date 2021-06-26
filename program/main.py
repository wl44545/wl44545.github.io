import cv2
from program.data import Data
from program.measuring_quality.measuring_quality import MeasuringQuality
from program.measuring_quality.statictics import Statistics
from program.algorithms.linear_classifiers.bernoulli_nbc import BernoulliNBC
from program.algorithms.linear_classifiers.complement_nbc import ComplementNBC
from program.algorithms.linear_classifiers.gaussian_nbc import GaussianNBC
from program.algorithms.linear_classifiers.multinomial_nbc import MultinomialNBC
from program.algorithms.support_vector_machines.linear_svm import LinearSVM
from program.algorithms.support_vector_machines.nonlinear_svm import NonLinearSVM
from program.algorithms.boosting.ada_boost import AdaBoost
from program.algorithms.boosting.gradient_boost import GradientBoost



data = Data()
data.load_data(100)
data.split_data()

print(len(data.X_train),len(data.X_test),len(data.y_train),len(data.y_test))

statistics = Statistics()


# statistics.insert(BernoulliNBC(data).start())
# statistics.insert(ComplementNBC(data).start())
# statistics.insert(GaussianNBC(data).start())
# statistics.insert(MultinomialNBC(data).start())
#
# statistics.insert(LinearSVM(data).start())
# statistics.insert(NonLinearSVM(data).start())

statistics.insert(AdaBoost(data).start())
statistics.insert(GradientBoost(data).start())


statistics.create_statistics()
statistics.show()
statistics.export_csv()
statistics.export_html()


