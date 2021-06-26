import cv2
from program.data import Data
from program.measuring_quality.measuring_quality import MeasuringQuality
from program.measuring_quality.statictics import Statistics
from program.algorithms.naive_bayes.bernoulli_nbc import BernoulliNBC
from program.algorithms.naive_bayes.complement_nbc import ComplementNBC
from program.algorithms.naive_bayes.gaussian_nbc import GaussianNBC
from program.algorithms.naive_bayes.multinomial_nbc import MultinomialNBC
from program.algorithms.support_vector_machines.linear_svm import LinearSVM
from program.algorithms.support_vector_machines.nonlinear_svm import NonLinearSVM
from program.algorithms.boosting.ada_boost import AdaBoost
from program.algorithms.boosting.gradient_boost import GradientBoost


data = Data()
# data.import_data(1000)
# data.dump_data(1000)

data.load_data(100)
data.split_data(0.25)

statistics = Statistics()

statistics.insert(BernoulliNBC(data).start())
statistics.insert(ComplementNBC(data).start())
statistics.insert(GaussianNBC(data).start())
statistics.insert(MultinomialNBC(data).start())

# statistics.insert(LinearSVM(data).start())
statistics.insert(NonLinearSVM(data).start())
# statistics.insert(AdaBoost(data).start())
# statistics.insert(GradientBoost(data).start())

statistics.create_statistics()
statistics.show()
statistics.export_csv()
statistics.export_html()




