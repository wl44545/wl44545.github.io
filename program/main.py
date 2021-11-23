import logging
from data import Data
from measuring_quality.statictics import Statistics
from algorithms.naive_bayes.discrete_nbc import DiscreteNBC
from algorithms.boosting.real_boost import RealBoost
from algorithms.naive_bayes.bernoulli_nbc import BernoulliNBC
from algorithms.naive_bayes.complement_nbc import ComplementNBC
from algorithms.naive_bayes.gaussian_nbc import GaussianNBC
from algorithms.naive_bayes.multinomial_nbc import MultinomialNBC
from algorithms.support_vector_machines.linear_svm import LinearSVM
from algorithms.support_vector_machines.nonlinear_svm import NonLinearSVM
from algorithms.boosting.ada_boost import AdaBoost
from algorithms.boosting.gradient_boost import GradientBoost
from algorithms.neighbors.k_neighbors import KNeighbors
from algorithms.neural_networks.densenet121 import DenseNet121
from algorithms.neural_networks.densenet169 import DenseNet169
from algorithms.neural_networks.densenet201 import DenseNet201
from algorithms.neural_networks.efficientnetb0 import EfficientNetB0
from algorithms.neural_networks.efficientnetb1 import EfficientNetB1
from algorithms.neural_networks.efficientnetb2 import EfficientNetB2
from algorithms.neural_networks.efficientnetb3 import EfficientNetB3
from algorithms.neural_networks.efficientnetb4 import EfficientNetB4
from algorithms.neural_networks.efficientnetb5 import EfficientNetB5
from algorithms.neural_networks.efficientnetb6 import EfficientNetB6
from algorithms.neural_networks.efficientnetb7 import EfficientNetB7
from algorithms.neural_networks.inceptionresnetv2 import InceptionResNetV2
from algorithms.neural_networks.inceptionv3 import InceptionV3
from algorithms.neural_networks.mobilenet import MobileNet
from algorithms.neural_networks.mobilenetv2 import MobileNetV2
from algorithms.neural_networks.mobilenetv3large import MobileNetV3Large
from algorithms.neural_networks.mobilenetv3small import MobileNetV3Small
from algorithms.neural_networks.resnet101 import ResNet101
from algorithms.neural_networks.resnet101v2 import ResNet101V2
from algorithms.neural_networks.resnet152 import ResNet152
from algorithms.neural_networks.resnet152v2 import ResNet152V2
from algorithms.neural_networks.resnet50 import ResNet50
from algorithms.neural_networks.resnet50v2 import ResNet50V2
from algorithms.neural_networks.vgg16 import VGG16
from algorithms.neural_networks.vgg19 import VGG19
from algorithms.neural_networks.xception import Xception

LOG_FORMAT = '%(asctime)s %(levelname)s - %(module)s.%(funcName)s() : %(message)s'
logging.basicConfig(filename='resources/logs/info.log', level=logging.INFO, format=LOG_FORMAT)

data = Data()
data.make_data(5, 5, 0.25, 0.1)
data.preprocess_data(32)
data.pca()
# data.import_data(5000)
# data.dump_data(500)
# data.load_data(500)
# data.augment_data(0.1)
# data.split_data(0.25)

statistics = Statistics()
statistics.update_data(data)

# statistics.insert(DiscreteNBC(data).start())
# statistics.insert(RealBoost(data).start())
#
# statistics.insert(BernoulliNBC(data).start())
# statistics.insert(ComplementNBC(data).start())
# statistics.insert(GaussianNBC(data).start())
# statistics.insert(MultinomialNBC(data).start())
#
# statistics.insert(KNeighbors(data).start())
#
# statistics.insert(LinearSVM(data).start())
# statistics.insert(NonLinearSVM(data).start())
#
# statistics.insert(AdaBoost(data).start())
# statistics.insert(GradientBoost(data).start())
#
# statistics.insert(DenseNet121(data).start())
# statistics.insert(DenseNet169(data).start())
# statistics.insert(DenseNet201(data).start())
# statistics.insert(EfficientNetB0(data).start())
# statistics.insert(EfficientNetB1(data).start())
# statistics.insert(EfficientNetB2(data).start())
# statistics.insert(EfficientNetB3(data).start())
# statistics.insert(EfficientNetB4(data).start())
# statistics.insert(EfficientNetB5(data).start())
# statistics.insert(EfficientNetB6(data).start())
# statistics.insert(EfficientNetB7(data).start())
# statistics.insert(InceptionResNetV2(data).start())
# statistics.insert(InceptionV3(data).start())
# statistics.insert(MobileNet(data).start())
# statistics.insert(MobileNetV2(data).start())
# statistics.insert(MobileNetV3Large(data).start())
# statistics.insert(MobileNetV3Small(data).start())
# statistics.insert(ResNet101(data).start())
# statistics.insert(ResNet101V2(data).start())
# statistics.insert(ResNet152(data).start())
# statistics.insert(ResNet152V2(data).start())
# statistics.insert(ResNet50(data).start())
# statistics.insert(ResNet50V2(data).start())
# statistics.insert(VGG16(data).start())
# statistics.insert(VGG19(data).start())
# statistics.insert(Xception(data).start())

statistics.create_statistics()
statistics.export_csv()
statistics.export_html()
