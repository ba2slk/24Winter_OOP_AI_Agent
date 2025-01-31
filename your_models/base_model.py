from abc import ABC, abstractmethod

class BaseModel:
    model_name = None
    pretrained = False
    model = None

    @abstractmethod
    def load_model(self, image_source):
        pass

    @abstractmethod
    def preprocess_image(self, image_source):
        pass



    # @abstractmethod
    # def load_model(self):
    #     pass

    # @abstractmethod
    # def load_data(self):
    #     pass

    # @abstractmethod
    # def train(self, input_data):
    #     pass

    # @abstractmethod
    # def evaluate(self, output_data):
    #     pass