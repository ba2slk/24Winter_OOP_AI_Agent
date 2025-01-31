import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from .openclip.src.open_clip.tokenizer import decode
from .openclip.src.open_clip.factory import create_model_from_pretrained, create_model_and_transforms, get_tokenizer

from .base_model import BaseModel


script_path = os.path.dirname(os.path.realpath(__file__))


class OpenCLIP(BaseModel):
    def __init__(
            self, 
            model_architecture='ViT-B-32', 
            pretrained=True, 
            pretrained_model='laion2b_s34b_b79k'):
        # torch가 사용할 device 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_architecture = model_architecture
        self.pretrained = pretrained
        self.pretrained_model = pretrained_model

    # 이미지 확인용 함수
    def show_image(self, image_source):
        image = Image.open(image_source)
        image = np.array(image.convert("RGB"))

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('on')
        plt.show()

    def load_model(self):
        # pretrained model만을 가정하므로 preprocess_train은 반환하지 않음.
        self.model, _, self.preprocess = create_model_and_transforms(self.model_architecture, pretrained=self.pretrained_model)

        self.model.eval() # 평가 모드
        self.tokenizer = get_tokenizer(self.model_architecture)

    
    # 텍스트만 전처리
    def preprocess_text(self, text_source):
        text = self.tokenizer(text_source)
        return text
    
    # 이미지만 전처리 (단일 이미지)
    def preprocess_image(self, image_source):
        image = Image.open(image_source).convert('RGB')
        image = self.preprocess(image).unsqueeze(0)
        return image
    
    # 다수의 이미지 전처리
    def preprocess_image_data(self, image_source):
        all_images = []

        image_dir = os.fsencode(image_source)
        # print(f"image_dir : {image_dir}")

        # 테스트: 일부 이미지만 로드
        for file in os.listdir(image_dir)[:1000]:
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".JPEG"): 
                image = self.preprocess_image(os.path.join(image_source, filename))
                all_images.append(image)
            else:
                continue
        return all_images
    
    # 이미지, 텍스트 한 번에 전처리
    def preprocessing(self, image_source, text_source):
        image = self.preprocess_image(image_source)
        text = self.preprocess_text(text_source)
        return image, text
    
    def compute_similarity(self, image_features, text_features):
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return probs
    
    def inference(self, model, image=None, text=None):
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            if image is not None and text is not None:
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = self.compute_similarity(image_features, text_features)
                return text_probs
            elif image is not None:
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                return image_features
            elif text is not None:
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                return text_features
            else:
                raise ValueError("Either image or text must be provided.")

    # 다수의 전처리된 이미지에 대해 임베딩 계산
    def compute_image_embeddings(self, images):
        all_embeddings = []
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            for image in images:
                # image_embedding = self.inference(self.model, image)
                image_embedding = self.model.encode_image(image)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
                all_embeddings.append(image_embedding)
        return all_embeddings
    
    # def compute_embeddings(self, images):
    #     all_embeddings = []
    #     with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
    #         for image in images:
    #             image_embedding = self.inference(self.model, image)
    #             all_embeddings.append(image_embedding)
    #     return all_embeddings

    def generate_text(self, model, image_source):
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            image = self.preprocess_image(image_source)
            generated = model.generate(image)
            print(decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
        return generated

# 테스트용 함수...
def test_generate_text():
    print("*********** Use pretrained model ***********")
    IMAGE_PATH = "../data/horse.jpg" # 텍스트를 생성할 이미지
    TEXT_SOURCE = ['a cat', 'a dog', 'a horse']

    # model = OpenCLIP(
    #     model_architecture='ViT-B-32', 
    #     pretrained=True, 
    #     pretrained_model='laion2b_s34b_b79k')

    model = OpenCLIP(
        model_architecture='coca_ViT-L-14', 
        pretrained=True, 
        pretrained_model='mscoco_finetuned_laion2B-s13B-b90k')
    
    # input = model.show_image(image_source=IMAGE_PATH)

    # loaded_model, tokenizer, preprocess = model.load_model()
    model.load_model()

    # model.generate_text(loaded_model, IMAGE_PATH, preprocess)
    model.generate_text(model.model, IMAGE_PATH)

# 테스트용 함수...
def test_text_probs():
    print("*********** Use pretrained model ***********")
    IMAGE_PATH = "../data/cat.jpg"
    TEXT_SOURCE = ['a cat', 'a dog', 'a horse']

    model = OpenCLIP()
    model.load_model()

    image, text = model.preprocessing(image_source=IMAGE_PATH, text_source=TEXT_SOURCE)

    text_probs = model.inference(model.model, image, text)

    print("* Image: ", IMAGE_PATH)
    print("* Text: ", TEXT_SOURCE)
    print("* Label probs:", text_probs)

def get_text_input():
    user_input = input("Enter the text: ")
    return user_input

def main():
    # 이미지 파일이 담긴 디렉터리 경로
    IMAGE_PATH = os.path.join(script_path, "../data/flickr30k/Images")

    # 모델 세팅
    openclip = OpenCLIP()
    openclip.load_model()

    # User input(text)을 임베딩 변환
    # text = get_text_input()
    text = ["a man wearing red shirt"]
    text = openclip.preprocess_text(text)
    text_embedding = openclip.inference(openclip.model, text=text)
    print("* Text embedding shape: ", text_embedding.shape)

    # 이미지 로드
    images = openclip.preprocess_image_data(IMAGE_PATH)

    # 이미지 임베딩 변환
    image_embeddings = openclip.compute_image_embeddings(images)
    print("* Image embedding shape: ", image_embeddings[0].shape)

    # 텍스트 임베딩과 이미지 임베딩을 비교하여 가장 유사한 이미지 검색
    similarities = []
    for image_embedding in image_embeddings:
        similarity = openclip.compute_similarity(image_embedding, text_embedding)
        similarities.append(similarity.item())

    print("* Similarity: ", similarities[3])

    # 가장 유사한 이미지의 인덱스를 찾음
    most_similar_index = np.argmax(similarities)
    print(f"* Most similar image index: {most_similar_index}")

    # 가장 유사한 이미지 출력
    most_similar_image_path = os.listdir(IMAGE_PATH)[most_similar_index]
    openclip.show_image(os.path.join(IMAGE_PATH, most_similar_image_path))


if __name__ == "__main__":
    main()
    # test_text_probs()