import torch
import faiss
import numpy as np
from PIL import Image
import csv
import os

# from your_models.dinov2_class import DINOV2 # 아래처럼 수정함
from .dinov2_class import DINOV2

from .openclip_class import OpenCLIP


script_path = os.path.dirname(os.path.realpath(__file__))


class FaissConstructor:
    def __init__(self, model):
        self.model = model
        self.index = faiss.IndexFlatL2(384)


    def add_vector_to_index(self, all_embeddings: list):
        """
            embeddings vector를 정규화해서 self.index에 추가함.
        """
        for embeddings in all_embeddings:
            embeddings = np.float32(embeddings)
            faiss.normalize_L2(embeddings)
            #Add to index
            self.index.add(embeddings)


    def write_index(self, vector_index):
        faiss.write_index(self.index, vector_index)
        print(f"Successfully created {vector_index}")


    def search_k_similar_images(self, vector_index, input_image, k=1):
        # index 파일 불러오기
        index = faiss.read_index(vector_index)

        # OpenClip에서 추출된 이미지를 dinov2 모델에 입력 가능한 형태로 변환하여 임베딩 계산
        preprocessed_image = self.model.preprocess_input_data(input_image)
        input_image_embeddings = self.model.compute_embeddings(preprocessed_image)[0]

        # FAISS 검색 수행
        distances, indices = index.search(input_image_embeddings, k)

        # 결과 출력
        # print("in FaissConstructor.search_k_similar_images: ")
        # print("distance: ", distances[0][0], " indices: ", indices[0][0])

        return indices
    

class OpenCLIPConstructor(FaissConstructor):
    def __init__(self, model):
        self.model = model
        self.index = faiss.IndexFlatIP(512)

    def fix_embedding_type(self, embedding):
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.to(torch.float32)  # 먼저 float32로 변환
            embedding = embedding.cpu().numpy()  # NumPy 배열로 변환
        embedding = np.float32(embedding)
        
        return embedding

    def add_vector_to_index(self, all_embeddings):
        for embedding in all_embeddings:
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.to(torch.float32)  # 먼저 float32로 변환
                embedding = embedding.cpu().numpy()  # NumPy 배열로 변환
            embedding = np.float32(embedding)  # 다시 float32로 변환
            self.index.add(embedding)
        print(f"* [OpenCLIP] {self.index.ntotal}개의 이미지 임베딩이 FAISS에 추가됨.")

    def search(self, text_embedding):
        distances, indices = self.index.search(text_embedding, 1)

        return indices

def openclip_faiss():
    IMAGE_PATH = os.path.join(script_path, "../data/flickr30k/Images")
    INEDEX_PATH = os.path.join(script_path, "../dinov2.index")

    openclip = OpenCLIP()
    openclip.load_model()

    images = openclip.preprocess_image_data(IMAGE_PATH)
    embeddings = openclip.compute_image_embeddings(images)

    embedding_results = dinov2.compute_embeddings(images)

    fc = FaissConstructor(dinov2)
    fc.add_vector_to_index(embedding_results)
    fc.write_index("openclip.index")
    

if __name__ == "__main__":
    IMAGE_PATH = os.path.join(script_path, "../data/flickr30k/Images")
    INEDEX_PATH = os.path.join(script_path, "../dinov2.index")

    dinov2 = DINOV2()
    dinov2.load_model('vits14')
    images = dinov2.preprocess_input_data(IMAGE_PATH)
    embedding_results = dinov2.compute_embeddings(images)

    fc = FaissConstructor(dinov2)
    fc.add_vector_to_index(embedding_results)
    fc.write_index("dinov2.index")

    target = os.path.join(script_path, "../data/val/")

    image_index = fc.search_k_similar_images(INEDEX_PATH, input_image=target)
    print(image_index[0][0])