import torch
import faiss
import numpy as np
from PIL import Image
import csv
from dinov2_class import preprocess_input_data, DINOV2


class FaissConstructor:
    def __init__(self, model):
        self.model = model
        self.index = faiss.IndexFlatL2(384)


    def add_vector_to_index(self, embeddings):
        """
            embeddings vector를 정규화해서 self.index에 추가함.
        """
        # Convert to float32 numpy
        vector = np.float32(embeddings)

        # Normalize vector: important to avoid wrong results when searching
        faiss.normalize_L2(vector)

        #Add to index
        self.index.add(vector)


    def extract_embeddings(self, images: list):
        for image_path in images:
            with torch.no_grad():
                preprocessed_image = preprocess_input_data(image_path)
                features = self.model.compute_embeddings(preprocessed_image)
            self.add_vector_to_index(embedding=features)


    def write_index(self, vector_index):
        faiss.write_index(self.index, vector_index)
        print(f"Successfully created {vector_index}")


    def search_k_similar_images(self, vector_index, input_image, k=1):
        # index 파일 불러오기
        index = faiss.read_index(vector_index)

        # OpenClip에서 추출된 이미지를 dinov2 모델에 입력 가능한 형태로 변환하여 임베딩 계산
        input_image_embeddings = self.model.compute_embeddings(preprocess_input_data(input_image[0]))

        # FAISS 검색 수행
        distances, indices = index.search(input_image_embeddings, k)

        # 결과 테스트
        print("distance: ", distances[0][0], " indices: ", indices[0][0])
        return
        


if __name__ == "__main__":
    IMAGE_PATH = ['/home/baesik/24Winter_OOP_AI_Agent/data/cat10.jpg']
    dinov2 = DINOV2()
    dinov2.load_model('vits14')
    fc = FaissConstructor(dinov2)
    fc.extract_embeddings(IMAGE_PATH)
    fc.write_index("vector.index")

    fc.search_k_similar_images("/home/baesik/24Winter_OOP_AI_Agent/vector.index", IMAGE_PATH)