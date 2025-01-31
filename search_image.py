"""
    모든 파이프라인을 결합한 데모 시연용 코드가 작성되는 곳
"""
import sys
import os

def add_project_paths():
    """
        커스텀 패키지를 인식하도록 sys.path에 패키지에 경로를 추가함.
    """
    cwd = os.getcwd()
    src_dir = os.path.join(cwd, "src/")
    your_models_dir = os.path.join(cwd, "your_models/")

    paths = [src_dir, your_models_dir]
    for path in paths:
        sys.path.append(path)

# sys.path에 패키지 경로 추가
add_project_paths()

from your_models.openclip_class import OpenCLIP
from your_models.dinov2_class import DINOV2
from your_models.faiss_constructor import FaissConstructor, OpenCLIPConstructor
from src.index import generate_image_index, get_image_path_by_index

# 기본 데이터 경로
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_PATH, "data/flickr30k/Images")
INDEX_PATH = os.path.join(SCRIPT_PATH, "dinov2.index")
IMAGE_INDEX_PATH = os.path.join(SCRIPT_PATH, "image_index.json")


# 이미지 인덱스 파일 생성 (.json) 
# e.g) {idx : image_file_path}
generate_image_index(IMAGE_PATH, IMAGE_INDEX_PATH)

# dinov2 사전 훈련 모델 로드
dinov2 = DINOV2()
dinov2.load_model('vits14', use_pretrained=True)

# OpenCLIP 사전 훈련 모델 로드
openclip = OpenCLIP()
openclip.load_model()

# dinov2 이미지 전처리
preprocessed_images = dinov2.preprocess_input_data(IMAGE_PATH)

# 전처리된 이미지 각각의 임베딩 생성
embedding_results = dinov2.compute_embeddings(preprocessed_images)

# dinov2.index 생성
dinov2_fc = FaissConstructor(dinov2)
dinov2_fc.add_vector_to_index(embedding_results)
dinov2_fc.write_index("dinov2.index")

# 임베딩 생성
openclip_images = openclip.preprocess_image_data(IMAGE_PATH)
openclip_image_embeddings = openclip.compute_image_embeddings(openclip_images)

# openclip.index 생성
openclip_fc = OpenCLIPConstructor(openclip)
openclip_fc.add_vector_to_index(openclip_image_embeddings)
openclip_fc.write_index("openclip.index")

# 이미지 & 캡션 파일 전처리
# caption = ["a child pushing a stroller"]

def search_image(input_query):
    text_embedding = openclip.preprocess_text([input_query])
    text_embedding = openclip.inference(openclip.model, text=text_embedding)

    # 캡션 전처리 2: 텍스트 임베딩을 검색에 적합하도록 변환
    text_embeddings = openclip_fc.fix_embedding_type(text_embedding)

    # 사용자가 입력한 검색어(text)의 의미와 가장 유사한 이미지 검색
    # 예시) openclip_result_image_path = openclip_fc.search(text)
    openclip_result_image_index = str(openclip_fc.search(text_embeddings)[0][0])
    openclip_result_image_file = os.path.join(IMAGE_PATH, get_image_path_by_index(openclip_result_image_index, IMAGE_INDEX_PATH))

    # OpenCLIP 이미지 검색을 통해 얻은 이미지 인덱스를 통해 얻은 이미지에 대해 dinov2로 유사한 이미지를 검색함.
    image_index = dinov2_fc.search_k_similar_images(INDEX_PATH, input_image=openclip_result_image_file, k=3)
    dinov2_result_image_file = get_image_path_by_index(str(image_index[0][1]), image_index_path=IMAGE_INDEX_PATH)

    return dinov2_result_image_file

