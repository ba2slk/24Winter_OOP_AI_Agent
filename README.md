# EasyViT: Effortlessly Integrate the Latest ViT Models into Your Application
---
## Welcome to EasyViT
EasyVIT는 범용적이고 직관적인 메서드를 기반으로, 사전 훈련된 ViT(Vision Transformer) 모델을 손쉽게 자신의 프로젝트에 적용할 수 있도록 돕는 프레임워크입니다. 

PyTorch 기반의 사전 훈련된 가중치만 있다면, Hugging Face에서 다운로드한 모델 뿐만 아니라 자체 데이터셋으로 훈련된 모델도 EasyViT를 활용해 이미지 분류, 검색 등 다양한 다운스트림 작업에 유연하게 적용할 수 있습니다.

---
## 지원하는 모델
- [dinov2](https://github.com/facebookresearch/dinov2.git)
- [OpenClip](https://github.com/mlfoundations/open_clip.git)
---
# 프로젝트 구조
```
├── app: Flask 기반 웹 애플리케이션 데모
├── checkpoints: 사전 훈련된 체크포인트 모음
│   └── dinov2
│       ├── backbones
│       └── heads
├── data: 데이터 셋 저장
├── src: 유틸리티 스크립트
└── your_models: 모델별 아키텍처 원본 및 각 모델의 클래스(xxx_class.py)
│   ├── dinov2
│   └── openclip
└── search_image.py: 구현 예) 이미지 검색 파이프라인
```
___
## 사용 방법
### 1. 사용 환경 설정하기
프로젝트 구동 환경은 다음과 같습니다.
```
	운영체제: Windows + Ubuntu 22.04 (WSL2)
	언어: Python 3.9.21
	딥러닝 프레임워크: PyTorch (CUDA 지원 가능)
```

Conda Prompt에서 가상 환경을 활성화하고, `requirements.txt`에 담긴 패키지를 모두 설치합니다.
```
	pip install -r requirements.txt
```

### 2. 데이터 셋 준비하기
EasyViT에서는 사용하고 싶은 어떠한 이미지 데이터 셋도 사용 가능합니다. 데이터 셋이 없다면 `data/` 디렉토리를 생성한 다음, [여기](https://github.com/awsaf49/flickr-dataset.git)서 `flickr30k`이미지 데이터 셋을 다운로드 할 수 있습니다.

### 3. 이미지 인덱스 생성하기
`src/index.py` 에서 `generate_image_index()` 함수를 호출해서 다운로드한 이미지 파일 각각에 대해 인덱스가 부여된 JSON 파일(e.g `image_index.json`)을 생성합니다.
```
생성 예
{
    "0": "1000092795.jpg",
    "1": "10002456.jpg",
    "2": "1000268201.jpg",
    "3": "1000344755.jpg",
    "4": "1000366164.jpg",
    "5": "1000523639.jpg",
    
    ...
}
```
### 4. 사전 훈련 모델 불러오기
EasyVit는 PyTorch 기반으로 사전 훈련된 모든 모델(.pth, .pt, .pkl)을 지원합니다. `src/load_pretrained_dinov2_backbone.py` 를 실행해서 torchHub를 통해 사전 훈련된 dinov2 모델을 다운로드할 수 있습니다.

사용하고 싶은 모델 클래스의 객체를 생성하고 직관적인 메소드를 호출하는 것만으로 ViT 모델을 쉽게 사용할 수 있습니다. 다음과 같은 방식으로 모델을 로드하고, 모델의 특성에 맞게 이미지를 전처리하는 것은 물론, 각 이미지에 대한 임베딩을 손쉽게 계산할 수 있습니다.

```python
from your_models.dinov2_class import DINOV2
from your_models.openclip_class import OpenCLIP

# dionv2
dinov2 = DINOV2() # dinov2 객체 생성
dinov2.load_model('vits14', use_pretrained=True) # dinov2 사전 훈련 모델 불러오기
preprocessed_images = dinov2.preprocess_image(IMAGE_PATH) # 데이터셋 내의 모든 이미지 데이터 전처리
embedding_results = dinov2.compute_embeddings(preprocessed_images) # 전처리된 모든 이미지에 대해 임베딩 계산

# OpenCLIP
openclip = OpenCLIP() # openclip 객체 생성
openclip.load_model() # openclip 사전 훈련 모델 불러오기
text_embedding = openclip.preprocess_text(caption) # 사용자 입력 텍스트 전처리
text_embedding = openclip.inference(openclip.model, text=text_embedding) # 텍스트 임베딩 생성
openclip_images = openclip.preprocess_image(IMAGE_PATH) # 데이터셋 내의 모든 이미지에 대해 전처리
openclip_image_embeddings = openclip.compute_image_embeddings(openclip_images) # 전처리된 모든 이미지에 대해 임베딩 계산
```

### 5. 인덱스 생성하기
EasyViT는 `faiss_constructor.py`에서 유사도 기반의 다운스트림 작업을 수행하는 데 필요한 faiss 기반의 `FaissConstructor` 클래스를 지원합니다. 여기서 개별 모델의 index DB를 생성할 수 있습니다. 역시 사용 방법은 간단합니다.

```python
# dinov2
dinov2_fc = FaissConstructor(dinov2)
dinov2_fc.add_vector_to_index(embedding_results)
dinov2_fc.write_index("dinov2.index")

# OpenCLIP
openclip_fc = OpenCLIPConstructor(openclip)
openclip_fc.add_vector_to_index(openclip_image_embeddings)
openclip_fc.write_index("openclip.index")
```

### 6. 유사도 검색하기
`FaissConstructor` 클래스와 이를 상속한 `OpenClipConstructor` 클래스가 지원하는 `search()`메소드를 사용해서 유사도 기반으로 이미지를 검색할 수 있습니다. 모델 특성상 dinov2는 `이미지-이미지` 유사도 검색을, OpenCLIP은 `텍스트-이미지` 유사도 검색을 수행합니다. (`이미지-텍스트` 등, 더 많은 응용은 추후 지원)

```python
# OpenCLIP
text_embeddings = openclip_fc.fix_embedding_type(text_embedding)
openclip_result_image_index = str(openclip_fc.search(text_embeddings)[0][0])
dinov2_result_image_file = get_image_path_by_index(str(image_index[0][0]), image_index_path=IMAGE_INDEX_PATH) # 인덱스에 해당하는 이미지 파일명 반환

# dinov2
image_index = dinov2_fc.search_k_similar_images(INDEX_PATH, openclip_result_image_file, k=3) # 상위 3개의 이미지 인덱스 반환
```
---
## EasyViT 활용 예시
### OpenCLIP과 DINOv2를 사용하여 유사도 기반 이미지 검색 파이프라인 구축하기

 이 예시 프로젝트는 사용자가 입력한 검색어 텍스트에 대해 OpenCLIP으로 텍스트 임베딩을 생성하고, 이를 기반으로 벡터 DB에서 가장 유사한 이미지를 검색합니다. 이후 DINOv2를 거쳐 검색된 이미지에 대한 고품질 특징 벡터를 추출한 다음, 이미지 유사도 검색을 수행하여 최종적으로 검색된 이미지를 제공합니다.

![24Winter_OOP_AI_Agent_Demo](https://github.com/user-attachments/assets/a0c4f8da-1b67-4983-bd2e-d5ac926e067b)
🔼 사전 훈련된 DINOv2와 OpenCLIP을 활용한 이미지 검색 파이프라인 구현 예시

---
# Envision Your Idea with EasyViT

EasyViT로 여러분만의 ViT 기반 딥러닝 프로젝트를 손쉽게 완성해 보세요. 😊

---
## Reference
EasyViT는 아래 두 논문의 모델들의 구현을 포함하고 있습니다.
- [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
