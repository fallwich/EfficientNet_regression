import os
import sys
import json
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torchvision.models as models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efficientnet_pytorch import EfficientNet

# PIL 경고 무시 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 대소문자를 무시하고 파일 존재 여부 확인 함수
def file_exists_ignore_case(path):
    directory, filename = os.path.split(path)
    if not os.path.exists(directory):
        return False
    for existing_file in os.listdir(directory):
        if existing_file.lower() == filename.lower():
            return os.path.join(directory, existing_file)
    return False

# 데이터 준비 함수
def prepare_data(image_base_path, json_base_path):
    image_files = []
    json_files = []

    # 이미지 파일 경로 수집
    for root, dirs, files in os.walk(image_base_path):
        for file in files:
            if file.endswith('.png'):
                image_files.append(os.path.join(root, file))

    # JSON 파일 경로 수집
    for root, dirs, files in os.walk(json_base_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    print(f'Found {len(image_files)} image files.')
    print(f'Found {len(json_files)} JSON files.')

    data = []
    missing_files = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            filename = json_data['filename']
            age_past = json_data['age_past']
            folder_name = f"VS_{filename[:4]}"
            
            # 확장자가 이미 포함되어 있는지 확인하고 제거
            if filename.lower().endswith('.png'):
                filename = filename[:-4]
                
            image_path = os.path.join(image_base_path, folder_name, filename) + ".png"
            
            # 파일 존재 여부 확인 (대소문자 무시)
            actual_path = file_exists_ignore_case(image_path)
            if actual_path:
                data.append({'image_location': actual_path, 'filename': filename, 'age': age_past})
            else:
                missing_files.append(image_path)

    print(f'Missing files: {len(missing_files)}')
    for missing_file in missing_files:
        print(missing_file)
    df = pd.DataFrame(data)
    return df

# 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.copy()
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_location"]
        image = Image.open(img_path).convert("RGB")
        age = self.dataframe.iloc[idx]["age"].astype(np.float32)
        filename = self.dataframe.iloc[idx]["filename"]  # filename 추가
        if self.transform:
            image = self.transform(image)
        return image, age, filename  # filename 반환

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터 준비
image_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Validation/01.image"
json_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Validation/02.label"
df = prepare_data(image_base_path, json_base_path)

# 데이터셋 생성
dataset = CustomDataset(df, transform=transform)

# 데이터 로더 생성
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)



# CustomEfficientNet 모델 정의
class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0', dropout_rate=0.4):
        super(CustomEfficientNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained(model_name)
        
        # Pretrained weights freeze
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Rebuild top
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(self.base_model._fc.in_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.base_model._fc.in_features, 1)  # 회귀 문제를 위한 선형 레이어
        
        for n, p in self.base_model.named_parameters():
            if '_blocks.7.' in n or '_blocks.8.' in n or \
                '_blocks.9.' in n or '_blocks.10.' in n or '_blocks.11.' in n or \
                '_blocks.12.' in n or '_blocks.13.' in n or '_blocks.14.' in n or \
                '_blocks.15.' in n or '_conv_head' in n or '_bn1' in n or '_fc' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x):
        x = self.base_model.extract_features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomEfficientNet().to(device)
model.load_state_dict(torch.load("/home/jaewoong/EF_Regression/Weight/Age_Regreesion_8_block/models/eff_net_best.pth"))
model.eval()

# 예측 수행
predictions = []
true_ages = []
filenames = []
with torch.no_grad():
    for images, ages, filename in data_loader:  # filename 추가
        images = images.to(device)
        outputs = model(images)
        predictions.extend(outputs.cpu().numpy())
        true_ages.extend(ages.numpy())
        filenames.extend(filename)  # filename은 이미 리스트 형태로 저장되기 때문에 리스트 확장을 사용합니다.

# 결과 출력
for filename, true_age, pred_age in zip(filenames, true_ages, predictions):
    print(f"Image: {filename}.png, True Age: {true_age:.2f}, Predicted Age: {pred_age[0]:.2f}")

# 모델 평가 지표 계산
true_ages = np.array(true_ages)
predictions = np.array(predictions).flatten()

mae = np.mean(np.abs(true_ages - predictions))
# MAPE 계산 시 실제 값이 0인 경우 제외
mask = true_ages != 0
mape = np.mean(np.abs((true_ages[mask] - predictions[mask]) / true_ages[mask])) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")