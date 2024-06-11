import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # for n, p in self.base_model.named_parameters():
        #     if '_blocks.11.' in n or '_blocks.12.' in n or '_blocks.13.' in n or '_blocks.14.' in n or '_blocks.15.' in n or '_conv_head' in n or '_bn1' in n or '_fc' in n:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        for n, p in self.base_model.named_parameters():
            # 10번째 블록부터 학습 가능하게 설정
            if '_blocks.9.' in n or '_blocks.10.' in n or '_blocks.11.' in n or \
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

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.copy()
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_location"]
        image = Image.open(img_path).convert("RGB")
        price = self.dataframe.iloc[idx]["price"].astype(np.float32)
        zpid = self.dataframe.iloc[idx]["zpid"]  # zpid 추가
        if self.transform:
            image = self.transform(image)
        return image, price, zpid  # zpid 반환

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
df = pd.read_pickle("/home/jaewoong/Downloads/data/df.pkl")
df["image_location"] = "/home/jaewoong/Downloads/data/processed_images/" + df["zpid"] + ".png"

# 데이터프레임 정렬 (예시로 'image_location' 열 기준으로 정렬)
df = df.sort_values(by="image_location")

# zpid_list 정의
zpid_list = ["32114254", "32114330", "32113965", "32114049", "32114404", "32113346", "32113549", "32114191", "32113652", "32113635"]

# zpid_list에 해당하는 데이터프레임 필터링
filtered_df = df[df["zpid"].isin(zpid_list)]

# 데이터셋 생성
dataset = CustomDataset(filtered_df, transform=transform)

# 데이터 로더 생성
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 모델 로드
model = CustomEfficientNet().to(device)
model.load_state_dict(torch.load("./Regression_8_block/models/eff_net_best.pth"))
# model.load_state_dict(torch.load("./Regression_2/models/eff_net_best.pth"))
model.eval()

# 예측 수행
predictions = []
true_prices = []
zpids = []
with torch.no_grad():
    for images, prices, zpid in data_loader:  # zpid 추가
        images = images.to(device)
        outputs = model(images)
        predictions.extend(outputs.cpu().numpy())
        true_prices.extend(prices.numpy())
        zpids.extend(zpid)  # zpid는 이미 리스트 형태로 저장되기 때문에 리스트 확장을 사용합니다.

# 결과 출력
for zpid, true_price, pred_price in zip(zpids, true_prices, predictions):
    print(f"Image: {zpid}.png, True Price: {true_price:.2f}, Predicted Price: {pred_price[0]:.2f}")

# 모델 평가 지표 계산
true_prices = np.array(true_prices)
predictions = np.array(predictions).flatten()

mae = np.mean(np.abs(true_prices - predictions))
mape = np.mean(np.abs((true_prices - predictions) / true_prices)) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
