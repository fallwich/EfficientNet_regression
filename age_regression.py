from typing import Iterator, List, Union, Tuple, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, self.df.columns.get_loc('image_location')]
        image = plt.imread(img_path)  # plt.imread 사용

        # RGBA 이미지를 RGB로 변환
        if image.shape[-1] == 4:
            image = image[..., :3]

        # float32 타입을 uint8 타입으로 변환
        if image.dtype == 'float32':
            image = (image * 255).astype('uint8')

        label = self.df.iloc[idx, self.df.columns.get_loc('age')].astype('float32')
        label = torch.tensor(label).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image, label

def visualize_augmentations(df: pd.DataFrame):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # numpy 배열을 PIL 이미지로 변환
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.ToTensor()
    ])

    series = df.iloc[2]
    img_path = series['image_location']
    image = plt.imread(img_path)  # plt.imread 사용

    # float32 타입을 uint8 타입으로 변환
    if image.dtype == 'float32':
        image = (image * 255).astype('uint8')

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i in range(9):
        ax = axes[i // 3, i % 3]
        augmented_image = transform(image)
        ax.imshow(augmented_image.permute(1, 2, 0))
        ax.axis('off')
    plt.show()

def get_mean_baseline(train: pd.DataFrame, val: pd.DataFrame) -> float:
    # train 데이터셋의 평균 나이 계산
    y_hat = train["age"].mean()
    # val 데이터셋의 모든 샘플에 대해 평균 나이를 예측 값으로 설정
    val["y_hat"] = y_hat

    mae_loss = nn.L1Loss()

    # MAPE 계산 함수
    def mape_loss(y_true, y_pred):
        mask = y_true != 0  # 실제 값이 0이 아닌 경우에만 계산
        return torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    val_targets = torch.tensor(val["age"].values, dtype=torch.float32)
    val_predictions = torch.tensor(val["y_hat"].values, dtype=torch.float32)

    # MAE 계산
    mae = mae_loss(val_predictions, val_targets).item()
    # MAPE 계산
    mape = mape_loss(val_targets, val_predictions).item()

    print(f'MAE: {mae}')
    print(f'Mean baseline MAPE: {mape}')

    return mape

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(df, test_size=0.2, random_state=1)  # split the data with a validation size o 20%
    train, test = train_test_split(
        train, test_size=0.125, random_state=1
    )  # split the data with an overall  test size of 10%

    print("shape train: ", train.shape)
    print("shape val: ", val.shape)
    print("shape test: ", test.shape)

    print("Descriptive statistics of train:")
    print(train.describe())
    return train, val, test

def create_dataloaders(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, 
                        plot_augmentations: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if plot_augmentations:
        visualize_augmentations(df)

    train_dataset = CustomDataset(train, transform=transform_train)
    val_dataset = CustomDataset(val, transform=transform_val_test)
    test_dataset = CustomDataset(test, transform=transform_val_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_freq=10):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_loss = None
        self.save_freq = save_freq

    def __call__(self, model, epoch, val_loss):
        # 매 10 에포크마다 저장
        if epoch % self.save_freq == 0:
            periodic_path = self.filepath.replace(".pth", f"_epoch_{epoch}.pth")
            if self.verbose:
                print(f'Saving periodic model at epoch {epoch} (every {self.save_freq} epochs)')
            torch.save(model.state_dict(), periodic_path)
        
        # 최저 검증 손실 갱신 시 저장
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            best_path = self.filepath.replace(".pth", "_best.pth")
            if self.verbose:
                print(f'Saving best model at epoch {epoch} with val_loss: {val_loss}')
            torch.save(model.state_dict(), best_path)
    
from torch.utils.tensorboard import SummaryWriter

def run_model(
    model_name: str,
    model_function: nn.Module,
    lr: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 300,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_function.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    early_stopping = EarlyStopping(patience=30, min_delta=1)
    checkpoint = ModelCheckpoint(filepath=f"./Age_Regreesion_8_block/models/{model_name}.pth",
                                    verbose=1, save_freq=5)
    writer = SummaryWriter(log_dir=f"logs/scalars/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        checkpoint(model, epoch, val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    writer.close()

    # 테스트 데이터로 평가
    best_model_path = f"./Age_Regreesion_8_block/models/{model_name}_best.pth"
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

    test_loss = running_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    return history

import torchvision.models as models
from efficientnet_pytorch import EfficientNet
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
        #     # 12번째 블록부터 학습 가능하게 설정
        #     if '_blocks.11.' in n or \
        #         '_blocks.12.' in n or '_blocks.13.' in n or '_blocks.14.' in n or \
        #         '_blocks.15.' in n or '_conv_head' in n or '_bn1' in n or '_fc' in n:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        # for n, p in self.base_model.named_parameters():
        #     # 10번째 블록부터 학습 가능하게 설정
        #     if '_blocks.9.' in n or '_blocks.10.' in n or '_blocks.11.' in n or \
        #         '_blocks.12.' in n or '_blocks.13.' in n or '_blocks.14.' in n or \
        #         '_blocks.15.' in n or '_conv_head' in n or '_bn1' in n or '_fc' in n:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        # 8번째 블록부터 학습 가능하게 설정
        for n, p in self.base_model.named_parameters():
            if '_blocks.7.' in n or '_blocks.8.' in n or \
                '_blocks.9.' in n or '_blocks.10.' in n or '_blocks.11.' in n or \
                '_blocks.12.' in n or '_blocks.13.' in n or '_blocks.14.' in n or \
                '_blocks.15.' in n or '_conv_head' in n or '_bn1' in n or '_fc' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, inputs):
        x = self.base_model.extract_features(inputs)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def plot_results(model_history_eff_net, mean_baseline: float):
    epochs_eff_net = range(len(model_history_eff_net['train_loss']))

    sns.set(style="darkgrid")

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_eff_net, model_history_eff_net['train_loss'], label='EfficientNet Training Loss')
    plt.plot(epochs_eff_net, model_history_eff_net['val_loss'], label='EfficientNet Validation Loss')
    plt.axhline(y=mean_baseline, color='r', linestyle='--', label='Mean Baseline')

    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAPE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("training_validation.png")
    plt.show()
    
# 대소문자를 무시하고 파일 존재 여부 확인 함수
def file_exists_ignore_case(path):
    directory, filename = os.path.split(path)
    if not os.path.exists(directory):
        return False
    for existing_file in os.listdir(directory):
        if existing_file.lower() == filename.lower():
            return os.path.join(directory, existing_file)
    return False


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
            folder_name = f"TS_{filename[:4]}"
            
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

def run(small_sample=False):
    image_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/01.image"
    json_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/02.label"
    
    df = prepare_data(image_base_path, json_base_path)

    if small_sample:
        df = df.iloc[0:1000]  # set small_sample to True if you want to check if your code works without long waiting
    train, val, test = split_data(df)  # split your data

    mean_baseline = get_mean_baseline(train, val)

    train_loader, val_loader, test_loader = create_dataloaders(
        df=df, train=train, val=val, test=test, plot_augmentations=True
    )

    eff_net_history = run_model(
        model_name="eff_net",
        model_function=CustomEfficientNet(),
        lr=0.001,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    plot_results(eff_net_history, mean_baseline)


if __name__ == "__main__":
    run(small_sample=False)
