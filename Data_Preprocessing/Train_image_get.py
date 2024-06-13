# import os
# import matplotlib.pyplot as plt
# from PIL import Image
# import pandas as pd
# # base_path 설정
# base_path = "/home/jaewoong/Desktop/118.안면 인식 에이징(aging) 이미지 데이터/01-1.정식개방데이터/Training/01.원천데이터"

# # 이미지 파일 경로를 저장할 리스트
# image_files = []

# # 모든 TS 폴더를 탐색하여 이미지 파일 경로를 수집
# for root, dirs, files in os.walk(base_path):
#     for file in files:
#         if file.endswith('.png'):
#             image_files.append(os.path.join(root, file))

# # 이미지 파일 경로 출력 (예: 첫 5개 파일)
# print(image_files)

# # 모든 이미지 파일 읽기 및 표시 (예: 첫 5개 파일만)
# for image_path in image_files[:5]:
#     image = Image.open(image_path)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.show()

# # 이미지 파일 경로와 파일명 정보를 데이터프레임으로 저장
# data = [{'image_location': image_path, 'filename': os.path.basename(image_path)} for image_path in image_files]
# df = pd.DataFrame(data)

# # 데이터프레임 출력 (예: 첫 5개 행)
# print(df.head())


# import os
# import json
# import pandas as pd

# # 루트 경로 설정
# image_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/01.image"
# json_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/02.label"

# # 모든 이미지 파일 경로와 JSON 파일 경로 수집
# image_files = []
# json_files = []

# # 이미지 파일 경로 수집
# for root, dirs, files in os.walk(image_base_path):
#     for file in files:
#         if file.endswith('.png'):
#             image_files.append(os.path.join(root, file))

# # JSON 파일 경로 수집
# for root, dirs, files in os.walk(json_base_path):
#     for file in files:
#         if file.endswith('.json'):
#             json_files.append(os.path.join(root, file))

# print(f'Found {len(image_files)} image files.')
# print(f'Found {len(json_files)} JSON files.')

# # 필요한 정보 추출
# data = []
# for json_file in json_files:
#     with open(json_file, 'r') as file:
#         json_data = json.load(file)
#         filename = json_data['filename']
#         age_past = json_data['age_past']
#         # 중간 폴더명(TS_*)과 파일명을 조합하여 이미지 경로 재구성
#         folder_name = f"TS_{filename[:4]}"  # 파일명에서 앞 4자리 추출 (예: 0867)
#         image_path = os.path.join(image_base_path, folder_name, filename + ".png")
#         data.append({'image_location': image_path, 'filename': filename, 'age': age_past})

# # 데이터프레임 생성
# df = pd.DataFrame(data)

# # 데이터프레임 출력 (예: 첫 5개 행)
# print(df)

# import os
# import json
# import pandas as pd

# # 대소문자를 무시하고 파일 존재 여부 확인 함수
# def file_exists_ignore_case(path):
#     directory, filename = os.path.split(path)
#     if not os.path.exists(directory):
#         return False
#     for existing_file in os.listdir(directory):
#         if existing_file.lower() == filename.lower():
#             return os.path.join(directory, existing_file)
#     return False

# # 데이터 준비 함수
# def prepare_data(image_base_path, json_base_path):
#     image_files = []
#     json_files = []

#     # 이미지 파일 경로 수집
#     for root, dirs, files in os.walk(image_base_path):
#         for file in files:
#             if file.endswith('.png'):
#                 image_files.append(os.path.join(root, file))

#     # JSON 파일 경로 수집
#     for root, dirs, files in os.walk(json_base_path):
#         for file in files:
#             if file.endswith('.json'):
#                 json_files.append(os.path.join(root, file))

#     print(f'Found {len(image_files)} image files.')
#     print(f'Found {len(json_files)} JSON files.')

#     data = []
#     missing_files = []
#     for json_file in json_files:
#         with open(json_file, 'r') as file:
#             json_data = json.load(file)
#             filename = json_data['filename']
#             age_past = json_data['age_past']
#             folder_name = f"TS_{filename[:4]}"

#             if filename.lower().endswith('.png'):
#                 filename = filename[:-4]

#             image_path = os.path.join(image_base_path, folder_name, filename) + ".png"
            
#             # 대소문자를 무시하고 파일 존재 여부 확인
#             actual_path = file_exists_ignore_case(image_path)
#             if actual_path:
#                 data.append({'image_location': actual_path, 'filename': filename, 'age': age_past})
#             else:
#                 # 두 번째 시도: 중복된 확장자 제거
#                 corrected_image_path = os.path.join(image_base_path, folder_name, filename.split('.')[0]) + ".png"
#                 actual_path = file_exists_ignore_case(corrected_image_path)
#                 if actual_path:
#                     data.append({'image_location': actual_path, 'filename': filename, 'age': age_past})
#                 else:
#                     missing_files.append(image_path)

#     print(f'Missing files: {len(missing_files)}')
#     for missing_file in missing_files:
#         print(missing_file)
#     df = pd.DataFrame(data)
#     return df

# # 루트 경로 설정
# image_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/01.image"
# json_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/02.label"

# # 데이터 준비
# df = prepare_data(image_base_path, json_base_path)

# # 데이터프레임 출력 (예: 첫 5개 행)
# print(df.head())
# print(f'Total matched files: {len(df)}')

# import os
# import json
# import pandas as pd

# # 대소문자를 무시하고 파일 존재 여부 확인 함수
# def file_exists_ignore_case(path):
#     directory, filename = os.path.split(path)
#     if not os.path.exists(directory):
#         return False
#     for existing_file in os.listdir(directory):
#         if existing_file.lower() == filename.lower():
#             return os.path.join(directory, existing_file)
#     return False

# # 데이터 준비 함수
# def prepare_data(image_base_path, json_base_path):
#     image_files = []
#     json_files = []

#     # 이미지 파일 경로 수집
#     for root, dirs, files in os.walk(image_base_path):
#         for file in files:
#             if file.endswith('.png'):
#                 image_files.append(os.path.join(root, file))

#     # JSON 파일 경로 수집
#     for root, dirs, files in os.walk(json_base_path):
#         for file in files:
#             if file.endswith('.json'):
#                 json_files.append(os.path.join(root, file))

#     print(f'Found {len(image_files)} image files.')
#     print(f'Found {len(json_files)} JSON files.')

#     data = []
#     missing_files = []
#     for json_file in json_files:
#         with open(json_file, 'r') as file:
#             json_data = json.load(file)
#             filename = json_data['filename']
#             age_past = json_data['age_past']
#             folder_name = f"TS_{filename[:4]}"
            
#             # 확장자가 이미 포함되어 있는지 확인하고 제거
#             if filename.lower().endswith('.png'):
#                 filename = filename[:-4]
                
#             image_path = os.path.join(image_base_path, folder_name, filename) + ".png"
            
#             # 파일 존재 여부 확인 (대소문자 무시)
#             actual_path = file_exists_ignore_case(image_path)
#             if actual_path:
#                 data.append({'image_location': actual_path, 'filename': filename, 'age': age_past})
#             else:
#                 missing_files.append(image_path)

#     print(f'Missing files: {len(missing_files)}')
#     for missing_file in missing_files:
#         print(missing_file)
#     df = pd.DataFrame(data)
#     return df

# # 루트 경로 설정
# image_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/01.image"
# json_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/02.label"

# # 데이터 준비
# df = prepare_data(image_base_path, json_base_path)

# # 데이터프레임 출력 (예: 첫 5개 행)
# print(df.head())
# print(f'Total matched files: {len(df)}')

# import os
# import json
# import pandas as pd
# from PIL import Image, ImageFile

# # PIL 경고 무시 설정
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None

# # 대소문자를 무시하고 파일 존재 여부 확인 함수
# def file_exists_ignore_case(path):
#     directory, filename = os.path.split(path)
#     if not os.path.exists(directory):
#         return False
#     for existing_file in os.listdir(directory):
#         if existing_file.lower() == filename.lower():
#             return os.path.join(directory, existing_file)
#     return False

# # 이미지 파일 형식 확인 함수
# def is_png_file(file_path):
#     try:
#         with Image.open(file_path) as img:
#             return img.format == 'PNG'
#     except Exception as e:
#         print(f"Error checking file {file_path}: {e}")
#         return False

# # 데이터 준비 함수
# def prepare_data(image_base_path, json_base_path):
#     image_files = []
#     json_files = []

#     # 이미지 파일 경로 수집
#     for root, dirs, files in os.walk(image_base_path):
#         for file in files:
#             if file.endswith('.png'):
#                 image_files.append(os.path.join(root, file))

#     # JSON 파일 경로 수집
#     for root, dirs, files in os.walk(json_base_path):
#         for file in files:
#             if file.endswith('.json'):
#                 json_files.append(os.path.join(root, file))

#     print(f'Found {len(image_files)} image files.')
#     print(f'Found {len(json_files)} JSON files.')

#     data = []
#     missing_files = []
#     non_png_files = []
#     for json_file in json_files:
#         with open(json_file, 'r') as file:
#             json_data = json.load(file)
#             filename = json_data['filename']
#             age_past = json_data['age_past']
#             folder_name = f"TS_{filename[:4]}"
            
#             # 확장자가 이미 포함되어 있는지 확인하고 제거
#             if filename.lower().endswith('.png'):
#                 filename = filename[:-4]
                
#             image_path = os.path.join(image_base_path, folder_name, filename) + ".png"
            
#             # 파일 존재 여부 확인 (대소문자 무시)
#             actual_path = file_exists_ignore_case(image_path)
#             if actual_path:
#                 # 이미지가 실제 PNG 형식인지 확인
#                 if is_png_file(actual_path):
#                     data.append({'image_location': actual_path, 'filename': filename, 'age': age_past})
#                 else:
#                     non_png_files.append(actual_path)
#             else:
#                 missing_files.append(image_path)

#     print(f'Missing files: {len(missing_files)}')
#     for missing_file in missing_files:
#         print(missing_file)

#     print(f'Non-PNG files: {len(non_png_files)}')
#     for non_png_file in non_png_files:
#         print(non_png_file)

#     df = pd.DataFrame(data)
#     return df

# # 루트 경로 설정
# image_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/01.image"
# json_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/02.label"

# # 데이터 준비
# df = prepare_data(image_base_path, json_base_path)

# # 데이터프레임 출력 (예: 첫 5개 행)
# print(df.head())
# print(f'Total matched files: {len(df)}')

import os
import json
import pandas as pd
from PIL import Image, ImageFile

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

# 이미지 파일 형식 확인 함수
def is_png_file(file_path):
    try:
        with Image.open(file_path) as img:
            return img.format == 'PNG'
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
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
    non_png_files = []
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
                # 이미지가 실제 PNG 형식인지 확인
                if is_png_file(actual_path):
                    data.append({'image_location': actual_path, 'filename': filename, 'age': age_past})
                else:
                    non_png_files.append(actual_path)
            else:
                missing_files.append(image_path)

    print(f'Missing files: {len(missing_files)}')
    for missing_file in missing_files:
        print(missing_file)

    print(f'Non-PNG files: {len(non_png_files)}')
    for non_png_file in non_png_files:
        print(non_png_file)

    df = pd.DataFrame(data)
    return df, non_png_files

# 이미지 파일을 PNG 형식으로 변환하는 함수
def convert_to_png(file_path):
    try:
        with Image.open(file_path) as img:
            png_path = file_path
            img.save(png_path, 'PNG')
            print(f"Converted {file_path} to PNG format.")
    except Exception as e:
        print(f"Error converting file {file_path}: {e}")

# 루트 경로 설정
# image_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/01.image"
# json_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Training/02.label"
image_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Validation/01.image"
json_base_path = "/home/jaewoong/Desktop/Face_age_dataset/Validation/02.label"
# 데이터 준비
df, non_png_files = prepare_data(image_base_path, json_base_path)

# PNG 형식으로 변환
for non_png_file in non_png_files:
    convert_to_png(non_png_file)

# 데이터프레임 출력 (예: 첫 5개 행)
print(df.head())
print(f'Total matched files: {len(df)}')
