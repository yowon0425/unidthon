import numpy as np

def zip_folder(folder_path, output_zip):
    shutil.make_archive(output_zip, 'zip', folder_path)
    print(f"Created {output_zip}.zip successfully.")

    

class CustomDatasetTest(data.Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])
        
        # Convert numpy array to PIL image
        if isinstance(noisy_image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path


test_transform = Compose([
    ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

model = Restormer()
model.load_state_dict(torch.load('best_Restormer.pth'))

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 데이터셋 경로
test_data_path = './open/test/input'
output_path = './open/test/submission'

# 데이터셋 로드 및 전처리
test_dataset = CustomDatasetTest(test_data_path, transform=test_transform)

# 데이터 로더 설정
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 이미지 denoising 및 저장
for noisy_image, noisy_image_path in test_loader:
    noisy_image = noisy_image.to(device)
    denoised_image = model(noisy_image)
    
    # denoised_image를 CPU로 이동하여 이미지 저장
    denoised_image = denoised_image.cpu().squeeze(0)
    denoised_image = (denoised_image * 0.5 + 0.5).clamp(0, 1)
    denoised_image = transforms.ToPILImage()(denoised_image)

    # Save denoised image
    output_filename = noisy_image_path[0]
    denoised_filename = output_path + '/' + output_filename.split('/')[-1][:-4] + '.jpg'
    denoised_image.save(denoised_filename) 
    
    print(f'Saved denoised image: {denoised_filename}')