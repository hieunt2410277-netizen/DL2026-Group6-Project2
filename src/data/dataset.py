import os
import torch
import tarfile
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def download_and_extract(data_dir='./data'):
    """Tải và giải nén CUB-200-2011 với User-Agent giả lập trình duyệt."""
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, 'CUB_200_2011.tgz')
    extract_path = os.path.join(data_dir, 'CUB_200_2011')

    # Chỉ chạy quá trình tải hoặc bung nén NẾU thư mục ảnh chưa tồn tại
    if not os.path.exists(extract_path):
        
        # Nếu chưa có cả thư mục ảnh lẫn file nén -> Tiến hành tải
        if not os.path.exists(tgz_path):
            print("Đang kết nối để tải tập dữ liệu CUB-200-2011...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            with urllib.request.urlopen(req) as response, open(tgz_path, 'wb') as out_file:
                file_size = int(response.getheader('Content-Length', 0))
                print(f"Bắt đầu tải file (Kích thước: khoảng {file_size / (1024*1024):.1f} MB). Vui lòng đợi...")
                out_file.write(response.read())
            print("Tải xong!")
        
        # Đã có file nén (do tải tay hoặc tải code xong) -> Tiến hành giải nén
        print("Đang giải nén dữ liệu. Bạn đợi chút nhé...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("Giải nén hoàn tất!")
        # Tùy chọn: Xóa file tgz sau khi giải nén để đỡ tốn ổ cứng
        # os.remove(tgz_path) 
    
    return os.path.join(extract_path, 'images')

def get_dataloaders(data_dir='./data', batch_size=32, val_split=0.2, limited_data_ratio=1.0, num_workers=2):
    """Chuẩn bị dataset, Data Augmentation và chia DataLoader cố định."""
    img_dir = download_and_extract(data_dir)

    # 1. Móc tiền xử lý (Preprocessing hooks)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Khởi tạo Dataset
    full_dataset = datasets.ImageFolder(root=img_dir, transform=train_transform)

    # 3. Chia tập Train/Val cố định (No data leakage)
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Val dataset không dùng data augmentation
    val_ds.dataset.transform = val_test_transform

    # 4. Hỗ trợ mô phỏng thiếu dữ liệu (Limited-data sampling)
    if limited_data_ratio < 1.0:
        limited_size = int(len(train_ds) * limited_data_ratio)
        indices = torch.randperm(len(train_ds), generator=generator)[:limited_size].tolist()
        train_ds = Subset(train_ds, indices)

    # 5. Khởi tạo DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, full_dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, classes = get_dataloaders(limited_data_ratio=0.5)
    print(f"\nTổng số classes: {len(classes)}")
    for images, labels in train_loader:
        print(f"Kích thước Batch ảnh: {images.shape} | Batch nhãn: {labels.shape}")
        break