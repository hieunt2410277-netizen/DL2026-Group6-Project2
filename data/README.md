# Dataset: CUB-200-2011 (Caltech-UCSD Birds-200-2011)

## 1. Final Dataset Decision
Dựa trên yêu cầu phân loại các nhóm đối tượng có vẻ ngoài giống nhau dưới điều kiện dữ liệu hạn chế, nhóm thống nhất chọn tập dữ liệu **CUB-200-2011**.

## 2. Dataset Source / Link
* **Trang chủ:** https://www.vision.caltech.edu/datasets/cub_200_2011/
* **Cách tải (dành cho team):** Sử dụng `torchvision.datasets.CUB200` trong PyTorch.

## 3. Basic Dataset Statistics
* **Number of classes:** 200 (loài chim)
* **Number of images:** 11,788 ảnh tổng cộng (5,994 ảnh train, 5,794 ảnh test)
* **Class distribution:** Khá cân bằng, trung bình có khoảng 30 ảnh huấn luyện cho mỗi class.

## 4. Setup Instructions & DataLoader Usage

**Thiết lập môi trường (Windows & Linux):**
Dự án sử dụng file `requirements.txt` để đồng bộ thư viện tránh xung đột hệ điều hành.
1. Khởi tạo và kích hoạt môi trường ảo:
   - **Windows:** `python -m venv venv` rồi chạy `venv\Scripts\activate`
   - **Linux:** `python3 -m venv venv` rồi chạy `source venv/bin/activate`
2. Cài đặt toàn bộ thư viện đồng bộ: `pip install -r requirements.txt`

**Cách gọi DataLoader trong các tập lệnh huấn luyện:**
```python
from src.data.dataset import get_dataloaders

# Khởi tạo DataLoader mặc định (Batch=32, dùng toàn bộ Train)
train_loader, val_loader, classes = get_dataloaders(data_dir='./data')

# Khởi tạo DataLoader cho thực nghiệm Limited Data (Ví dụ: chỉ dùng 10%)
limited_train, val_loader, _ = get_dataloaders(data_dir='./data', limited_data_ratio=0.1)