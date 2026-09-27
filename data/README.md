# Dataset: CUB-200-2011 (Caltech-UCSD Birds-200-2011)

## 1. Final Dataset Decision
Dựa trên yêu cầu phân loại các nhóm đối tượng có vẻ ngoài giống nhau dưới điều kiện dữ liệu hạn chế, nhóm thống nhất chọn tập dữ liệu **CUB-200-2011**.

## 2. Dataset Source / Link
* **Trang chủ:** https://www.vision.caltech.edu/datasets/cub_200_2011/


## 3. Basic Dataset Statistics
* **Number of classes:** 200 (loài chim)
* **Number of images:** 11,788 ảnh tổng cộng (5,994 ảnh train, 5,794 ảnh test)
* **Class distribution:** Khá cân bằng, trung bình có khoảng 30 ảnh huấn luyện cho mỗi class.
## 4. Setup Instructions & DataLoader Usage

**Thiết lập môi trường (Đồng bộ Windows & Linux):**
Dự án sử dụng file `requirements.txt` để đồng bộ thư viện, giúp tránh xung đột giữa các hệ điều hành.

1. Khởi tạo và kích hoạt môi trường ảo:
   - **Windows:** Mở CMD/Terminal ở thư mục gốc và chạy 2 lệnh sau:
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Linux (Mint/Ubuntu):**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
2. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt