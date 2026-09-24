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