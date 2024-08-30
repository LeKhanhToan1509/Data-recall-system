# Data Recall System

## Giới thiệu

**Data Recall System** là một hệ thống phức hợp được thiết kế để quản lý dữ liệu, cảnh báo dữ liệu, quản lý phiên bản dữ liệu (data versioning), tự động gán nhãn (auto label), và huấn luyện mô hình học máy. Hệ thống này sử dụng nhiều công nghệ hiện đại nhằm tối ưu hóa hiệu suất, khả năng mở rộng và tính linh hoạt trong việc xử lý dữ liệu lớn.

## Các tính năng chính

- **Quản lý dữ liệu và phiên bản**: Hệ thống hỗ trợ quản lý dữ liệu và các phiên bản của chúng một cách hiệu quả, đảm bảo tính toàn vẹn và lịch sử của dữ liệu.
- **Cảnh báo dữ liệu**: Hệ thống có khả năng phát hiện các bất thường trong dữ liệu và đưa ra cảnh báo kịp thời, giúp ngăn ngừa các vấn đề có thể xảy ra trong quá trình xử lý và phân tích dữ liệu.
- **Tự động gán nhãn**: Hệ thống hỗ trợ tự động gán nhãn cho dữ liệu, giảm thiểu thời gian và công sức cần thiết cho việc chuẩn bị dữ liệu huấn luyện.
- **Huấn luyện mô hình**: Hệ thống tích hợp các công cụ huấn luyện mô hình học máy, hỗ trợ việc xây dựng và cải tiến các mô hình dựa trên dữ liệu được quản lý.

## Kiến trúc thư mục

Hệ thống bao gồm các thư mục chính sau:

### 1. `app/database`
- **database.py**: Quản lý kết nối và tương tác với cơ sở dữ liệu.
- **models.py**: Định nghĩa các mô hình dữ liệu.

### 2. `app/deepstream`
- **ds_run/**: Chứa các mã nguồn để chạy các pipeline của DeepStream.
- **models/**: Chứa các mô hình sử dụng trong DeepStream.
- **test_ds/**: Các tập tin kiểm thử cho DeepStream.
- **videos/**: Lưu trữ video đầu vào và kết quả đầu ra của DeepStream.

### 3. `app/deepstream_app`
- **deepstream_python_apps/**: Chứa các ứng dụng Python chạy trong môi trường DeepStream.
- **ds_run/**: Tương tự như trong `deepstream`, thư mục này chứa mã nguồn để chạy các pipeline.
- **models/**: Chứa các mô hình được sử dụng trong ứng dụng DeepStream.
- **test/**: Tập tin kiểm thử cho các ứng dụng DeepStream.
- **videos/**: Video đầu vào và kết quả xử lý của DeepStream.

### 4. `app/docker`
- **TriggerDockerfile**: Dockerfile được sử dụng để xây dựng các images liên quan đến trigger.

### 5. `app/etl`
- **cvat_tool.py**: Công cụ CVAT để gán nhãn dữ liệu hình ảnh.
- **image_quality_v2.py**: Mã xử lý và kiểm tra chất lượng hình ảnh phiên bản 2.
- **image_quality.py**: Kiểm tra chất lượng hình ảnh.
- **label_quality.py**: Kiểm tra chất lượng nhãn dữ liệu.

### 6. `app/pipeline`
- **pipeline_autolabel.py**: Pipeline tự động gán nhãn dữ liệu.
- **pipeline_yolo.py**: Pipeline liên quan đến YOLO, một mô hình học sâu cho nhiệm vụ phát hiện đối tượng.

### 7. `app/requirements`
- **trigger-requirements.txt**: Danh sách các thư viện và phụ thuộc cần thiết cho hệ thống trigger.

### 8. `app/serverless/task`
- **start_serverless.sh**: Script khởi động các chức năng không máy chủ (serverless functions).

### 9. `app/storage`
- **main_storage.py**: Mã liên quan đến lưu trữ dữ liệu chính trong hệ thống.

## Công nghệ sử dụng

### 1. Docker
- **Docker** được sử dụng để container hóa các dịch vụ, đảm bảo tính nhất quán và dễ dàng triển khai giữa các môi trường khác nhau.

### 2. Serverless Framework
- **Serverless Framework** giúp quản lý các chức năng không máy chủ (serverless functions), cung cấp khả năng mở rộng linh hoạt và tối ưu chi phí.

### 3. CVAT (Computer Vision Annotation Tool)
- **CVAT** là công cụ dùng để gán nhãn dữ liệu hình ảnh, hỗ trợ cho việc chuẩn bị dữ liệu huấn luyện cho các mô hình thị giác máy tính.

### 4. VLM (Vision-Language Model)
- **VLM** là một mô hình kết hợp giữa thị giác và ngôn ngữ, giúp hệ thống có khả năng hiểu và phân tích dữ liệu phức tạp từ cả hình ảnh và văn bản.

### 5. TelegramBot
- **TelegramBot** được sử dụng để tạo ra giao diện giao tiếp với người dùng cuối, cho phép gửi và nhận thông tin, yêu cầu, và cảnh báo trực tiếp qua Telegram.

### 6. Apache Spark
- **Apache Spark** là nền tảng xử lý dữ liệu phân tán, giúp tăng tốc độ xử lý và phân tích dữ liệu lớn, đồng thời hỗ trợ việc huấn luyện mô hình học máy trên quy mô lớn.

### 7. Apache Kafka
- **Apache Kafka** được sử dụng để xử lý các luồng dữ liệu thời gian thực, đảm bảo việc truyền tải và xử lý dữ liệu một cách liên tục và ổn định.

### 8. Redis
- **Redis** là một hệ thống lưu trữ dữ liệu trong bộ nhớ (in-memory data store), được sử dụng để lưu trữ các phiên bản dữ liệu tạm thời, cache và xử lý các tác vụ yêu cầu truy cập nhanh.

## Cài đặt và triển khai

### 1. Yêu cầu hệ thống
- Docker
- Docker Compose
- Serverless Framework
- Python 3.x

### 2. Cài đặt
Clone repository về máy của bạn:
```bash
git clone https://github.com/your_username/data-recall-system.git
cd data-recall-system
