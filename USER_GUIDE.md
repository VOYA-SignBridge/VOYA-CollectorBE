# Hướng dẫn sử dụng (User)

Tài liệu này dành cho **người dùng cuối** (người thu thập dữ liệu / người thử nhận diện). Không yêu cầu biết lập trình.

## 1) Bạn cần chuẩn bị

- Máy tính có webcam.
- Trình duyệt: Chrome hoặc Edge (khuyến nghị).
- Được cấp URL của hệ thống (ví dụ trong nội bộ: `http://localhost:5173`).

## 2) Mở hệ thống

1. Mở trình duyệt.
2. Truy cập link do người vận hành cung cấp.
3. Cho phép quyền **Camera** khi trình duyệt hỏi.

Nếu thấy thông báo không kết nối được backend, báo người vận hành (mục “Xử lý sự cố”).

## 3) Thu thập dữ liệu (quay và gửi mẫu)

Quy trình chuẩn (mỗi từ/ký hiệu là một “label”):

1. Nhập **User** (tên người thu).
2. Chọn/nhập **Label** (tên từ/ký hiệu).
3. (Nếu có) chọn **Dialect** (Bắc/Trung/Nam/Common…).
4. Bấm bắt đầu ghi (hoặc mở màn hình ghi toàn màn hình nếu có nút Fullscreen).
5. Thực hiện động tác ký hiệu trước camera.
6. Dừng ghi.
7. Bấm **Upload/Gửi** (nếu hệ thống không tự gửi).

Gợi ý để dữ liệu tốt:
- Tay nằm trọn trong khung hình, đủ sáng.
- Giữ khoảng cách ổn định, tránh rung mạnh.
- Một mẫu nên có phần **vào động tác → thực hiện → kết thúc** (không chỉ đứng yên).

## 4) Nhận diện realtime (trang /realtime)

Trang này dùng để **thử nhận diện** từ mô hình hiện tại.

1. Vào trang “Realtime recognition” (đường dẫn thường là `/realtime`).
2. Bấm mở camera.
3. Thực hiện ký hiệu.
4. Hệ thống sẽ hiển thị:
   - Từ/ký hiệu dự đoán (kết quả ổn định nhất gần đây)
   - Độ tin cậy (confidence)

### Nút “Từ tiếp theo”

Khi bạn muốn nhận diện **từ mới** ngay sau đó:
- Bấm **Từ tiếp theo** để xóa cửa sổ dữ liệu hiện tại nhưng **không cần reload camera**.

### Khi không thấy tay

Nếu hiện “Không phát hiện tay…”:
- Đưa tay vào khung hình
- Tăng ánh sáng
- Kiểm tra camera có bị che

## 5) Xử lý sự cố nhanh

### Không mở được camera
- Kiểm tra trình duyệt đã được cấp quyền camera.
- Đóng các ứng dụng khác đang dùng camera (Zoom/Teams…).
- Thử refresh trang.

### Dự đoán sai/"đứng yên cũng đoán"
- Thử thực hiện động tác rõ ràng hơn.
- Bấm “Từ tiếp theo” trước khi làm từ mới.
- Nếu vấn đề xảy ra thường xuyên, báo người vận hành để họ kiểm tra mô hình/threshold.

### Không kết nối được backend / báo lỗi mạng
- Báo người vận hành. Thường do backend chưa chạy hoặc URL sai.

---

Nếu bạn cần “Hướng dẫn cho người vận hành” (cài đặt/chạy hệ thống), xem tài liệu [OPERATOR_GUIDE.md](OPERATOR_GUIDE.md).
