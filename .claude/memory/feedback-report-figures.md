---
name: feedback-report-figures
description: Report/slide: kiến trúc vẽ bằng draw.io theo style ảnh mẫu (pastel, viền đậm cùng màu, mũi tên nét đứt), kết quả để bảng
metadata:
  type: feedback
---

Report IEEE và slide (2026-11-14):
- Kiến trúc phải có hình. Vẽ bằng draw.io theo style ảnh mẫu người dùng gửi: box bo góc, nền pastel, viền đậm cùng
  màu, chữ đậm, phụ đề nghiêng; mũi tên nét đứt, đầu mở, nhãn nằm giữa đường; khung xám nhạt bao hệ thống; hình người
  cho actor; hình trụ cho kho dữ liệu.
- Block chỉ ghi tên, không ghi mô tả dưới tên. Phần diễn giải viết trong bài, không nhét vào hình. Không vẽ khung lớn
  bao cả hệ thống. Không dùng emoji (không hợp report nộp thầy): icon SVG đơn sắc trong huy hiệu tròn nền trắng ở góc
  block, ổ khóa = frozen, ngọn lửa = trọng số được cập nhật (gộp fine-tune và train from scratch, phân biệt trong text);
  Hình chỉ cho biết component nào dùng sẵn (frozen) và cái nào train/fine-tune; "training only", fine-tune với train
  from scratch khác nhau ra sao thì diễn giải trong report, không đánh dấu trên hình. Icon có màu (ổ khóa xanh, ngọn
  lửa cam), viền huy hiệu cùng màu. Legend chỉ có 2 icon, nhãn ngắn: "frozen", "trained / fine-tuned".
- Màu block chỉ để dễ nhìn và gom nhóm (tím input/output, xanh model pretrain, trắng module thêm, vàng fusion, xám kho
  dữ liệu), **không chú thích màu** trong legend.
- Input là "Audio" (biểu tượng sóng âm), không vẽ hình người. Một đường ra từ input rồi mới tách nhánh tại một chấm.
- Hình luồng chạy chỉ ghi kích thước tensor (dùng ký hiệu như T) và công thức; không ghi giá trị ví dụ ("e.g. 2.4 s",
  xác suất mẫu, câu transcript mẫu) hay hyper-parameter (16 kHz, ≤ 192 tokens): những thứ đó viết trong bài.
- Cần 2 loại hình: tổng quan chỉ gồm các module (high-level), và hình luồng chạy trên một ví dụ, có ghi kích thước hay
  phép tính.
- Kết quả trình bày bằng bảng, không vẽ biểu đồ.
- Sườn report lấy từ bài cũ `~/Desktop/nlp-ck/overleaf_nlp_ck/main.tex` (ACL), có cải thiện; người dùng nói bài đó
  không phải tốt nhất.

**Why:** người dùng muốn người đọc nhìn là biết hệ có những module nào; số liệu thì so sánh bằng bảng dễ hơn.
**How to apply:** toàn bộ quy ước đã đưa vào skill research-paper ("Style B" trong `references/figure-style.md`,
helper `scripts/drawio_system.py`, mẫu `examples/eaasr/figures.py`, ảnh mẫu `assets/figure-refs/06-07`), kèm checklist
6 mục phải soát trên PNG trước khi báo xong. Script của project: `paper-review/figures/make_arch_figures.py`; sửa kiến
trúc thì sửa script rồi chạy lại để export PNG. Liên quan: [[feedback-output-style]].
