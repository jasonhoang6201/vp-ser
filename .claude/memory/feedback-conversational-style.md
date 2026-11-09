---
name: feedback-conversational-style
description: Người dùng thích văn phong giải thích kiểu trò chuyện (hỏi đáp, ví von đời thường, xác nhận chỗ hiểu đúng rồi chỉnh chỗ sai) cho trang học
metadata:
  type: feedback
---

2026-11-04 người dùng dán một đoạn chat với Gemini về MFCC và nói "tôi thích cách dùng từ giống vậy, nó làm tôi thấy dễ hiểu hơn",
yêu cầu viết lại `demo/05-mfcc.html` theo văn phong đó, kèm diagram.

Những điểm họ thích: xưng "bạn", trả lời thẳng câu họ hỏi; xác nhận phần hiểu đúng trước rồi mới chỉnh phần sai, kèm tên thuật ngữ chuẩn;
ví von đời thường (vận tốc/gia tốc ô tô cho delta, "máy đo hình dáng" cho DCT, phễu cho tam giác Mel); ví dụ số nhỏ ([1000,100,10,1] → log → [3,2,1,0]);
nêu nguồn gốc ("ai nghĩ ra, lấy từ đâu"), và "13 là do thực nghiệm".

**Why:** văn bản khô kiểu giáo trình làm họ khó theo; kiểu hỏi đáp bám đúng chỗ họ vướng.
**How to apply:** trang học mới hoặc trang sửa lại thì dùng khối "Bạn hỏi:" cho các câu họ hay thắc mắc, ví von trước rồi tới thuật ngữ.
Ví von chỉ để giải thích, không dùng làm tên gọi của khái niệm (xem [[feedback-correct-terminology]]).
Không chép các khẳng định quá đà kiểu "99%", "95% thông tin", "thần thánh"; giữ đúng số liệu.
Vẫn theo [[feedback-problem-first]] và [[feedback-no-forward-refs]].
