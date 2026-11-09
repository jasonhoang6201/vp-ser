# Project: Emotion-Aware Speech Recognition (bài cuối kì Voice Processing, thạc sĩ)

## Memory

Memory của project nằm trong repo để push GitHub và dùng trên nhiều máy:
`.claude/memory/`. Đọc `MEMORY.md` ở đầu mỗi session. Khi cần lưu memory mới, ghi
vào thư mục này (không ghi vào `~/.claude/projects/...`), mỗi file một fact, và thêm
một dòng vào `MEMORY.md`.

@memory/MEMORY.md

## Kiến thức research

- `.claude/research/papers.md`: kho paper, dataset, metric, phát hiện chính (có link).

Khi người dùng muốn đi sâu một phần, đọc `papers.md` trước, rồi bổ sung research mới
vào đúng file đó (hoặc tạo file con trong `.claude/research/`) để giữ làm reference.

## Output (HTML)

File HTML để xem/trình bày đặt ở thư mục project (ngoài `.claude/`). `.claude/` chỉ chứa
memory và research.

- `overview.html`: trang tổng quan có diagram SVG (light, minimal).
- `slides.html`: bản trình chiếu dạng slide của overview (phím ← →, F toàn màn hình, Cmd+P ra PDF).
