---
name: feedback-memory-location
description: Lưu memory và kiến thức research trong .claude/ của project (không phải ~/.claude) để push GitHub
metadata:
  type: feedback
---

Mọi memory ghi vào `<project>/.claude/memory/`, research ghi vào `<project>/.claude/research/`.
`.claude/` chỉ chứa memory và research. File HTML (overview, slide, ...) đặt ở thư mục project bình thường, không đặt trong `.claude/`.

**Why:** người dùng push thư mục `.claude` lên GitHub để dùng trên nhiều máy; người dùng nói rõ "html thì đặt trong folder bình thường thôi, .claude chỉ lưu memory và research".
**How to apply:** không ghi memory của project này vào `~/.claude/projects/...`; cập nhật `MEMORY.md` trong repo; output HTML ghi ra root project.
Liên quan: [[reference-research-kb]].
