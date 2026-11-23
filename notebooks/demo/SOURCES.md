# Clip demo

Tất cả là 16 kHz, mono, 16 bit (bản Expresso chuyển từ 48 kHz bằng ffmpeg). Không clip nào thuộc IEMOCAP, nên
model của fold nào cũng chưa từng thấy chúng. Demo dùng model fold S5 (train trên session 1-4).

## Bộ demo (`clips/`)

Chọn từ bộ ứng viên bên dưới sau khi chạy model: 4 câu đúng, 2 câu sai nhưng giải thích được.

| File | Nguồn | Nhãn | Lời | Dự đoán (fold S5) | Ý cho demo |
|---|---|---|---|---|---|
| `01_joy_amazing_book.wav` | Expresso `ex01_happy_00049` | joy | "That was an amazing book! Some other books like it include The Last Mile and The Dream of Thieves." | joy 0.90 | Lời và giọng cùng vui |
| `02_anger_wonder.wav` | CREMA-D `1041_IWW_ANG_XX` | anger | "I wonder what this is about." | anger 0.97 | Nhánh text đoán neutral, gate dồn 0.79 cho audio |
| `03_neutral_surface.wav` | CREMA-D `1066_TSI_NEU_XX` | neutral | "The surface is slick." | neutral 0.77 | |
| `04_sadness_tomorrow.wav` | CREMA-D `1089_MTI_SAD_XX` | sadness | "Maybe tomorrow it will be cold." | sadness 0.78 | Cả hai nhánh cùng đoán buồn |
| `05_joy_airplane.wav` | CREMA-D `1074_TAI_HAP_XX` | joy | "The airplane is almost full." | anger 0.70 (sai) | Joy và anger cùng arousal cao: lỗi lệch miền phổ biến nhất |
| `06_sadvoice_amazing_book.wav` | Expresso `ex01_sad_00049` | sadness | giống clip 01 | joy 0.88 (sai) | Lời tích cực lấn giọng buồn; thêm ngữ cảnh buồn thì joy còn 0.69 |

## Bộ ứng viên (`candidates/`)

39 clip CREMA-D, mỗi lớp 10 (sadness 9 vì ít clip đạt ngưỡng), cộng 7 clip thử ban đầu. Cách chọn: từ
`finishedResponses.csv`, chỉ lấy lượt nghe không có hình (`queryType = 1`), giữ clip có ít nhất 8 lượt nghe và tỉ lệ
người nghe đoán đúng nhãn gốc ≥ 0.8, mức cường độ XX hoặc HI, mỗi diễn viên tối đa một clip mỗi lớp, mỗi câu tối đa
hai clip mỗi lớp, chọn ngẫu nhiên (seed 42). Tỉ lệ đồng ý từng clip ở cột `listener_agreement` của `candidates.csv`.

Kết quả fold S5 trên 39 clip CREMA-D: **đúng 23/39 (59%)**.

| Nhãn \ Dự đoán | anger | joy | neutral | sadness |
|---|---|---|---|---|
| anger | 10 | 0 | 0 | 0 |
| joy | 7 | 1 | 1 | 1 |
| neutral | 0 | 0 | 8 | 2 |
| sadness | 0 | 0 | 5 | 4 |

Hai kiểu lỗi chính đều là nhầm giữa hai lớp cùng mức arousal: joy thành anger (cao) và sadness thành neutral
(thấp). CREMA-D là câu đọc theo kịch bản với lời trung tính, khác hội thoại ứng tác của IEMOCAP, nên phần lớn thông
tin cảm xúc phải đến từ nhánh audio.

## Nguồn và license

- **CREMA-D**: Cao et al., "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset", IEEE Trans. Affective
  Computing, 2014. https://github.com/CheyneyComputerScience/CREMA-D, Open Database License (ODbL 1.0),
  nội dung theo Database Contents License.
- **Expresso**: Nguyen et al., "EXPRESSO: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis",
  Interspeech 2023. Bản HF `ylacombe/expresso` (phần `read`), CC BY-NC 4.0, chỉ dùng phi thương mại.
