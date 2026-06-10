# LiteEvo — Reframing: Speculative Scientific Discovery

> Tài liệu định hướng cách phát biểu bài toán và kể chuyện (problem framing & storytelling)
> cho LiteEvo, sau feedback của mentor. Đây **không** phải thay đổi thuật toán — chỉ là cách
> nghĩ và cách viết. Cảm hứng: *Fast Inference from Transformers via Speculative Decoding*
> (Leviathan et al., 2022 — https://arxiv.org/pdf/2211.17192).

---

## 0. Vì sao reframe

Câu chuyện cũ là một **tradeoff engineering**:

> "Chúng tôi làm automated algorithm design rẻ hơn ~8× mà vẫn near-SOTA."

Đúng, nhưng dưới mắt một nhà khoa học nó "tầm thường" vì nó chỉ nói *tôi tối ưu chi phí* — không
có một **nguyên lý** nào đằng sau. Mentor đề xuất mượn **tinh thần** của speculative decoding để
nâng nó từ một tradeoff thành một **nguyên tắc khám phá**.

Lưu ý quan trọng: mượn **idea**, không mượn **cơ chế**. Ta không tái hiện draft model /
rejection sampling / "phân phối đầu ra y hệt target". Khám phá thuật toán không có khái niệm
"phân phối target" để mà bằng — cố ép sẽ gượng. Chữ *speculative* ở đây là **kim chỉ nam ý niệm
xuyên suốt**, không phải một khuôn kỹ thuật để copy.

---

## 1. Hạt nhân (một câu)

> **Khám phá thuật toán không nên trả giá đắt đồng đều cho mọi bước — phần lớn bước là dễ, hãy
> đoán rẻ; chỉ trả giá đắt ở thiểu số bước thực sự khó.**

Dùng câu này để tự kiểm: mọi đoạn viết, mọi thành phần phải quy về được câu này. Đoạn nào
không phục vụ nó → cắt.

---

## 2. Tinh thần mượn từ speculative decoding

Speculative decoding có hai tầng. Ta chỉ lấy tầng trên:

- **Tầng dưới (KHÔNG mượn — cơ chế cụ thể):** draft model, rejection sampling, bảo đảm "phân
  phối y hệt target model".
- **Tầng trên (MƯỢN — ý tưởng tổng quát):**
  > *Đừng trả giá đắt đồng đều cho mọi bước. Đoán rẻ trước, rồi chỉ trả giá đắt ở những chỗ việc
  > đoán rẻ không đủ tốt.*

Điều này **tự nhiên đúng** với automated discovery — không phải gán ép — vì độ khó của các bước
khám phá vốn không đồng nhất.

---

## 3. Phát biểu bài toán mới (one-liner)

> Khám phá thuật toán bằng LLM tiêu tiền **đồng đều** cho mọi bước, dù phần lớn bước chỉ là tinh
> chỉnh cục bộ mà một mô hình rẻ thừa sức. LiteEvo theo một nguyên tắc **speculative**: *để mô
> hình rẻ đoán phần lớn các bước, và chỉ huy động mô hình mạnh ở những bước thực sự khó (đổi
> paradigm).* Nhờ đó nó đạt kết quả ngang các quy trình đắt với một phần nhỏ chi phí.

---

## 4. Mạch kể (storyline) — 4 nhịp

1. **Quan sát mở đầu (KHÔNG mở bằng "LLM đắt").**
   Trong tiến hóa chương trình bằng LLM, các bước cải tiến *không đồng nhất về độ khó*: đa số là
   tinh chỉnh cục bộ, thiểu số là bước nhảy paradigm. Nhưng các phương pháp hiện tại gọi cùng một
   mô hình mạnh cho **mọi** bước — trả giá đắt đồng đều cho một bài toán *không* đồng đều.

2. **Nguyên tắc speculative (kim chỉ nam).**
   Lấy cảm hứng từ tinh thần speculative decoding — *đoán rẻ trước, chỉ huy động sức mạnh khi việc
   đoán rẻ không đủ* — LiteEvo để một mô hình rẻ chủ động đề xuất phần lớn các bước, và dành mô
   hình frontier cho đúng những lúc cần một bước nhảy. (Nêu rõ một câu: ta mượn *tinh thần*, không
   tái hiện cơ chế; bài toán khám phá không có "phân phối target" để mà bằng.)

3. **Framework là hệ quả, không phải đóng góp tự thân.**
   Two-tier model, MAP-Elites, meta-advisor — trình bày từng cái như *câu trả lời tất yếu* cho
   nguyên tắc trên, không phải một rổ kỹ thuật rời rạc. (Xem mục 5.)

4. **Kết quả xác nhận nguyên tắc.**
   Ngang các quy trình đắt trên hai bài toán, ở ~⅛ chi phí → bằng chứng rằng "đa số bước vốn dễ"
   là một quan sát *đúng và khai thác được*, không phải may mắn.

---

## 5. Câu chuyện đằng sau từng lựa chọn

Mentor đòi giải thích rõ *vì sao* mỗi thành phần tồn tại. Khuôn trả lời: luôn dưới dạng
**"sự không-đồng-đều X ⇒ buộc phải có Y"**, KHÔNG BAO GIỜ dưới dạng "để tiết kiệm tiền".

| Thành phần | Câu chuyện đằng sau |
|---|---|
| **Two-tier model** (rẻ + frontier) | Vì độ khó các bước là **long-tailed**: dùng frontier cho bước dễ là lãng phí, dùng model rẻ cho bước khó là vô vọng. Hai tầng là phản ứng với *hình dạng* phân phối độ khó. |
| **MAP-Elites** | Vì mô hình rẻ đoán **trúng hơn nhiều** khi có một kho lời giải đa dạng đã kiểm chứng làm bệ phóng — đoán rẻ cần ngữ cảnh tốt. |
| **Meta-advisor** | Vì cần một tín hiệu *khi nào đoán rẻ đã cạn tác dụng* để biết lúc nào huy động frontier — đây là nơi tinh thần speculative thể hiện rõ nhất mà không cần copy rejection rule. |

---

## 6. Bảng đối chiếu cảm hứng (chỉ để định hướng tư duy — KHÔNG đưa nguyên vào paper)

| Speculative decoding | LiteEvo (tương tự về *tinh thần*) |
|---|---|
| Token "dễ" vs "khó" | Bước cải tiến **dễ** (tinh chỉnh cục bộ) vs **khó** (đổi paradigm) |
| Draft model (rẻ) đoán phần lớn | Mô hình rẻ đề xuất phần lớn mutation |
| Target model chỉ vào cuộc khi cần | Frontier model chỉ dùng cho bước nhảy paradigm |
| Bất đối xứng: verify rẻ hơn generate | Bất đối xứng *gắt hơn*: chạy/đo fitness rẻ hơn nhiều bậc so với gọi LLM frontier |

> Ranh giới: bảng này giúp **bạn** nghĩ. Trong paper chỉ giữ phần "lấy cảm hứng từ tinh thần
> speculative", **không** ánh xạ 1-1 từng thành phần (sẽ mời reviewer vặn "sao không giống").

---

## 7. Việc viết tiếp (không cần chạy lại thí nghiệm)

1. **Abstract & Intro:** mở bằng quan sát bất đối xứng độ khó (nhịp 1), không mở bằng "LLM đắt".
2. **Method section:** trình bày mỗi thành phần theo khuôn mục 5 ("X buộc phải có Y").
3. **Kết quả:** đóng khung con số near-SOTA thành "ngang dải các quy trình frontier ở ~⅛ chi phí"
   → bằng chứng cho nguyên tắc, không phải một tradeoff "thua một chút".
4. (Tùy chọn) Đặt một tên cơ chế trích dẫn được, ví dụ *Speculate-then-Escalate*, để bài có một
   danh từ riêng — nhưng chỉ nếu nó tự nhiên, không gượng.

---

## 8. Những điều TUYỆT ĐỐI tránh

- Đừng tuyên bố "no-regret / phân phối y hệt" — ta không có đối chứng frontier-only và discovery
  không có phân phối target. Tuyên bố mạnh hơn dữ liệu = mời reviewer bác.
- Đừng đổi tên cứng mọi thành phần thành proposer/verifier/rejection — đó là bắt chước *cơ chế*.
- Đừng mở bài bằng chi phí. Mở bằng **quan sát về cấu trúc bài toán** (độ khó không đồng đều).
