# Workflow của FORE

Tài liệu này mô tả luồng hoạt động chi tiết của phương pháp **FORE - Fertility-Oriented Reflective Evolution**, dựa trên `FORE_METHOD_PLAN.md`. Mục tiêu là giải thích cách phương pháp vận hành ở mức hệ thống và thuật toán, không đi vào code triển khai.

## 1. Ý tưởng trung tâm

FORE thay đổi câu hỏi chọn parent trong evolutionary search:

- Cách phổ biến: chọn chương trình đang có fitness tốt nhất, hoặc chọn theo UCB/đa dạng ở mức island.
- FORE: chọn chương trình có khả năng sinh ra hậu duệ tốt nhất trong các bước tiếp theo.

Khái niệm chính là **Posterior Offspring Value** (POV): giá trị kỳ vọng của một parent nếu tiếp tục mở rộng vùng tìm kiếm quanh parent đó. Vì vậy, một chương trình hiện tại chưa phải tốt nhất vẫn có thể được chọn nếu lịch sử cho thấy nó thường sinh ra child cải thiện mạnh, hoặc nó nằm ở một lineage còn mới và đáng khám phá.

FORE có 3 thành phần cốt lõi:

1. **Fertility estimation**: ước lượng khả năng sinh offspring tốt của từng program bằng posterior Bayesian trên improvement dương.
2. **Thompson sampling parent selection**: chọn parent bằng cách lấy mẫu từ posterior, cân bằng exploration và exploitation.
3. **Reflective Review**: khi search bị kẹt, LLM đọc fertility map và viết lại chiến lược tìm kiếm cho vài generation tiếp theo.

## 2. Các trạng thái chính trong hệ thống

Mỗi program trong archive cần mang thêm thông tin FORE:

- **fitness**: điểm đánh giá từ evaluator.
- **parent id**: program cha đã sinh ra candidate này.
- **strategy description**: mô tả ngắn về hướng tiếp cận thuật toán của candidate.
- **hypothesis**: giả thuyết vì sao hướng này có thể tốt.
- **diff from parent**: candidate khác parent ở điểm nào.
- **verdict**: kết quả sau khi đánh giá, ví dụ improved, regressed, stepping stone, dead end.
- **cluster id**: lineage/strategy cluster mà program thuộc về.
- **fertility stats**: thống kê hậu duệ của program, gồm số child đã thử, tổng improvement dương, tổng bình phương improvement dương, số lần regression, novelty và rarity.

Ở mức database, FORE duy trì thêm:

- **program archive**: toàn bộ candidate còn đang được giữ.
- **fertility map**: thống kê fertility theo từng program và từng strategy cluster.
- **cluster map**: nhóm các program có strategy description/code tương tự.
- **recent improvements**: cửa sổ improvement gần đây để phát hiện stagnation.
- **active review**: reflective review đang được inject vào prompt, nếu có.
- **review history**: các review đã dùng, để tránh lặp chiến lược cũ.

## 3. Luồng tổng thể khi chạy FORE

### 3.1 Khởi tạo run

Runner đọc config và tạo các thành phần chính:

1. Load initial program và evaluator.
2. Load config FORE, bao gồm population size, prior Bayesian, tham số cluster, trigger review và số context program.
3. Tạo `FOREDatabase`.
4. Tạo `FOREController`.
5. Tạo context builder chuyên cho FORE.
6. Khởi tạo LLM pool cho generation và guide LLM cho reflective review.
7. Nếu không resume từ checkpoint, evaluate initial program và thêm vào database.

Sau bước này, database có ít nhất một seed program. Program seed chưa có child nên fertility posterior còn rộng, tức vẫn có khả năng được chọn nhờ uncertainty.

### 3.2 Vòng lặp mỗi iteration

Mỗi iteration của FORE đi qua các bước sau:

1. Kiểm tra stagnation.
2. Nếu cần, chạy Reflective Review.
3. Chọn parent bằng POV và Thompson sampling.
4. Chọn context program bổ trợ.
5. Xây prompt cho LLM.
6. LLM sinh candidate mới.
7. Parse candidate và strategy metadata.
8. Evaluate candidate.
9. Cập nhật archive, fertility stats, cluster và verdict.
10. Checkpoint/logging.

Các bước này lặp đến khi hết budget iteration, early stopping, hoặc user dừng run.

## 4. Bước chọn parent bằng POV

Đây là phần khác biệt nhất của FORE.

### 4.1 Thu thập dữ liệu cho từng parent

Với mỗi program trong archive, FORE lấy:

- Fitness hiện tại của program.
- Fertility stats từ các child từng được sinh ra từ program đó.
- Novelty score: program khác các top neighbor đến mức nào.
- Cluster rarity: strategy cluster của program hiếm hay phổ biến.
- Age: program đã tồn tại bao lâu.
- K remaining: budget tìm kiếm còn lại.

Improvement được tính theo parent-child:

Delta = fitness(child) - fitness(parent)

FORE tập trung vào phần improvement dương:

Delta plus = max(Delta, 0)

Lý do: một parent có thể sinh nhiều child trung bình không tốt, nhưng chỉ cần có đuôi phải mạnh, nó vẫn là stepping stone đáng mở rộng.

### 4.2 Cập nhật posterior fertility

Mỗi parent có một posterior Bayesian cho kỳ vọng improvement dương. Khi parent sinh child mới, database cập nhật:

- Số child đã đánh giá.
- Tổng Delta plus.
- Tổng bình phương Delta plus.
- Số lần Delta âm hoặc regression đáng kể.

Posterior càng có nhiều dữ liệu thì càng chắc chắn. Parent ít dữ liệu sẽ có posterior rộng hơn, nên Thompson sampling vẫn đôi khi chọn nó để exploration.

### 4.3 Structural prior cho cold start

Khi program chưa có child, chỉ nhìn lịch sử là không đủ. FORE dùng prior có cấu trúc:

- Program càng mới lạ thì prior fertility càng cao.
- Cluster càng hiếm thì prior fertility càng cao.
- Program quá cũ mà chưa sinh kết quả tốt thì bị phạt age.

Điểm quan trọng là strategy description không chỉ để đọc log; nó tham gia vào cluster rarity, tức ảnh hưởng trực tiếp đến xác suất parent được chọn.

### 4.4 Thompson sampling

Ở mỗi iteration, FORE lấy một mẫu fertility từ posterior của từng program. Sau đó tính POV:

POV = fitness hiện tại + fertility multiplier x sampled positive fertility

Trong đó fertility multiplier đại diện cho giá trị kỳ vọng khi còn nhiều bước search phía trước. Khi budget còn dài, một parent giàu fertility nhưng fitness hiện tại thấp vẫn đáng thử. Khi budget gần hết, fitness hiện tại có thể trở nên quan trọng hơn.

Program có POV sampled cao nhất được chọn làm parent.

### 4.5 Ý nghĩa hành vi

Luồng chọn parent tạo ra 4 hành vi mong muốn:

- Parent fitness cao và fertility cao: được chọn thường xuyên.
- Parent fitness cao nhưng fertility thấp: dần bị giảm ưu tiên vì vùng quanh nó đã exhausted.
- Parent fitness thấp nhưng fertility cao: vẫn được chọn như stepping stone.
- Parent mới, lạ, ít dữ liệu: đôi khi được chọn nhờ uncertainty và structural prior.

## 5. Bước chọn context program

Sau khi chọn parent chính, FORE chọn thêm context để LLM có thông tin so sánh.

Context nên gồm:

- Một sibling cùng cluster nếu có, để LLM thấy những biến thể gần parent đã thử.
- Một số exemplar từ cluster khác, ưu tiên đa dạng và POV cao.
- Các program có verdict hữu ích, ví dụ improved hoặc stepping stone.

Mục tiêu của context không phải chỉ đưa các best program, mà là giúp LLM hiểu landscape:

- Hướng nào đang hiệu quả.
- Hướng nào đã bão hòa.
- Hướng nào còn non trẻ nhưng có tín hiệu.
- Những mutation nào đã thất bại để tránh lặp.

## 6. Bước xây prompt cho LLM

FOREContextBuilder tạo prompt từ các nguồn:

1. Current parent program.
2. Lý do parent được chọn, gồm fitness, POV, fertility signal và cluster.
3. Active Reflective Review nếu đang có.
4. Sibling verdicts: các thử nghiệm gần parent và kết quả.
5. Context programs từ cluster khác.
6. Yêu cầu LLM sinh candidate mới.
7. Yêu cầu LLM đính kèm metadata chiến lược.

Prompt cần làm rõ rằng LLM có thể:

- Cải thiện trực tiếp parent nếu lineage vẫn hiệu quả.
- Chuyển hướng có chủ đích nếu review cho biết lineage đã exhausted.
- Tạo stepping stone nếu hypothesis hợp lý, kể cả chưa chắc cải thiện ngay.

Output mong muốn gồm solution/diff và một metadata block cho FORE. Metadata này được parse để lưu strategy description, hypothesis và diff_from_parent.

## 7. Bước sinh candidate

LLM nhận prompt và sinh candidate mới. Candidate có thể ở dạng full rewrite hoặc diff tùy config.

FORE không thay đổi cơ chế sinh/evaluate nền của SkyDiscover. Nó tận dụng controller hiện có cho:

- Gọi LLM.
- Retry khi generation lỗi.
- Parse solution.
- Chạy evaluator.
- Ghi log.
- Checkpoint.

Điểm FORE thêm vào là metadata chiến lược và logic chọn parent/context.

## 8. Bước evaluate candidate

Candidate được gửi vào evaluator của benchmark/task. Evaluator trả về metrics. Từ metrics, hệ thống lấy score chính để so sánh với parent và global best.

Sau evaluate, FORE xác định:

- Candidate có hợp lệ không.
- Fitness của candidate là bao nhiêu.
- Candidate có cải thiện parent không.
- Candidate có cải thiện global best không.
- Candidate nên được gắn verdict gì.

Nếu evaluation lỗi, candidate không nên làm hỏng fertility map. Tùy thiết kế chi tiết, lỗi có thể được ghi như một failed attempt riêng hoặc bỏ qua khỏi posterior Delta plus.

## 9. Cập nhật database sau mỗi candidate

Khi có candidate hợp lệ, FOREDatabase thực hiện các cập nhật chính.

### 9.1 Lưu program mới

Database lưu:

- Solution.
- Metrics.
- Iteration sinh ra.
- Parent id.
- Metadata FORE.
- Cluster id.

Nếu LLM không trả metadata, database dùng fallback description rỗng hoặc sinh label mặc định. Run không được crash chỉ vì thiếu block metadata.

### 9.2 Cập nhật fertility của parent

Database tìm parent của candidate và tính Delta.

- Nếu Delta dương, tăng tổng Delta plus.
- Nếu Delta âm, tăng negative count.
- Nếu Delta gần 0 nhưng candidate mới lạ, có thể gắn verdict stepping stone.
- Nếu Delta âm mạnh và parent đã nhiều regression, có thể gắn verdict dead end.

Thông tin này ảnh hưởng đến lần chọn parent tiếp theo.

### 9.3 Gán cluster

Candidate được gán vào strategy cluster bằng similarity giữa description/code với các cluster hiện có.

- Nếu đủ giống một cluster, thêm vào cluster đó.
- Nếu khác biệt rõ, tạo cluster mới.

Cluster được dùng cho:

- Cluster rarity trong structural prior.
- Fertility summary cho Reflective Review.
- Context selection.
- Phân tích post-run.

### 9.4 Cập nhật best program

Nếu candidate có score tốt hơn best hiện tại, database cập nhật global best. Checkpoint và output cuối run cần phản ánh best này.

### 9.5 Eviction khi archive quá lớn

Nếu số program vượt population size, FORE cần loại bớt program. Chính sách eviction nên bảo vệ:

- Global best.
- Program có POV cao.
- Program thuộc cluster hiếm nhưng còn tiềm năng.
- Program mới chưa có đủ cơ hội được thử.

Các program có fitness thấp, fertility thấp, cluster quá đông và age cao nên bị loại trước.

## 10. Reflective Review khi search bị kẹt

Reflective Review là meta-step để thoát plateau.

### 10.1 Khi nào trigger review

FORE dùng nhiều tín hiệu stagnation thay vì chỉ nhìn global best:

1. **Rate trigger**: improvement trung bình trong cửa sổ gần đây quá thấp.
2. **POV-floor trigger**: nhóm parent có POV cao nhất cũng không còn tiềm năng rõ ràng.
3. **All-cluster exhausted**: mọi cluster đều có mean Delta plus thấp.

Khi một trong các trigger đúng và đã qua cooldown, controller gọi guide LLM để tạo review.

### 10.2 Input cho review

Guide LLM nhận:

- Fertility summary theo cluster.
- Các attempt gần đây và verdict.
- Global best score.
- Các lineage effective, exhausted, embryonic theo dữ liệu hiện tại.
- Có thể kèm evaluator/task context nếu config cho phép.

Fertility summary nên đủ ngắn để guide LLM đọc được, nhưng đủ thông tin để phân biệt:

- Cluster nào có mean improvement tốt.
- Cluster nào đông nhưng hết fertility.
- Cluster nào nhỏ nhưng còn uncertainty.
- Strategy label tiêu biểu của từng cluster.

### 10.3 Output của review

Review nên có cấu trúc:

- **effective lineages**: hướng đang hiệu quả, nên khai thác tiếp.
- **exhausted lineages**: hướng đã thử nhiều nhưng ít cải thiện, nên tránh hoặc pivot.
- **embryonic lineages**: hướng mới, ít dữ liệu, có thể đáng thử.
- **next steps**: chỉ dẫn cụ thể cho vài generation tiếp theo.

Review được lưu vào database như active review.

### 10.4 Vòng đời của review

Active review chỉ được inject vào prompt trong một số generation giới hạn, ví dụ 3 lần. Sau đó:

- `uses_remaining` giảm về 0.
- Review được đưa vào history.
- Prompt không tiếp tục dùng review cũ.

Cách này tránh việc LLM bị neo vào một nhận xét đã lỗi thời.

## 11. Checkpoint, resume và log

FORE cần checkpoint đầy đủ hơn các search method đơn giản vì nó có state riêng.

Checkpoint nên lưu:

- Programs.
- Best program.
- Fertility stats từng program.
- Cluster map.
- Program-to-cluster map.
- Recent improvements.
- Active review.
- Review history.
- Iteration counter.
- Random seed/state nếu cần reproducibility.

Log nên có các event:

- Parent được chọn và POV sampled.
- Fitness, Delta và verdict của child.
- Cluster assignment.
- Eviction decision.
- Stagnation trigger reason.
- Reflective Review generated/used/expired.

Các log này rất quan trọng cho paper vì cho phép phân tích liệu FORE chọn đúng stepping stone hay chỉ tình cờ cải thiện.

## 12. Luồng kết thúc run

Khi hết iteration:

1. Controller dừng vòng search.
2. Runner lấy best program từ database.
3. Best program được re-evaluate ở test mode nếu hệ thống hỗ trợ.
4. Test metrics được ghi lại.
5. Checkpoint cuối và best solution được lưu.
6. Log/monitor nhận event run completed.

Output cuối cần cho phép trả lời:

- Best fitness đạt được là bao nhiêu.
- Parent nào sinh ra best candidate.
- Lineage nào tạo ra best candidate.
- Reflective Review có xuất hiện trước improvement lớn không.
- POV có chọn các parent khác fitness-greedy không.

## 13. Luồng dữ liệu rút gọn

Một iteration có thể tóm tắt như sau:

1. Archive hiện tại cung cấp programs, fitness, metadata và fertility stats.
2. FOREDatabase tính posterior fertility và sample POV cho từng program.
3. Program có sampled POV cao nhất trở thành parent.
4. Context builder tạo prompt từ parent, sibling verdicts, cross-cluster context và active review.
5. LLM sinh candidate và strategy metadata.
6. Evaluator chấm candidate.
7. Database cập nhật parent fertility bằng Delta.
8. Candidate được gán cluster và lưu vào archive.
9. Stagnation signals được cập nhật.
10. Nếu cần, review mới sẽ ảnh hưởng các iteration kế tiếp.

## 14. Điểm khác so với AdaEvolve, EvoX và TusoAI

So với AdaEvolve:

- AdaEvolve dùng UCB/adaptive search ở mức island.
- FORE dùng Thompson sampling ở mức program.
- AdaEvolve tập trung vào paradigm breakthrough cho mutation.
- FORE tạo review ở mức fertility map và lineage.

So với EvoX:

- EvoX co-evolve search strategy riêng.
- FORE giữ một meta-loop duy nhất nhưng làm parent selection thông minh hơn bằng posterior offspring value.

So với TusoAI:

- TusoAI dùng strategy memory và semantic clustering.
- FORE biến strategy description thành state trong archive và đưa nó vào prior/cluster/review, nhưng v1 không phụ thuộc embedding ngoài.

## 15. Những điểm cần cải thiện

### 15.1 Cải thiện về độ bền implementation

1. **Parser metadata cần fallback tốt hơn**

   LLM có thể quên hoặc viết sai metadata block. Nên có fallback nhiều tầng: parse mềm, sửa JSON nhẹ, hoặc gọi guide LLM ngắn để tóm tắt strategy từ candidate.

2. **Checkpoint cần serialize đầy đủ state FORE**

   Nếu chỉ lưu programs mà mất fertility stats, resume sẽ làm parent selection sai. Cần test resume từ checkpoint như một flow chính thức.

3. **Cần kiểm soát race khi chạy parallel iteration**

   Nếu nhiều child được evaluate song song, thứ tự cập nhật fertility và cluster có thể ảnh hưởng kết quả. V1 nên khuyến nghị `max_parallel_iterations = 1`, sau đó mới harden cho parallel.

4. **Eviction cần cẩn thận**

   Nếu eviction chỉ nhìn fitness, FORE mất stepping stone. Nếu chỉ nhìn POV sampled, kết quả có thể nhiễu. Nên dùng điểm eviction ổn định hơn, ví dụ median POV nhiều lần cộng với bảo vệ novelty/cluster rarity.

5. **Config validation**

   Các tham số như threshold cluster, review window, prior beta, cooldown nên được validate để tránh cấu hình vô lý làm run im lặng hỏng.

### 15.2 Cải thiện về mô hình fertility

1. **Clip hoặc chuẩn hóa Delta plus**

   Nhiều benchmark có scale score khác nhau. Nếu không chuẩn hóa, prior và threshold khó dùng lại giữa các task.

2. **Tách regression penalty khỏi Delta plus**

   Chỉ học trên Delta plus có thể bỏ qua việc parent sinh quá nhiều child tệ. Negative count nên ảnh hưởng rõ hơn đến POV hoặc eviction.

3. **Ước lượng uncertainty theo cluster**

   Program mới trong cluster đã exhausted không nên được ưu tiên quá cao chỉ vì ít dữ liệu. Có thể kết hợp posterior cấp program và posterior cấp cluster.

4. **Fertility multiplier nên phụ thuộc budget thật**

   Hiện plan dùng multiplier đơn giản. Có thể cải thiện bằng cách giảm trọng số fertility khi gần hết budget và tăng trọng số fitness/test reliability.

5. **Cold-start prior cần ablation**

   Novelty, rarity và age penalty đều hợp lý nhưng dễ overfit. Cần ablation để biết thành phần nào thực sự có ích.

### 15.3 Cải thiện về clustering

1. **Jaccard description cluster khá thô**

   Hai strategy giống nhau có thể dùng từ khác nhau. Hai strategy khác nhau có thể chung nhiều token. V1 chấp nhận được, nhưng nên cân nhắc embedding hoặc hybrid code-description similarity.

2. **Cluster cần merge/split**

   Cluster ban đầu có thể sai. Khi archive lớn hơn, nên có cơ chế merge cluster quá giống và split cluster quá đa dạng.

3. **Cluster summary cần chống nhiễu**

   Một child may mắn có Delta lớn có thể làm cluster trông hiệu quả. Summary nên dùng median, trimmed mean hoặc số mẫu tối thiểu.

### 15.4 Cải thiện Reflective Review

1. **Review cần có tiêu chí đánh giá sau khi dùng**

   Sau mỗi review, nên đo improvement trong vài generation kế tiếp. Nếu review không giúp, giảm trọng số những next_steps tương tự trong tương lai.

2. **Tránh review quá thường xuyên**

   Nếu review trigger nhạy, LLM sẽ liên tục pivot và mất khả năng khai thác. Cooldown và review_window cần được tune theo benchmark.

3. **Review nên chỉ ra hành động cụ thể**

   Output "explore more diverse ideas" quá chung chung không hữu ích. Prompt review nên yêu cầu next_steps gắn với cluster id, strategy label và lý do.

4. **Lưu tried_reviews để tránh lặp**

   Nếu không lưu history, guide LLM có thể đề xuất lại cùng một hướng đã thất bại.

5. **Có thể thêm confidence cho từng nhận định**

   Review nên phân biệt "cluster exhausted vì đủ dữ liệu" với "cluster chưa rõ vì ít sample".

### 15.5 Cải thiện prompt và metadata

1. **Metadata cần ngắn nhưng bắt buộc có nghĩa**

   Nếu description quá chung, cluster và review sẽ yếu. Prompt nên yêu cầu strategy_label cụ thể, hypothesis kiểm chứng được, diff_from_parent rõ ràng.

2. **Parent selection reason cần vừa đủ**

   Nếu prompt nói quá nhiều về POV, LLM có thể tối ưu theo giải thích thay vì task. Nên trình bày ngắn: vì sao parent được chọn và lineage cần thử gì tiếp.

3. **Sibling verdicts cần giới hạn số lượng**

   Quá nhiều attempt cũ làm prompt dài và nhiễu. Nên chọn các verdict đại diện: best improved, latest failed, most novel stepping stone.

### 15.6 Cải thiện thực nghiệm

1. **Cần baseline fitness-greedy rõ ràng**

   Để chứng minh RQ chính, phải so FORE với cùng pipeline nhưng parent selection chỉ dựa fitness hiện tại.

2. **Cần ablation từng thành phần**

   Tối thiểu gồm no structural prior, no review, no strategy description, no Thompson sampling.

3. **Cần đo theo API cost**

   FORE thêm guide LLM review nên so sánh theo iteration thôi là chưa đủ. Nên báo cáo best fitness theo token/cost.

4. **Cần seed nhiều lần**

   Thompson sampling và LLM đều stochastic. Một run đơn lẻ không đủ kết luận.

5. **Cần log lineage tạo ra best**

   Paper cần chứng minh FORE tìm được stepping stone, không chỉ đạt score cao. Lineage trace là bằng chứng quan trọng.

## 16. Ưu tiên triển khai khuyến nghị

Thứ tự nên làm:

1. Implement math fertility và unit test trước.
2. Implement database sampling bằng POV nhưng chưa bật review.
3. Chạy smoke test để đảm bảo parent selection, add program và checkpoint ổn.
4. Thêm metadata strategy description vào prompt và checkpoint.
5. Thêm clustering và context selection theo cluster.
6. Thêm Reflective Review sau cùng.
7. Chạy ablation nhỏ để tune threshold trước khi benchmark lớn.

Lý do: nếu review được thêm quá sớm, rất khó biết improvement đến từ POV parent selection hay từ LLM review.

## 17. Kết luận

FORE là một search method xoay quanh khả năng sinh hậu duệ của program, thay vì chỉ nhìn fitness hiện tại. Luồng vận hành chính là: lưu strategy description, học posterior fertility từ lịch sử parent-child, chọn parent bằng Thompson-sampled POV, dùng fertility map để phát hiện stagnation, rồi cho LLM review các lineage khi cần pivot.

Điểm cần làm chắc nhất trong v1 là tính đúng fertility stats, checkpoint đầy đủ state, parse metadata bền, và có log đủ chi tiết để chứng minh hành vi stepping-stone. Các cải thiện nâng cao như embedding cluster, cluster-level posterior và review scoring có thể đưa vào v2 sau khi pipeline cơ bản chạy ổn.
