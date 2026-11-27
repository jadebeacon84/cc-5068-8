# 小范围场景声音事件识别原型（纯本地链路）

识别四类**非敏感**目标声音，并输出事件类别与时间区间：

| 类别 | 含义 | 训练信号特征 |
| --- | --- | --- |
| `knock` | 敲击（3–6 个脉冲的敲击串） | 短脉冲、带通瞬态 |
| `bell` | 机械式车铃（含同一只铃的重复尾响） | 非谐和泛音、较长指数衰减 |
| `chime` | 电子提示音（旧 `bell` 标签拆出的新类） | 高频整数谐波、硬包络短促蜂鸣 |
| `machine` | 机械持续声 | 稳态谐波 + 幅度调制 + 气流噪声 |

另有 `unknown`（扫频、啁啾、风声、类浊音块等词表外声音）与 `background`
（房间底噪/风扇纹理）。**未知声音允许拒绝，背景段不会被默认等同于任何
目标类别。**

## 标签体系与旧标签迁移（label version 2）

旧体系只有三类（`knock` / `bell` / `machine`），其中共享的 `bell` 标签被
**一对多**拆成当前的机械车铃 `bell` 与电子提示音 `chime`（见
`ser/labeling.py` 的 `LEGACY_TO_CURRENT`，反向映射 `CURRENT_TO_LEGACY`）。
拆类后不允许静默归属：旧标签没有明确指向哪一个新类时，解析结果为
`unresolved`，必须经试听审定后才能进入训练。

每个标注区间带有两种维度的状态：

- 旧标注粒度 `label_scope`：`single_event`（标记单次事件）或
  `whole_segment`（标记一整段，拆类后该段可能含多个新类事件）。
- 试听审定状态 `audit_status`：
  - `audited`：试听通过，**唯一**允许进入单类训练集的状态；
  - `pending_review`：由旧标签转换而来、尚未试听审定，训练打包时 withheld；
  - `unclaimed`：干声源轨被移走后区间身份无法主张，自动降级且禁止入训练集。

数据集在 train/val 各包含一个**时间重叠的 bell/chime 示例**（二者曾共享旧
`bell` 标签），其 `legacy_annotations` 同时保留单事件与整段两种旧粒度；
val 副本中的 chime 保持 `pending_review`，直到在界面中审定。

### 试听审定约束样本打包

`ser/dataset.py` 的 `label_windows(..., audited_only=True)` 只打包满足
`is_trainable_event` 的区间：类别属于当前四类、`audit_status=audited`、
未标记 `identity_uncertain`、且声源轨在磁盘上。评估时使用
`audited_only=False` 的完整真值。因此审定结论直接决定一条区间是否贡献
正样本窗口，该规则由 `ser/label_check.py` 交叉校验。

### 声源轨隔离与恢复（待认领）

`ser/labeling.py` 提供可逆的分轨隔离：

- `quarantine_source_tracks`：把指定区间的干声源 WAV 移到
  `sources_quarantined/`，对应区间降级为 `unclaimed`、置
  `identity_uncertain=True` 并记录原状态；该区间立即退出训练打包。
- `restore_source_tracks`：移回声轨，按区间 `event_id` 恢复原类别、边界与
  审定状态，重新进入打包。
- `ser/release.py` 的 `verify_quarantine_cycle` 按**区间身份**（event id、
  类别、起止边界、审定状态、是否重新打包）核对整个隔离-恢复循环。

整个处理链——合成标注录音、STFT/log-mel 特征、训练、窗口打分、事件合并/
断开、提示单编辑、标签审定、失败检查——全部在本机 Python 进程内完成，
**没有任何远端识别服务调用**（`tests/test_local_chain.py` 会静态检查禁用
网络库导入）。

## 1. 安装

需要 Python 3.11（仅用 CPU）。

```bash
python3 -m pip install --user -r requirements.txt
```

## 2. 图形界面（声音事件的主入口）

声音事件的试听、选择、审定与提示单编辑以 **GUI 为主入口**，命令行脚本仅
用于数据准备、训练、检查等辅助环节。

```bash
python3 -m scripts.run_ui            # 默认 http://127.0.0.1:5000
# 或：python3 -m ser.webapp
```

浏览器打开 `http://127.0.0.1:5000`。界面只消费声音事件内核返回的 JSON，
所有控件都绑定本地处理链：

- **声音分析窗口**：左侧录音列表（split/录制条件/事件数）；中间是可实际
  播放的场景录音波形，叠加 1 秒分析窗网格，时间轴绘制事件块并标注事件
  标签与发生位置。
- **选择事件**：点击时间轴事件块或提示单选择框回放预测片段；点击概率
  曲线选中对应分析窗。
- **标注资料与试听审定面板**：表格逐条呈现真实标注的事件 id、标签、
  发生位置（开始–结束、时长）、审定状态与身份确定性，并给出旧标签映射；
  可直接播放混音/分轨，然后“通过审定”或“保留待审”。审定结果经
  `POST /api/recordings/<id>/audit` 写回 `labels.json`，并即时改变该录音的
  可训练区间集合（`GET /api/recordings/<id>/annotations` 可复核）。
- **结果区**：提示单表格呈现预测事件标签与位置（置信度、可能次数、分轨
  身份、状态），支持改类、拆分、删除、报失败。
- **证据区**：展示该分析窗**模型真正吃进去的 40×50 log-mel 输入**、四类
  概率 vs 校准阈值、保留/拒绝裁决、同窗标注与关联事件，并可回放这 1 秒。
- **失败反馈回内核**：“报失败”写入 `data/feedback.jsonl` 并可用分轨核对，
  不是弹窗了事。

## 3. 命令行辅助脚本与一键复现

在仓库根目录依次执行，下列命令均可直接运行：

```bash
python3 -m scripts.prepare_data     # 本地合成带独立声源轨的标注录音（重建前清空旧 split 目录）
python3 -m scripts.train_model      # 录音级划分训练多标签随机森林
python3 -m scripts.check_splits     # 检查划分、窗口、特征形状、录音级泄漏
python3 -m scripts.check_labels     # 标签时长/划分/摘要/旧标签覆盖/打包规则/陈旧目录
python3 -m scripts.evaluate         # 事件级评估（留出集 + 干扰集）
python3 -m scripts.build_release    # 重建发布包并按区间身份验证隔离/恢复
python3 -m scripts.cut_restore_demo # 裁剪-合并 / 补回-拆回 + 分轨核对
python3 -m pytest tests/ -q         # 43 项本地测试
```

`check_labels` 只读磁盘产物、不修改数据，可指向任意净目录
（`--data-root DIR`，`--json` 输出机读报告），任一检查失败时退出码为 1。
它覆盖：

- 区间时长与边界（短于 0.05 s 硬失败；短于配置的 0.15 s 仅告警，不伪报）；
- 训练/评估录音级互斥，同一录音不会同时进入训练与评估；
- 每个 WAV 的 SHA-256 摘要（缺失/被篡改/未登记文件）与跨录音重复素材；
- 旧标签一对多覆盖、未裁决的拆类区间；
- 审定打包规则（待审/待认领区间不得贡献正样本窗口）；
- **manifest 之外的陈旧录音目录**：重新生成数据集时 `prepare_data` 会清空
  旧 split 目录，净目录下该项恒为 0，因此不会产生伪报。

`build_release` 先要求源数据集通过全部标签检查，再把数据复制到全新发布
目录（排除 `cue_sheet.json`、`feedback.jsonl`、`sources_quarantined` 等
运行时文件），在副本上对每个重叠的已审定事件对执行隔离-恢复循环并按
区间身份核对，最后再跑一次完整检查并写出校验和清单与
`release_report.json`。工作数据集全程不被修改。

生成物：

- `data/<split>/<recording_id>/mix.wav`：模型真正分析的混音
- `data/<split>/<recording_id>/sources/*.wav`：每个事件一条**干声源轨**，
  文件名只含事件 id，不含类别名（无法靠文件名反推标签）
- `data/<split>/<recording_id>/labels.json`：事件区间、审定状态、声源映射、
  旧标签标注、文件摘要与合成参数
- `experiments/model/`：模型、mel 滤波器组、标准化器、校准阈值、配置快照
- `experiments/evaluation/*_report.json`：事件级评估报告与可检查配置
- `experiments/release/`：通过隔离/恢复身份核对的发布包与校验和清单

## 4. 关键设计如何对应需求

### 重叠铃声：尾响 vs 另一只铃 / 车铃 vs 提示音

- 合成阶段把机械车铃（低频、非谐和泛音、长衰减）与电子提示音（高频、
  整数谐波、3 ms/10 ms 硬包络蜂鸣）做成可声学区分的两类，并强制生成时间
  重叠的 bell/chime 对以继承共享旧标签。
- 混音可分时：`events.match_sources` 用匈牙利算法按 IoU 把预测事件挂到干
  声源轨（`source_match=exact`，带源 id）。
- 只剩混音时：`annotate_bell_counts` 从铃声概率曲线数峰值，给出
  `possible_count` 与“仅事件数（混音歧义）”，不硬安身份。

### 裁剪-补回驱动提示单合并/拆回

`ser/cue.py` 的 `cut_section` / `restore_section` 是可逆操作：剪掉两声
之间的一小段后时间轴收缩、同类相邻标记自动合并；补回时从快照恢复原
标记。`cut_restore_demo` 与 Web 的“剪掉中间段 / 补回最近裁剪”按钮走同一
套代码，补回后用留出分轨核对原事件身份与起止位置（IoU、边界误差）。

### 跨窗合并，但真实停声必须断开

`events.assemble_events` 先把重叠窗口的概率平均到帧级，再按每类阈值取
连通段：小于 `merge_gap_seconds=0.30s` 的间隙合并（同一事件横跨两个分析
窗不会被切成两段）；同时用能量门 `_silence_breaks` 检测混音中的真实停声，
停声超过合并间隙就强制断开。短事件（敲击、提示音）短于 1 秒分析窗，
事件级门限采用“至少一个窗口的最大概率过阈”而非均值，避免背景帧稀释。

### 防泄漏划分

- 划分单位是**整条录音**：`assert_no_leakage` 保证同一录音 id 不会跨
  train/val/test/interference；窗口只是录音内部的视图。
- 训练/验证/测试使用不同录音与独立录制条件（`clean_room` /
  `room_reverb` / `interference`）。
- 评估含专门的 `interference` 留出集（混入稳态风扇纹理与密集未知声）。
- 各 split 的合成种子由固定基准种子加确定性偏移得到，整库可逐字节复现。

### 不靠文件夹名、不靠远端

- 标签只来自 `labels.json`；测试 `test_predictions_independent_of_folder_name`
  把录音复制到具有误导性类名的目录后，断言模型输出概率完全不变。
- 处理包子树静态禁用 `requests` / `aiohttp` / `urllib.request` 等导入。

## 5. 模型与可检查参数

- 特征：16 kHz、`n_fft=1024`、帧移 320（20 ms/帧）、40 个 mel 频带，
  分析窗 50 帧 = 1.0 秒，窗跳 10 帧 = 200 ms；窗口与事件重叠 ≥0.15 s 才
  打正标签（保证短敲声/提示音也能赢得训练窗口）。
- 模型：每类一个 `RandomForestClassifier`（一对其余），未知/背景窗为全零
  多标签负例；阈值在验证窗上按精确率下限 0.55 内取 F1 最优（同点取召回
  更高者），宁可拒绝也不把背景吸进目标类。
- 所有参数随模型保存到 `experiments/model/config_snapshot.json`，
  评估目录另有 `inspectable_config.json` 记录 split 明细与训练元数据。

## 6. 目录结构

```
ser/
  config.py      可检查的音频/事件/数据集参数、四类目标与审定常量
  audio.py       WAV 读写、重采样、能量门
  synthesis.py   本地标注录音合成（独立声源轨、混响、干扰、重叠 bell/chime）
  features.py    mel 滤波器组、log-mel、窗口切分、标准化
  dataset.py     录音级加载、审定约束的窗口多标签、泄漏断言
  labeling.py    旧标签一对多映射、审定状态、摘要、隔离/恢复
  label_check.py 标签资料检查（时长/划分/摘要/覆盖/打包/陈旧目录）
  release.py     发布包构建与按区间身份的隔离-恢复核验
  model.py       每类随机森林、阈值校准、工件序列化
  events.py      推理内核：帧证据、事件合并/断开、铃计数、分轨匹配
  cue.py         可编辑提示单、裁剪/补回、合并/拆分、分轨验收
  evaluation.py  事件级 IoU 裁决、误报/拒绝率、分条件汇总
  feedback.py    失败反馈 JSONL
  pipeline.py    训练/推理/审定/证据/波形编排
  webapp.py      本地 Flask 路由（内核驱动控件）
  static/, templates/  原生 JS/canvas 界面
scripts/         可复现实验、标签检查、发布与界面启动入口
tests/           43 项测试（含标签迁移、隔离身份、无伪报、泄漏、端到端 Web、无网络护栏）
```
