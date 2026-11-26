# 小范围场景声音事件识别原型（纯本地链路）

识别三类**非敏感**目标声音，并输出事件类别与时间区间：

| 类别 | 含义 | 训练信号特征 |
| --- | --- | --- |
| `knock` | 敲击（2–5 个脉冲的敲击串） | 短脉冲、带通瞬态 |
| `bell` | 铃声（含同一只铃的重复尾响） | 非谐和泛音、指数衰减 |
| `machine` | 机械持续声 | 稳态谐波 + 幅度调制 + 气流噪声 |

另有 `unknown`（扫频、鸟鸣状啁啾、风声噪声、类浊音块等词表外声音）与
`background`（房间底噪/风扇纹理）。**未知声音允许拒绝，背景段不会被默认
等同于任何目标类别。**

整个处理链——合成标注录音、STFT/log-mel 特征、训练、窗口打分、事件合并/
断开、提示单编辑、失败检查——全部在本机 Python 进程内完成，**没有任何
远端识别服务调用**（`tests/test_local_chain.py` 会静态检查禁用网络库导入）。

## 1. 安装

需要 Python 3.11（仅用 CPU）。

```bash
python3 -m pip install --user -r requirements.txt
```

## 2. 一键复现实验

在仓库根目录依次执行：

```bash
python3 -m scripts.prepare_data     # 本地合成带独立声源轨的标注录音
python3 -m scripts.train_model      # 录音级划分训练多标签随机森林
python3 -m scripts.check_splits     # 可检查：划分、窗口、特征形状、泄漏
python3 -m scripts.evaluate         # 事件级评估（留出集 + 干扰集）
python3 -m scripts.cut_restore_demo # 裁剪-合并 / 补回-拆回 + 分轨核对
python3 -m pytest tests/ -q         # 24 项本地测试
```

生成物：

- `data/<split>/<recording_id>/mix.wav`：模型真正分析的混音
- `data/<split>/<recording_id>/sources/*.wav`：每个事件一条**干声源轨**，
  文件名只含事件 id，不含类别名（无法靠文件名反推标签）
- `data/<split>/<recording_id>/labels.json`：事件区间、声源映射、合成参数
- `experiments/model/`：模型、mel 滤波器组、标准化器、校准阈值、配置快照
- `experiments/evaluation/*_report.json`：事件级评估报告与可检查配置

## 3. 启动界面

```bash
python3 -m scripts.run_ui            # 默认 http://127.0.0.1:5000
# 或：python3 -m ser.webapp
```

浏览器打开 `http://127.0.0.1:5000`。界面只消费声音事件内核返回的 JSON：

- **声音分析窗口**：左侧录音列表（含 split/录制条件/事件数）；中间是可播放
  的场景录音波形，叠加 1 秒分析窗网格，下方时间轴绘制事件块。
- **选择事件**：点击时间轴事件块或提示单单选框，可回放预测片段；点击概率
  曲线任意位置选中对应分析窗。
- **结果区**：提示单表格呈现事件标签与发生位置（开始/结束、置信度、可能
  次数、分轨身份、状态），支持改类、拆分、删除、报失败。
- **证据区**：展示该分析窗**模型真正吃进去的 40×50 log-mel 输入**（画布渲染）、
  三类概率 vs 校准阈值、保留/拒绝裁决、同窗标注与关联事件，并可回放这 1 秒。
- **失败反馈回内核**：“报失败”会把条目标记为 `failure`，写入
  `data/feedback.jsonl`，并可再次用分轨核对；不是弹窗了事。

## 4. 关键设计如何对应需求

### 重叠铃声：尾响 vs 另一只铃

- 合成时约 40% 的铃是同一只铃的两次鸣响（`ring_count=2`，一条声源轨）；
  房间/干扰条件下再以 60% 概率叠加**第二只独立铃**（各自独立声源轨，
  `overlaps` 互记）。
- 混音可分时：`events.match_sources` 用匈牙利算法按 IoU 把预测事件挂到干
  声源轨（`source_match=exact`，带源 id）。
- 只剩混音时：`annotate_bell_counts` 从铃声概率曲线数峰值，给出
  `possible_count` 与“仅事件数（混音歧义）”，不硬安身份。

### 裁剪-补回驱动提示单合并/拆回

`ser/cue.py` 的 `cut_section` / `restore_section` 是可逆操作：剪掉两声
之间的一小段后，时间轴收缩、同类相邻标记自动合并（状态
`merged_by_cut`）；补回时从快照恢复原标记。`cut_restore_demo` 与 Web 的
“剪掉中间段 / 补回最近裁剪”按钮走同一套代码，补回后用留出分轨核对原
事件身份与起止位置（IoU、起止边界误差）。

### 跨窗合并，但真实停声必须断开

`events.assemble_events` 先把重叠窗口的概率平均到帧级，再按每类阈值取
连通段：小于 `merge_gap_seconds=0.30s` 的间隙合并（同一事件横跨两个分析
窗不会被切成两段）；同时用能量门 `_silence_breaks` 检测混音中的真实停声，
停声超过合并间隙就强制断开。裁决是**事件级**（二分图 IoU 匹配），不做
帧级误差统计。

### 防泄漏划分

- 划分单位是**整条录音**：`assert_no_leakage` 保证同一录音 id 不会跨
  train/val/test/interference；窗口只是录音内部的视图。
- 训练/验证/测试使用不同录音与独立录制条件（`clean_room` /
  `room_reverb` / `interference`，混响 RT60、底噪、干扰密度不同）。
- 评估含专门的 `interference` 留出集（混入稳态风扇纹理与密集未知声）。
- `check_splits` 打印每个 split 的录音 id、窗口数、标签分布与特征形状，
  全部可检查。

### 不靠文件夹名、不靠远端

- 标签只来自 `labels.json`；测试 `test_predictions_independent_of_folder_name`
  把录音复制到具有误导性类名的目录后，断言模型输出概率完全不变。
- 处理包子树静态禁用 `requests` / `aiohttp` / `urllib.request` 等导入。

## 5. 模型与可检查参数

- 特征：16 kHz、`n_fft=1024`、帧移 320（20 ms/帧）、40 个 mel 频带，
  分析窗 50 帧 = 1.0 秒，窗跳 10 帧 = 200 ms；窗口与事件重叠 ≥0.25 s 才打正标签。
- 模型：每类一个 `RandomForestClassifier`（一对其余），未知/背景窗为全零
  多标签负例；阈值在验证窗上按“精确率下限 0.85 内取 F1 最优”校准，
  宁可拒绝也不把背景吸进目标类。
- 所有参数随模型保存到 `experiments/model/config_snapshot.json`，
  评估目录另有 `inspectable_config.json` 记录 split 明细与训练元数据。

## 6. 目录结构

```
ser/
  config.py      可检查的音频/事件/数据集参数与快照
  audio.py       WAV 读写、重采样、能量门
  synthesis.py   本地标注录音合成（独立声源轨、混响、干扰、纯背景录音）
  features.py    mel 滤波器组、log-mel、窗口切分、标准化
  dataset.py     录音级加载、窗口多标签、泄漏断言
  model.py       每类随机森林、阈值校准、工件序列化
  events.py      推理内核：帧证据、事件合并/断开、铃计数、分轨匹配
  cue.py         可编辑提示单、裁剪/补回、合并/拆分、分轨验收
  evaluation.py  事件级 IoU 裁决、误报/拒绝率、分条件汇总
  feedback.py    失败反馈 JSONL
  pipeline.py    训练/推理/证据/波形编排
  webapp.py      本地 Flask 路由（内核驱动控件）
  static/, templates/  原生 JS/canvas 界面
scripts/         可复现实验与界面启动入口
tests/           24 项测试（含泄漏、拒绝、端到端 Web、无网络调用护栏）
```
