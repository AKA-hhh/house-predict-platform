# data/ 目录说明

本目录下的**数据文件不入库**（见根目录 `.gitignore`）——它们是运行产物，
体积大（合计约 630 MB）且可完全重新生成。这里只保留目录骨架说明。

## 目录结构与产出方式

| 路径 | 内容 | 是否入库 | 生成方式 |
| --- | --- | --- | --- |
| `raw/` | 原始数据集（`new.csv`，约 58 MB） | ✗ | `python tools/etl.py download --source lianjia_bj_raw` |
| `processed/` | 清洗+特征后的 Parquet 列存快照（约 15 MB / 版本） | ✗ | `python tools/etl.py run --source lianjia_bj_raw` |
| `models/<version_id>/` | 时间切分后的 `train / valid / test.parquet` | ✗ | 同上（流水线第 ⑤ 阶段） |
| `models/*.joblib` | 已训练的价格模型文件 | ✗ | `python run.py` 后在「模型训练」页训练，或调 `POST /api/models/train` |
| `models/registry.json` | 模型仓库索引（算法、参数、指标） | ✗ | 同上，训练时自动登记 |
| `registry/` | **数据版本登记簿**（每版本一个 JSON + `index.json`） | ✓ | 每次流水线运行自动写入 |
| `reports/` | 预聚合仪表盘指标 JSON 快照 | ✗ | 同上（流水线第 ⑥ 阶段） |
| `house.db` | SQLite 明细库（`house_detail` + `agg_*` 表） | ✗ | 同上（流水线第 ⑤ 阶段） |

## 从零重建全部数据

```bash
# 1) 下载原始数据（约 58 MB）
python tools/etl.py download --source lianjia_bj_raw

# 2) 跑一次完整流水线：导入 → 清洗 → 特征 → 划分 → 存储 → 预聚合
python tools/etl.py run --source lianjia_bj_raw --missing median --outlier iqr --split time

# 3) 启动服务，在「模型训练与对比」页训练模型（或直接调 API）
python run.py
```

或者一步到位：`python run.py --init`

流水线跑完后会生成一个新的数据版本（例如 `registry/ds-YYYYMMDD-HHMMSS-xxxx.json`），
其中记录了本次运行的**全部参数、各阶段行数变化、原始文件 SHA-256、产出物校验和与列级统计指纹**。
已入库的 `registry/` 下保留了本项目实测的 9 个版本记录，可以直接看到版本记录的完整结构。

## 为什么这些文件不入库

- GitHub 单文件上限 100 MB —— `house.db` 约 130 MB，直接超限；
- `raw/` + `processed/` + `models/` 合计约 630 MB，每次跑流水线都会新增版本，
  入库会让仓库迅速膨胀；
- 原始数据是**公开可下载**的（CC0 许可），没有必要复制一份到本仓库。
