# 🔦 光学仿真系统 — Optical Simulation System

基于 Python 的光学照度仿真工具，支持 **IES 光度数据文件**解析、**多光源排列**、**球面/平面接收屏**照度分析、**光线追迹**可视化及 **配光曲线**分析。

---

## 📋 目录

- [原理概述](#-原理概述)
- [项目结构](#-项目结构)
- [安装与运行](#-安装与运行)
- [使用介绍（命令行）](#-使用介绍命令行)
- [使用介绍（UI 界面）](#-使用介绍-ui-界面)
- [参数说明](#-参数说明)
- [输出文件说明](#-输出文件说明)
- [错误问答](#-错误问答)
- [注意事项](#-注意事项)

---

## 🔬 原理概述

### 光度学基础

仿真基于光度学三大定律：

| 物理量 | 符号 | 单位 | 说明 |
|--------|------|------|------|
| 光强 | \(I\) | cd (坎德拉) | 光源在单位立体角内发出的光通量 |
| 照度 | \(E\) | lux (勒克斯) | 单位面积上接收到的光通量，\(E = I / r^2 \cdot \cos\alpha\) |
| 光通量 | \(\Phi\) | lm (流明) | 光源发出的总可见光能量 |

### IES 文件格式

IESNA:LM-63-2002 是照明工程学会标准光度数据文件格式，描述灯具在空间各方向的光强分布：

- **Type C 坐标系**：垂直角 \(\theta\)（0°=正下方 → 180°=正上方）和水平角 \(\phi\)（绕垂直轴旋转）
- **坎德拉矩阵**：每个 \((\theta, \phi)\) 方向对应的光强值
- 本项目中使用的 **LTE-C1726-ZH-GL.ies** 为旋转对称配光（所有水平角数据相同）

### 照度计算流程

```
IES 文件 → 解析坎德拉矩阵 → 生成插值查找表
                                      ↓
光源位置/朝向 → 计算到接收屏各点的方向向量
                                      ↓
查 IES 表得光强 I(θ) → 平方反比衰减 → 入射角余弦修正
                                      ↓
                        累加各光源贡献 → 总照度分布
```

- **平方反比定律**：\(E = I / r^2\)（点光源近似）
- **入射角修正**：\(\cos\alpha\)，\(\alpha\) 为光线与表面法线的夹角
- **光源朝向**：通过 pitch/yaw 角定义光源本地 +Z 轴在全局坐标系中的方向

### 接收屏模型

- **球面接收屏**：以原点为中心的球面，法线方向为径向向外，适合评估光源的全空间分布
- **平面接收屏**：固定 Z 高度的 XY 平面，法线为 +Z 方向，模拟实际照明场景

---

## 📁 项目结构

```
optical_simulation/
├── LTE-C1726-ZH-GL.ies      # IES 光度数据文件（示例光源）
├── optical_simulation.py     # 命令行仿真脚本（球面+平面）
├── ui_app.py                 # Gradio 图形界面应用
├── .venv/                    # Python 虚拟环境（首次安装后生成）
└── output/                   # 运行生成的输出文件（自动创建）
    ├── sphere_irradiance_3d.png
    ├── sphere_irradiance_2d.png
    ├── ray_tracing_3d.png
    ├── candela_distribution.png
    ├── plane_irradiance_2d.png
    ├── plane_irradiance_3d.png
    ├── simulation_data_*.csv
    ├── plane_data_*.csv
    └── *_summary.json
```

---

## 🛠 安装与运行

### 环境要求

- Python ≥ 3.9
- pip（Python 包管理器）

### 快速安装

```bash
# 1. 进入项目目录
cd optical_simulation

# 2. 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install numpy matplotlib gradio pandas
```

### 运行方式

#### 方式一：命令行模式

```bash
lsof -ti:7860 | xargs kill -9 
source .venv/bin/activate
MPLCONFIGDIR=/tmp/mplcache python3 optical_simulation.py
```

输出结果保存在当前目录。

#### 方式二：图形界面模式（推荐）
lsof -ti:7860 | xargs kill -9 关闭端口（如果端口开启）
```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/mplcache python3 ui_app.py
```

启动后在浏览器中打开 **http://localhost:7860**

> **💡 提示**：`MPLCONFIGDIR=/tmp/mplcache` 可避免 macOS 上 matplotlib 的权限错误。如果不需要可以省略。

---

## 🖥️ 使用介绍（命令行）

### `optical_simulation.py`

直接运行将使用默认配置进行仿真：

```bash
python3 optical_simulation.py
```

默认参数：
- IES 文件：`LTE-C1726-ZH-GL.ies`
- 6 个光源在 XY 平面 R=10mm 圆周均匀排布，全部朝向 +Z
- 球面接收屏 R=500mm（采样 181×360）
- 平面接收屏 Z=500mm（范围 ±1500mm，采样 301×301）
- 500 条光线/光源 用于追迹

如需修改参数，请直接编辑脚本开头的配置区域：

```python
IES_FILE = "LTE-C1726-ZH-GL.ies"
R_SPHERE = 500.0
N_SOURCES = 6
N_THETA = 181
N_PHI = 360
N_RAYS_PER_SOURCE = 500
Z_PLANE = 500.0
PLANE_RANGE = 1500.0
PLANE_N = 301
```

---

## 🎨 使用介绍（UI 界面）

### 启动

```bash
source .venv/bin/activate
python3 ui_app.py
```

浏览器打开 **http://localhost:7860**

### 界面布局

```
┌─────────────────────────────────────────────────────────────┐
│  🔦 光学仿真系统                                             │
├───────────────────┬─────────────────────────────────────────┤
│  📂 配置面板      │  📊 结果                                 │
│                   │                                         │
│  📄 IES 文件     │  [○] 3D 照度分布                          │
│  [选择文件...]    │  [○] 2D 照度分布    ← 选择可视化类型      │
│                   │  [●] 光线追迹                             │
│  💡 光源配置     │  [○] 配光曲线                             │
│  数量: [6]        │                                         │
│  ┌────┬───┬──┬───┬──┐  ┌─────────────────────────────┐     │
│  │ x  │ y │z │pch│yaw│  │                             │     │
│  ├────┼───┼──┼───┼──┤  │   仿真结果图片显示区域        │     │
│  │10.0│0.0│0 │0  │0  │  │                             │     │
│  │5.0 │8.7│0 │0  │0  │  │                             │     │
│  └────┴───┴──┴───┴──┘  └─────────────────────────────┘     │
│  光线数: [200]     │                                         │
│                   │  ┌─────────────────────────────────┐     │
│  📐 接收屏       │  │ 结果摘要 (JSON)                  │     │
│  [● 球面] [○ 平面]│  │ {...}                           │     │
│  半径: [500] mm   │  └─────────────────────────────────┘     │
│  极角: [181]      │                                         │
│  方位角: [360]    │                                         │
│                   │                                         │
│  [🚀 运行仿真]    │                                         │
└───────────────────┴─────────────────────────────────────────┘
```

### 操作步骤

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 选择/拖放 IES 文件 | 默认加载项目中已有的 `.ies` 文件 |
| 2 | 调节光源数量和位置 | 滑块改变数量，表格中编辑 \(x,y,z,\text{pitch},\text{yaw}\) |
| 3 | 选择接收屏类型 | 球面（Sphere）或平面（Plane） |
| 4 | 设置对应参数 | 球面半径/采样数 或 平面Z高度/范围/点数 |
| 5 | 点击 **🚀 运行仿真** | 等待计算完成 |
| 6 | 切换可视化类型 | 选择 3D/2D/光线追迹/配光曲线 查看 |
| 7 | 阅读结果摘要 | 右侧文本框显示关键数据 |

### 光源朝向说明

每个光源有 5 个参数：

| 参数 | 说明 | 范围 |
|------|------|------|
| `x, y, z` | 光源在空间中的位置 (mm) | 任意 |
| `pitch_deg` | 俯仰角：光源从 +Z 方向倾斜的角度 | 0°（朝+Z）~ 90°（朝XY平面） |
| `yaw_deg` | 偏航角：绕 Z 轴旋转的角度 | 0° ~ 360° |

---

## 📐 参数说明

### IES 文件参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| IES 文件 | `LTE-C1726-ZH-GL.ies` | 标准 IESNA:LM-63-2002 格式 |
| 最大光强 | 100 cd | 单灯峰值光强 |
| 配光对称性 | 旋转对称 | 所有水平角数据相同 |

### 光源参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 光源数量 | 6 | 1 ~ 24 |
| 位置 | XY 圆周 R=10mm | 每个光源可单独编辑 |
| 朝向 | pitch=0°, yaw=0° | 默认全部朝 +Z |
| 光线数 | 200~500 | 用于光线追迹图的采样密度 |

### 球面接收屏参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 半径 | 500 mm | 球面半径 |
| 极角采样 | 181 | 0°~180°，决定垂直分辨率 |
| 方位角采样 | 360 | 0°~360°，决定水平分辨率 |

### 平面接收屏参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Z 高度 | 500 mm | 平面距光源平面的距离 |
| XY 范围 | ±1500 mm | 采样区域大小 |
| 每轴点数 | 301 | 网格分辨率（301×301） |

---

## 📂 输出文件说明

运行一次仿真生成以下文件：

### 可视化图像（PNG）

| 文件名 | 说明 |
|--------|------|
| `sphere_irradiance_3d.png` | 球面照度 3D 彩色分布图 |
| `sphere_irradiance_2d.png` | 球面照度 2D 等距圆柱投影图 |
| `plane_irradiance_2d.png` | 平面照度伪彩色图 + 等照度线 + 截面曲线 |
| `plane_irradiance_3d.png` | 平面照度 3D 曲面图 |
| `ray_tracing_3d.png` | 光线追迹 3D 可视化 |
| `candela_distribution.png` | 总配光曲线（极坐标+笛卡尔坐标） |

### 数据文件（CSV / JSON）

| 文件名 | 说明 |
|--------|------|
| `simulation_data_irradiance.csv` | 球面各点照度值（抽样） |
| `simulation_data_candela.csv` | 各极角对应的总光强 (cd) |
| `simulation_data_summary.json` | 球面仿真结果摘要 |
| `plane_data_irradiance.csv` | 平面照度网格数据 |
| `plane_data_cross_section.csv` | 平面 X 轴截面照度曲线 |
| `plane_data_summary.json` | 平面仿真结果摘要 |

### JSON 摘要字段

```json
{
  "config": {
    "ies_file": "LTE-C1726-ZH-GL.ies",
    "n_sources": 6,
    "source_circle_radius_mm": 10.0,
    "sphere_radius_mm": 500.0,
    "source_orientation": "+Z"
  },
  "results": {
    "peak_irradiance_lux": 2397.14,
    "mean_irradiance_lux": 977.37,
    "total_flux_on_sphere_lm": 2743.0,
    "peak_total_intensity_cd": 599.25,
    "half_max_angle_deg_FWHM": 80.0,
    "single_lamp_peak_intensity_cd": 100.0
  }
}
```

---

## ❓ 错误问答

### 1. 运行时出现 `too many values to unpack (expected 2, got 5)`

**原因**：旧版本 `ui_app.py` 中回调函数返回值数量不匹配。

**解决**：更新到最新版本。此问题已在 `run_and_update` 函数中修复。

---

### 2. 点击运行后结果摘要显示 `Error: 'y'`

**原因**：Gradio Dataframe 组件将光源数据以 `{'x': [0], 'y': [0], ...}` 的列形式 dict 传入，但旧版代码未处理此格式。

**解决**：更新到最新版本。目前已支持 3 种输入格式：
- 标准 Gradio dict：`{'data': [[...]], 'headers': [...]}`
- 列形式 dict：`{'x': [...], 'y': [...], ...}`
- 纯列表：`[[...], ...]`

---

### 3. matplotlib 报错 `mkdir -p failed for path /Users/user/.matplotlib`

**原因**：macOS 上 matplotlib 缓存目录权限问题。

**解决**：
```bash
# 方式一：设置临时缓存目录（推荐）
MPLCONFIGDIR=/tmp/mplcache python3 ui_app.py

# 方式二：创建用户缓存目录
mkdir -p ~/.matplotlib
```

---

### 4. 启动 UI 时报端口占用 `Cannot find empty port in range: 7860-7860`

**原因**：默认端口 7860 已被占用（可能上次关闭未完全退出）。

**解决**：
```bash
# 方式一：杀掉占用进程
lsof -ti:7860 | xargs kill -9

# 方式二：使用其他端口
GRADIO_SERVER_PORT=7861 python3 ui_app.py
```

---

### 5. 仿真结果照度值为 0 或极小

**可能原因**：
- 光源朝向错误（如 pitch=90° 导致光轴与接收屏平行）
- 光源位置在接收屏后方
- IES 文件未正确加载
- 采样分辨率过低
- 平面接收屏的 XY 范围过大，导致平均照度稀释

**排查**：检查 JSON 摘要中的 `peak_irradiance_lux`，如果为 0 则检查 IES 文件和光源配置。

---

### 6. 平面接收屏的光斑直径显示为 `null`

**原因**：光斑边缘未达到半高值，或 FWHM 计算失败（通常因为照度分布过于平坦或范围不够大）。

**解决**：减小 XY 采样范围，或将光源移近接收屏。

---

### 7. 运行命令行脚本时卡住无输出

**原因**：球面 181×360 网格 + 6 光源的照度计算可能需要 1~3 分钟，属正常现象。

**解决**：可降低 `N_THETA` 和 `N_PHI` 的值（如 91×180）以提高速度。

---

### 8. `import pandas` 报错

**原因**：缺少 pandas 依赖。

**解决**：
```bash
pip install pandas
```

---

## ⚠️ 注意事项

### 单位制

| 量 | 内部单位 | 说明 |
|----|----------|------|
| 坐标/距离 | mm | 所有位置参数以毫米为单位 |
| 照度 | lux (lm/m²) | 自动从 mm 换算为 m 后计算 |
| 光强 | cd | 直接使用 IES 文件中的坎德拉值 |
| 光通量 | lm | 通过球面积分得到 |

### IES 文件兼容性

- 支持 **IESNA:LM-63-2002** 标准格式
- 要求文件中包含 `TILT=NONE` 标记
- 支持 Type C 光度类型（垂直角 + 水平角配光）
- 对非旋转对称配光，目前使用第一个水平角的数据（忽略水平变化）
- 如果 IES 文件中 `lumensPerLamp` 与坎德拉数据矛盾，以坎德拉数据为准

### 性能建议

| 场景 | 建议配置 | 预计耗时 |
|------|----------|----------|
| 快速预览 | N_THETA=46, N_PHI=90, PLANE_N=101 | ~10 秒 |
| 标准精度 | N_THETA=181, N_PHI=360, PLANE_N=301 | ~1-3 分钟 |
| 高精度 | N_THETA=361, N_PHI=720, PLANE_N=501 | ~5-10 分钟 |

### 光源模型假设

- **点光源近似**：每个光源视为理想点光源，不考虑光源几何尺寸
- **无相互反射**：不考虑光源间相互反射或接收屏的二次辐射
- **无吸收/散射**：光线传播路径中的空气吸收和散射忽略不计
- **远场近似**：平方反比定律适用于远场条件（距离 >> 光源尺寸）

### 浏览器兼容

- Gradio UI 支持 Chrome、Firefox、Safari、Edge 等现代浏览器
- 建议使用最新版本浏览器以获得最佳体验
- 移动端浏览器可正常访问但布局可能需缩放

### macOS 专属

- 如果遇到 matplotlib 缓存问题，请设置 `MPLCONFIGDIR` 环境变量
- 建议使用系统自带 Python 3.9+ 或通过 Homebrew 安装

### Windows 专属

- 虚拟环境激活命令：`.venv\Scripts\activate`
- 路径分隔符为 `\`，在脚本中注意转义
- 建议使用 PowerShell 或 Windows Terminal

---

## 📜 技术栈

- **Python** 3.9+ — 核心语言
- **NumPy** — 数值计算与矩阵运算
- **Matplotlib** — 2D/3D 可视化
- **Gradio** 6.x — Web 图形界面
- **pandas** — 数据处理
- **IESNA:LM-63-2002** — 光度数据标准

---

## 📄 License

MIT License — 本项目仅供学习和研究使用。

---

*生成于 2026 年 · Optical Simulation System v1.0*
