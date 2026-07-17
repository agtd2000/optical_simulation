#!/usr/bin/env python3
"""
光学仿真: 6×LTE-C1726-ZH-GL 圆周排布 (Z=0平面) + 球面接收屏 (R=500)
所有光源沿 +Z 方向照射

输出:
  1. 3D 球面照度分布图
  2. 2D 等距圆柱投影照度图
  3. 光线追迹 3D 可视化
  4. 总配光曲线（极坐标+笛卡尔坐标）
  5. 数值数据 CSV + JSON 摘要
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import json
import os

# =====================================================
# 配置参数
# =====================================================
IES_FILE = "LTE-C1726-ZH-GL.ies"
R_SPHERE = 500.0          # 球面接收屏半径 (mm)
N_SOURCES = 6             # 光源数量
R_CIRCLE = 10.0           # 光源排布圆周半径 (mm)
N_THETA = 181             # 极角采样数 (0~180°, 每1°一点)
N_PHI = 361               # 方位角采样数 (0~360°, 每1°一点)
N_RAYS_PER_SOURCE = 500   # 每个光源发射的光线数 (用于追迹图)

# 平面接收屏参数
Z_PLANE = 500.0        # 平面 Z 坐标 (mm)
PLANE_RANGE = 1500.0   # 平面 XY 采样范围: ±1500 mm
PLANE_N = 301          # 平面 XY 每轴采样点数


# =====================================================
# 1. IESNA LM-63-2002 解析器
# =====================================================
def parse_ies(filepath):
    """解析 IESNA:LM-63-2002 格式光度数据文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.strip().split('\n')

    # 定位 TILT=NONE 后的数据起始
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'TILT=NONE':
            data_start = i
            break

    # 合并所有数据行（去掉元数据行）
    data_lines = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped and stripped != 'TILT=NONE':
            data_lines.append(stripped)

    all_tokens = ' '.join(data_lines).split()

    # tokens[0:10] = IES 数据头
    # tokens[10:13] = 灯泡几何参数 (lamp数量/长度/宽度)
    # tokens[13:]   = 垂直角度 + 水平角度 + 坎德拉值
    header_tokens = all_tokens[0:10]
    nLamps = int(header_tokens[0])
    lumensPerLamp = float(header_tokens[1])
    candelaMult = float(header_tokens[2])
    nVAngles = int(header_tokens[3])
    nHAngles = int(header_tokens[4])
    photometricType = int(header_tokens[5])
    units = int(header_tokens[6])     # 1=英尺, 2=米
    width = float(header_tokens[7])
    length = float(header_tokens[8])
    height = float(header_tokens[9])

    # 跳过 IES 头 (10) + 几何参数 (3) = 13，剩余为角度+光强数据
    data_tokens = [float(t) for t in all_tokens[13:]]

    # 垂直角度 (0~180°)
    v_angles = np.array(data_tokens[:nVAngles])
    # 水平角度 (0~360°)
    h_angles = np.array(data_tokens[nVAngles:nVAngles + nHAngles])
    # 坎德拉值 (nHAngles × nVAngles)
    raw = np.array(data_tokens[nVAngles + nHAngles:])
    intensity = raw.reshape(nHAngles, nVAngles) * candelaMult

    return {
        'nLamps': nLamps,
        'lumensPerLamp': lumensPerLamp,
        'candelaMult': candelaMult,
        'vAngles': v_angles,
        'hAngles': h_angles,
        'intensity': intensity,        # (nH, nV)
        'photometricType': photometricType,
        'units': units,
    }


# =====================================================
# 2. IES 光强插值
# =====================================================
def _interp_1d(x, xp, fp):
    """一维线性插值（标量）"""
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    i = np.searchsorted(xp, x) - 1
    i = max(0, min(i, len(xp) - 2))
    t = (x - xp[i]) / (xp[i + 1] - xp[i]) if xp[i + 1] > xp[i] else 0.0
    return float(fp[i] + t * (fp[i + 1] - fp[i]))


def interpolate_intensity(theta, phi, ies_data):
    """
    从 IES 数据插值获取 (theta, phi) 方向的光强 (cd)

    参数:
        theta: 垂直角 0~180° (0° = +Z, 180° = -Z)
        phi:   水平角 0~360°
    返回:
        光强值 (cd)
    """
    v_ang = ies_data['vAngles']
    I_mat = ies_data['intensity']

    # IES 数据旋转对称 (所有水平角相同)，直接取第一行
    I_v = I_mat[0, :]

    # 夹紧 theta
    theta = float(np.clip(theta, v_ang[0], v_ang[-1]))

    # --- 垂直角线性插值 ---
    vi = np.searchsorted(v_ang, theta) - 1
    vi = max(0, min(vi, len(v_ang) - 2))
    v0, v1 = vi, vi + 1
    dv = v_ang[v1] - v_ang[v0]
    w_v = (theta - v_ang[v0]) / dv if dv > 0 else 0.0

    I = (1 - w_v) * float(I_v[v0]) + w_v * float(I_v[v1])
    return I


# =====================================================
# 3. 几何计算
# =====================================================
def get_source_positions(n=None, radius=None):
    """计算 N 个光源在 XY 平面圆周上的位置"""
    if n is None:
        n = N_SOURCES
    if radius is None:
        radius = R_CIRCLE
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [np.array([radius * np.cos(a), radius * np.sin(a), 0.0]) for a in angles]


def get_sphere_samples(R=None, n_theta=None, n_phi=None):
    """生成球面上的采样网格点 (经纬度网格)"""
    if R is None:
        R = R_SPHERE
    if n_theta is None:
        n_theta = N_THETA
    if n_phi is None:
        n_phi = N_PHI

    theta_1d = np.linspace(0, np.pi, n_theta)     # 极角 0~π
    phi_1d   = np.linspace(0, 2 * np.pi, n_phi)    # 方位角 0~2π
    THETA, PHI = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    return X, Y, Z, THETA, PHI


# =====================================================
# 4. 照度计算引擎
# =====================================================
def calculate_irradiance(sources, X, Y, Z, ies_data, verbose=True):
    """
    计算球面上各点的总照度 (lux)
    """
    n_theta, n_phi = X.shape
    E_total = np.zeros((n_theta, n_phi))

    # 预计算 IES 插值表 (垂直角 0~180°, 每 0.1° 一个采样点)
    v_ang = ies_data['vAngles']
    I_v = ies_data['intensity'][0, :]
    theta_lut = np.linspace(0, 180, 1801)  # 0.1° 分辨率
    I_lut = np.interp(theta_lut, v_ang, I_v)

    for idx, src in enumerate(sources):
        # 光源到球面点的向量 (mm → m)
        dx = (X - src[0]) / 1000.0
        dy = (Y - src[1]) / 1000.0
        dz = (Z - src[2]) / 1000.0
        r_sq = dx**2 + dy**2 + dz**2
        r = np.sqrt(r_sq)

        # 方向单位向量
        nz = dz / r  # dir · z_hat

        # IES 垂直角: 从 +Z 轴测量
        nz = np.clip(nz, -1.0, 1.0)
        theta_ies = np.degrees(np.arccos(nz))

        # 查表获取光强
        idx_lut = np.clip((theta_ies / 0.1).astype(int), 0, len(I_lut) - 1)
        I = I_lut[idx_lut]

        # 入射角余弦: cos(α) = (P · (P - S)) / (R * r)  (单位已统一为 m)
        P_x = X / 1000.0
        P_y = Y / 1000.0
        P_z = Z / 1000.0
        cos_alpha = (P_x * dx + P_y * dy + P_z * dz) / ((R_SPHERE / 1000.0) * r)
        cos_alpha = np.clip(cos_alpha, 0.0, 1.0)

        # 照度贡献 E = I / r² * cos(α)  (lux = cd/m²)
        E_contrib = I / r_sq * cos_alpha
        E_total += E_contrib

        if verbose:
            print(f"  光源 {idx+1}: ({src[0]:.1f}, {src[1]:.1f}, {src[2]:.1f}) "
                  f"=> 贡献 {np.sum(E_contrib):.6f} lux·m²")

    return E_total


# =====================================================
# 5. 可视化: 3D 球面照度分布
# =====================================================
def plot_3d_sphere(X, Y, Z, E, sources, output_path='sphere_irradiance_3d.png'):
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')

    norm = Normalize(vmin=0, vmax=np.max(E))
    colors = cm.viridis(norm(E))

    # 绘制球面 (隔点采样以提升性能)
    stride = max(1, min(N_THETA // 60, N_PHI // 120))
    surf = ax.plot_surface(X, Y, Z, facecolors=colors,
                           rstride=stride, cstride=stride,
                           alpha=0.95, linewidth=0, antialiased=True)

    # 颜色条
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array(E)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08)
    cbar.set_label('Illuminance (lux)', fontsize=11)

    # 光源位置
    src_x = [s[0] for s in sources]
    src_y = [s[1] for s in sources]
    src_z = [s[2] for s in sources]
    ax.scatter(src_x, src_y, src_z, color='red', s=90,
               label='Light Sources', marker='o', edgecolors='white', linewidths=0.5)

    # 光源圆周
    th_c = np.linspace(0, 2 * np.pi, 80)
    ax.plot(R_CIRCLE * np.cos(th_c), R_CIRCLE * np.sin(th_c),
            np.zeros_like(th_c), 'r--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Z (mm)', fontsize=10)
    ax.set_title(f'3D Sphere Irradiance Distribution\n'
                 f'6 sources on XY circle (R={R_CIRCLE} mm), Sphere R={R_SPHERE} mm',
                 fontsize=12)
    ax.legend(fontsize=9)

    lim = R_SPHERE * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


# =====================================================
# 6. 可视化: 2D 照度投影图 (等距圆柱投影)
# =====================================================
def plot_2d_projection(THETA, PHI, E, output_path='sphere_irradiance_2d.png'):
    fig, ax = plt.subplots(figsize=(18, 6))

    # THETA: (n_theta, n_phi), PHI: (n_theta, n_phi)
    theta_deg = np.degrees(THETA[:, 0])   # 0~180
    phi_deg   = np.degrees(PHI[0, :])     # 0~360

    extent = [0, 360, 180, 0]   # [left, right, bottom, top]
    im = ax.imshow(E, extent=extent, aspect='auto', origin='upper',
                   cmap='viridis', interpolation='bilinear')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Illuminance (lux)', fontsize=11)

    ax.set_xlabel('Azimuth φ (degrees)', fontsize=11)
    ax.set_ylabel('Polar angle θ (degrees)', fontsize=11)
    ax.set_title('2D Cylindrical Projection of Sphere Irradiance\n'
                 '(θ = 0° = +Z top, θ = 180° = −Z bottom)',
                 fontsize=12)

    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.set_yticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°', '360°'])
    ax.set_yticklabels(['0° (+Z)', '30°', '60°', '90° (XY)', '120°', '150°', '180° (−Z)'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


# =====================================================
# 7. 光线追迹 (蒙特卡洛)
# =====================================================
def generate_rays(sources, ies_data, n_per_source=None):
    """
    使用拒绝采样法生成光线: 每个光源发射 n_per_source 条光线,
    方向按 IES 光强分布抽样。
    """
    if n_per_source is None:
        n_per_source = N_RAYS_PER_SOURCE

    max_I = float(np.max(ies_data['intensity']))
    rays = []

    for idx, src in enumerate(sources):
        count = 0
        attempts = 0
        max_attempts = n_per_source * 200

        while count < n_per_source and attempts < max_attempts:
            attempts += 1
            # 在 0~90° 采样 (IES 在 90° 以上为 0)
            theta = np.random.uniform(0, 90)
            phi   = np.random.uniform(0, 360)

            I = interpolate_intensity(theta, phi, ies_data)
            if np.random.random() < I / max_I:
                count += 1
                th_r = np.radians(theta)
                ph_r = np.radians(phi)
                # 光源本地坐标: theta=0 → +Z
                d = np.array([np.sin(th_r) * np.cos(ph_r),
                              np.sin(th_r) * np.sin(ph_r),
                              np.cos(th_r)])
                rays.append({
                    'source_idx': idx,
                    'source_pos': src.copy(),
                    'direction': d,
                    'intensity': I,
                })

    return rays


def trace_rays_to_sphere(rays, R=None):
    """光线与球面求交 (直线-球面交点)"""
    if R is None:
        R = R_SPHERE

    traced = []
    for ray in rays:
        s = ray['source_pos']
        d = ray['direction']

        # |s + t*d|² = R²  =>  t² + 2(s·d)t + |s|² - R² = 0
        b = 2.0 * np.dot(s, d)
        c = np.dot(s, s) - R**2
        disc = b**2 - 4.0 * c

        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            t1 = (-b + sqrt_disc) / 2.0
            t2 = (-b - sqrt_disc) / 2.0
            candidates = [t for t in (t1, t2) if t > 1e-8]
            if candidates:
                t = min(candidates)
                hit = s + t * d
                traced.append({**ray, 't': t, 'hit_point': hit})

    return traced


def plot_ray_tracing(sources, traced_rays, output_path='ray_tracing_3d.png'):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 半透明球面
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    Xs = R_SPHERE * np.outer(np.cos(u), np.sin(v))
    Ys = R_SPHERE * np.outer(np.sin(u), np.sin(v))
    Zs = R_SPHERE * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(Xs, Ys, Zs, alpha=0.08, color='gray', linewidth=0.3)

    # 光源
    src_x = [s[0] for s in sources]
    src_y = [s[1] for s in sources]
    src_z = [s[2] for s in sources]
    ax.scatter(src_x, src_y, src_z, color='red', s=120,
               label='Light Sources', marker='o', edgecolors='white', linewidths=0.5, zorder=5)

    # 光源圆周
    th_c = np.linspace(0, 2 * np.pi, 80)
    ax.plot(R_CIRCLE * np.cos(th_c), R_CIRCLE * np.sin(th_c),
            np.zeros_like(th_c), 'r--', alpha=0.4, linewidth=0.8)

    # 最多显示 300 条光线
    step = max(1, len(traced_rays) // 300)
    colors = plt.cm.tab10(np.linspace(0, 1, N_SOURCES))

    for i, ray in enumerate(traced_rays[::step]):
        src = ray['source_pos']
        hit = ray['hit_point']
        c = colors[ray['source_idx'] % N_SOURCES]
        # 光线线段
        ax.plot([src[0], hit[0]], [src[1], hit[1]], [src[2], hit[2]],
                color=c, alpha=0.12, linewidth=0.6)

    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Z (mm)', fontsize=10)
    ax.set_title(f'Ray Tracing Visualization\n'
                 f'{len(traced_rays)} rays traced (showing 1/{step})',
                 fontsize=12)
    ax.legend(fontsize=9)

    lim = R_SPHERE * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=50)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


# =====================================================
# 8. 配光曲线分析
# =====================================================
def calculate_total_candela(E, THETA):
    """从球面照度反推远场总光强: I_total(θ) = E(θ, φ) * R², 再对方位角平均"""
    # E: (n_theta, n_phi), 单位转换 mm→m
    I_avg = np.mean(E, axis=1) * (R_SPHERE / 1000.0)**2
    theta_deg = np.degrees(THETA[:, 0])
    return I_avg, theta_deg


def plot_candela_distribution(I_avg, theta_deg, theta_half_max=None,
                              output_path='candela_distribution.png'):
    fig = plt.figure(figsize=(16, 6))

    # --- 左: 极坐标 ---
    ax1 = fig.add_subplot(121, projection='polar')
    th_r = np.radians(theta_deg)
    ax1.plot(th_r, I_avg, 'b-', linewidth=2, label='Total Intensity')
    ax1.fill(th_r, I_avg, alpha=0.25, color='steelblue')
    ax1.set_title('Polar Candela Distribution\n(azimuthal average)', fontsize=12,
                  pad=15)
    ax1.set_ylim(0, np.max(I_avg) * 1.15)
    ax1.set_yticks([])
    ax1.set_xticks(np.radians([0, 30, 60, 90, 120, 150, 180]))
    ax1.set_xticklabels(['0°\n(+Z)', '30°', '60°', '90°', '120°', '150°', '180°\n(−Z)'])
    ax1.legend(loc='upper right', fontsize=9)

    # --- 右: 笛卡尔 ---
    ax2 = fig.add_subplot(122)
    ax2.plot(theta_deg, I_avg, 'b-', linewidth=2, label='Total Intensity')
    ax2.fill_between(theta_deg, I_avg, alpha=0.25, color='steelblue')

    # 半高标记
    half_max = np.max(I_avg) / 2
    ax2.axhline(y=half_max, color='gray', linestyle=':', alpha=0.6, linewidth=1)
    ax2.text(theta_deg[-1] * 0.02, half_max * 1.05,
             f'50% = {half_max:.1f} cd', color='gray', fontsize=9)

    if theta_half_max is not None:
        ax2.axvline(x=theta_half_max, color='r', linestyle='--', alpha=0.7, linewidth=1.2)
        ax2.text(theta_half_max + 1.5, half_max * 0.9,
                 f'FWHM = {theta_half_max:.1f}°', color='r', fontsize=10, fontweight='bold')

    # 6× 峰值标注（单个光源峰值 * 6）
    ax2.axhline(y=np.max(I_avg), color='green', linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.text(theta_deg[-1] * 0.02, np.max(I_avg) * 1.02,
             f'Peak = {np.max(I_avg):.1f} cd', color='green', fontsize=9)

    ax2.set_xlabel('Polar angle θ (degrees)', fontsize=11)
    ax2.set_ylabel('Total Intensity (cd)', fontsize=11)
    ax2.set_title('Cartesian Candela Distribution\n(azimuthal average)',
                  fontsize=12)
    ax2.set_xlim(0, 180)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


# =====================================================
# 9. 数值数据输出
# =====================================================
def compute_total_flux(E, THETA):
    """球面积分计算总光通量: Φ = ∫∫ E(θ,φ) * R² * sin(θ) dθ dφ  (lux → lm)"""
    n_theta, n_phi = E.shape
    theta_1d = THETA[:, 0]   # 0~π
    d_theta = np.pi / (n_theta - 1)
    d_phi   = 2 * np.pi / n_phi

    flux = 0.0
    for i in range(n_theta):
        sin_th = np.sin(theta_1d[i])
        for j in range(n_phi):
            dA = (R_SPHERE / 1000.0)**2 * sin_th * d_theta * d_phi  # m²
            flux += E[i, j] * dA
    return flux


def save_data(E, THETA, PHI, I_avg, theta_deg, theta_half_max, summary_extra=None,
              prefix='simulation_data'):
    """保存 CSV + JSON"""
    n_theta, n_phi = E.shape
    theta_1d = np.degrees(THETA[:, 0])
    phi_1d   = np.degrees(PHI[0, :])

    # --- 照度 CSV (抽样, 避免文件过大) ---
    # 每 5° 抽样
    step_th = max(1, n_theta // 37)
    step_ph = max(1, n_phi // 72)

    with open(f'{prefix}_irradiance.csv', 'w') as f:
        f.write('theta_deg,phi_deg,irradiance_lux\n')
        for i in range(0, n_theta, step_th):
            for j in range(0, n_phi, step_ph):
                f.write(f'{theta_1d[i]:.2f},{phi_1d[j]:.2f},{E[i, j]:.8f}\n')

    # --- 配光曲线 CSV ---
    with open(f'{prefix}_candela.csv', 'w') as f:
        f.write('theta_deg,intensity_cd\n')
        for i in range(len(theta_deg)):
            f.write(f'{theta_deg[i]:.2f},{I_avg[i]:.6f}\n')

    # --- JSON 摘要 ---
    flux = compute_total_flux(E, THETA)

    summary = {
        'config': {
            'ies_file': IES_FILE,
            'n_sources': N_SOURCES,
            'source_circle_radius_mm': R_CIRCLE,
            'sphere_radius_mm': R_SPHERE,
            'source_orientation': '+Z (IES vertical 0° = +Z)',
            'sampling': f'{N_THETA}×{N_PHI} (theta×phi)',
        },
        'results': {
            'max_irradiance_lux': float(np.max(E)),
            'min_irradiance_lux': float(np.min(E)),
            'mean_irradiance_lux': float(np.mean(E)),
            'total_flux_on_sphere_lm': float(flux),
            'peak_total_intensity_cd': float(np.max(I_avg)),
            'half_max_angle_deg_FWHM': (float(theta_half_max)
                                         if theta_half_max is not None else None),
        },
    }
    if summary_extra:
        summary['results'].update(summary_extra)

    with open(f'{prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✓ 已保存数据文件:")
    print(f"    {prefix}_irradiance.csv  (抽样照度数据)")
    print(f"    {prefix}_candela.csv     (配光曲线数据)")
    print(f"    {prefix}_summary.json    (结果摘要)")

    return summary


def calculate_plane_irradiance(sources, ies_data, z_mm=Z_PLANE,
                                extent=PLANE_RANGE, n=PLANE_N):
    """
    计算 Z=z_mm 处 XY 平面上的照度分布 (lux)

    返回:
        X, Y, E: 坐标网格 (mm) 和照度 (lux)
    """
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_mm)

    # 预计算 IES 插值表
    v_ang = ies_data['vAngles']
    I_v = ies_data['intensity'][0, :]
    theta_lut = np.linspace(0, 180, 1801)
    I_lut = np.interp(theta_lut, v_ang, I_v)

    E_total = np.zeros_like(X)

    for idx, src in enumerate(sources):
        # 光源到平面点向量 (mm → m)
        dx = (X - src[0]) / 1000.0
        dy = (Y - src[1]) / 1000.0
        dz = (Z - src[2]) / 1000.0
        r_sq = dx**2 + dy**2 + dz**2
        r = np.sqrt(r_sq)

        # IES 垂直角 (0° = +Z)
        nz = dz / r
        nz = np.clip(nz, -1.0, 1.0)
        theta_ies = np.degrees(np.arccos(nz))

        # 查光强
        idx_lut = np.clip((theta_ies / 0.1).astype(int), 0, len(I_lut) - 1)
        I = I_lut[idx_lut]

        # 入射角余弦: 平面法线 = (0,0,1), 入射方向单位向量 = (dx,dy,dz)/r
        # cos(α) = (dx,dy,dz)/r · (0,0,1) = dz/r
        cos_alpha = np.clip(dz / r, 0.0, 1.0)

        # 照度 E = I / r² * cos(α)  (lux)
        E_total += I / r_sq * cos_alpha

        if idx == 0:
            peak_c = np.max(I / r_sq * cos_alpha)
            print(f"  光源 {idx+1}: 峰值贡献 {peak_c:.2f} lux")

    return X, Y, E_total


def plot_plane_irradiance_2d(X, Y, E, output_path='plane_irradiance_2d.png'):
    """平面照度 2D 伪彩色图 + 等照度线"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左: 伪彩色图 ---
    ax1 = axes[0]
    extent_mm = [X.min(), X.max(), Y.min(), Y.max()]
    im = ax1.imshow(E, extent=extent_mm, origin='lower',
                    cmap='viridis', aspect='equal',
                    interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cbar.set_label('Illuminance (lux)', fontsize=10)

    # 等照度线
    levels = np.linspace(0, np.max(E), 11)
    cs = ax1.contour(X, Y, E, levels=levels,
                     colors='white', linewidths=0.5, alpha=0.5)
    ax1.clabel(cs, fmt='%.0f', fontsize=7, colors='white')

    # 标记光源位置
    sources = get_source_positions()
    src_x = [s[0] for s in sources]
    src_y = [s[1] for s in sources]
    ax1.scatter(src_x, src_y, color='red', marker='o', s=40,
                edgecolors='white', linewidths=0.5, zorder=5,
                label='Sources')

    ax1.set_xlabel('X (mm)', fontsize=11)
    ax1.set_ylabel('Y (mm)', fontsize=11)
    ax1.set_title(f'Plane Illuminance at Z={Z_PLANE:.0f} mm\n'
                  f'(pseudo-color + contour)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8, loc='upper right')

    # --- 右: 截面曲线 (沿 X 和 Y) ---
    ax2 = axes[1]
    mid = PLANE_N // 2
    x_axis = X[mid, :]
    y_axis = Y[:, mid]
    e_x = E[mid, :]
    e_y = E[:, mid]

    ax2.plot(x_axis, e_x, 'b-', linewidth=1.5, label='Along X-axis (Y=0)')
    ax2.plot(y_axis, e_y, 'r--', linewidth=1.5, label='Along Y-axis (X=0)')
    ax2.set_xlabel('Position (mm)', fontsize=11)
    ax2.set_ylabel('Illuminance (lux)', fontsize=11)
    ax2.set_title('Cross-section Profiles', fontsize=12)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=9)

    # 半高标记
    half_max = np.max(E) / 2
    ax2.axhline(y=half_max, color='gray', linestyle=':', alpha=0.5)
    ax2.text(X.max()*0.02, half_max*1.05, f'50% = {half_max:.0f} lux',
             color='gray', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")

    return x_axis, e_x  # 返回 X 轴截面供后续分析


def plot_plane_irradiance_3d(X, Y, E, output_path='plane_irradiance_3d.png'):
    """平面照度 3D 曲面图"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 降采样以提升绘图性能
    stride = max(1, PLANE_N // 80)
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    Es = E[::stride, ::stride]

    surf = ax.plot_surface(Xs, Ys, Es, cmap='viridis',
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=True, alpha=0.95)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Illuminance (lux)', fontsize=10)

    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Illuminance (lux)', fontsize=10)
    ax.set_title(f'3D Plane Illuminance Surface\n(Z={Z_PLANE:.0f} mm, {N_SOURCES} sources)',
                 fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def save_plane_data(X, Y, E, x_axis, e_x, prefix='plane_data'):
    """保存平面照度数据"""
    # 完整的照度网格 (降采样以控制文件大小)
    step = max(1, PLANE_N // 100)
    with open(f'{prefix}_irradiance.csv', 'w') as f:
        f.write('x_mm,y_mm,irradiance_lux\n')
        for i in range(0, PLANE_N, step):
            for j in range(0, PLANE_N, step):
                f.write(f'{X[i,j]:.2f},{Y[i,j]:.2f},{E[i,j]:.6f}\n')

    # X 轴截面
    with open(f'{prefix}_cross_section.csv', 'w') as f:
        f.write('x_mm,irradiance_lux\n')
        for i in range(len(x_axis)):
            f.write(f'{x_axis[i]:.2f},{e_x[i]:.6f}\n')

    # 分析特征
    mid = PLANE_N // 2
    peak = float(np.max(E))
    half_max = peak / 2

    # 光斑直径 (半高处)
    # 沿 X 轴找到左右半高点
    left_idx = np.argmax(e_x[:mid] >= half_max)
    right_idx = mid + np.argmax(e_x[mid:] < half_max)
    if left_idx > 0 and right_idx > mid:
        # 线性插值精确定位
        x_arr = x_axis
        # 左半高
        for i in range(mid-1, -1, -1):
            if e_x[i] <= half_max and e_x[i+1] > half_max:
                x_left = x_arr[i] + (half_max - e_x[i]) * (x_arr[i+1] - x_arr[i]) / (e_x[i+1] - e_x[i])
                break
        else:
            x_left = -PLANE_RANGE
        # 右半高
        for i in range(mid, len(e_x)-1):
            if e_x[i] >= half_max and e_x[i+1] < half_max:
                x_right = x_arr[i] + (half_max - e_x[i]) * (x_arr[i+1] - x_arr[i]) / (e_x[i+1] - e_x[i])
                break
        else:
            x_right = PLANE_RANGE
        spot_diameter = x_right - x_left
    else:
        spot_diameter = None

    # 平面总光通量: Φ = ∫ E(x,y) dx dy (lux * m² → lm)
    dA = ((2 * PLANE_RANGE / 1000) / (PLANE_N - 1))**2  # m² per pixel
    total_flux = float(np.sum(E) * dA)

    summary = {
        'config': {
            'plane_z_mm': Z_PLANE,
            'plane_range_mm': PLANE_RANGE,
            'sampling': f'{PLANE_N}×{PLANE_N}',
        },
        'results': {
            'peak_irradiance_lux': peak,
            'mean_irradiance_lux': float(np.mean(E)),
            'half_max_irradiance_lux': half_max,
            'spot_diameter_mm_FWHM': (float(spot_diameter)
                                       if spot_diameter is not None else None),
            'total_flux_on_plane_lm': total_flux,
        },
    }

    with open(f'{prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✓ 已保存平面数据:")
    print(f"    {prefix}_irradiance.csv")
    print(f"    {prefix}_cross_section.csv")
    print(f"    {prefix}_summary.json")

    return summary
# =====================================================
def main():
    np.random.seed(42)

    print("=" * 64)
    print("  光 学 仿 真  —  6×LTE-C1726-ZH-GL  圆周排布 + 球面+平面接收屏")
    print("=" * 64)

    # ---- [1] 解析 IES ----
    print("\n[1/8] 解析 IES 文件 ...")
    ies_data = parse_ies(IES_FILE)
    v_min, v_max = ies_data['vAngles'][0], ies_data['vAngles'][-1]
    h_min, h_max = ies_data['hAngles'][0], ies_data['hAngles'][-1]
    print(f"  灯具类型: {'Type C' if ies_data['photometricType'] == 1 else 'Type B'}")
    print(f"  垂直角范围: {v_min:.0f}° ~ {v_max:.0f}°  ({len(ies_data['vAngles'])} 点)")
    print(f"  水平角范围: {h_min:.0f}° ~ {h_max:.0f}°  ({len(ies_data['hAngles'])} 点)")
    print(f"  坎德拉倍率: {ies_data['candelaMult']}")
    print(f"  最大光强: {np.max(ies_data['intensity']):.4f} cd")
    print(f"  所有水平角数据相同: 旋转对称配光")

    # ---- [2] 光源位置 ----
    print("\n[2/8] 计算光源位置 ...")
    sources = get_source_positions()
    for i, s in enumerate(sources):
        print(f"  光源 {i+1}:  ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})")
    print(f"  朝向: 全部沿 +Z 方向 (垂直于 XY 平面)")

    # ---- [3] 球面采样 ----
    print(f"\n[3/8] 生成球面采样网格 ({N_THETA}×{N_PHI}) ...")
    X, Y, Z, THETA, PHI = get_sphere_samples()
    print(f"  球面半径: {R_SPHERE} mm")

    # ---- [4] 照度计算 ----
    print("\n[4/8] 计算球面照度分布 (遍历 6 个光源)...")
    E = calculate_irradiance(sources, X, Y, Z, ies_data)
    print(f"  ────────────")
    print(f"  最大照度: {np.max(E):.8f} lux")
    print(f"  最小照度: {np.min(E):.8f} lux")
    print(f"  平均照度: {np.mean(E):.8f} lux")

    # ---- [5] 照度分布图 ----
    print("\n[5/8] 生成照度分布可视化 ...")
    plot_3d_sphere(X, Y, Z, E, sources, 'sphere_irradiance_3d.png')
    plot_2d_projection(THETA, PHI, E, 'sphere_irradiance_2d.png')

    # ---- [6] 光线追迹 ----
    print("\n[6/8] 光线追迹 ...")
    rays = generate_rays(sources, ies_data)
    traced = trace_rays_to_sphere(rays)
    print(f"  发射: {len(rays)} 条光线")
    print(f"  到达球面: {len(traced)} 条 ({100*len(traced)/len(rays):.1f}%)")
    plot_ray_tracing(sources, traced, 'ray_tracing_3d.png')

    # ---- [7] 配光曲线 + 数据输出 ----
    print("\n[7/8] 配光曲线分析 & 数据输出 ...")
    I_avg, theta_deg = calculate_total_candela(E, THETA)

    # 半高角 (FWHM)
    half_max = np.max(I_avg) / 2
    theta_half = None
    for i in range(len(theta_deg) - 1):
        if (I_avg[i] - half_max) * (I_avg[i+1] - half_max) <= 0:
            t0, t1 = theta_deg[i], theta_deg[i+1]
            I0, I1 = I_avg[i], I_avg[i+1]
            theta_half = t0 + (half_max - I0) * (t1 - t0) / (I1 - I0)
            break

    plot_candela_distribution(I_avg, theta_deg, theta_half, 'candela_distribution.png')

    # 原始 IES 峰值 (单灯)
    single_peak = float(np.max(ies_data['intensity']))
    summary = save_data(E, THETA, PHI, I_avg, theta_deg, theta_half,
                        summary_extra={
                            'single_lamp_peak_intensity_cd': single_peak,
                            'n_lamp_peak_intensity_cd': single_peak * N_SOURCES,
                        },
                        prefix='simulation_data')

    # ---- [8] 平面照度分析 (Z=0.5m) ----
    print("\n[8/8] 平面照度分析 (Z=0.5m) ...")
    print(f"  采样范围: ±{PLANE_RANGE} mm, {PLANE_N}×{PLANE_N} 网格")
    Xp, Yp, Ep = calculate_plane_irradiance(sources, ies_data)
    print(f"  ────────────")
    peak_Ep = np.max(Ep)
    print(f"  平面峰值照度: {peak_Ep:.2f} lux")
    print(f"  平面平均照度: {np.mean(Ep):.2f} lux")

    # 平面照度图
    x_axis, e_x = plot_plane_irradiance_2d(Xp, Yp, Ep, 'plane_irradiance_2d.png')
    plot_plane_irradiance_3d(Xp, Yp, Ep, 'plane_irradiance_3d.png')
    plane_summary = save_plane_data(Xp, Yp, Ep, x_axis, e_x, 'plane_data')

    # ---- 结果摘要 ----
    print("\n" + "=" * 64)
    print("  仿 真 完 成  —  结 果 摘 要")
    print("=" * 64)
    print(f"  光源配置:          6 × LTE-C1726-ZH-GL")
    print(f"  排布:              XY 平面圆周 (R={R_CIRCLE} mm), 朝向 +Z")
    print(f"  接收屏:            球面 (R={R_SPHERE} mm)")
    print(f"  ────────────────────────────────────────")
    print(f"  峰值照度 (球面):   {summary['results']['max_irradiance_lux']:.6f} lux")
    print(f"  平均照度 (球面):   {summary['results']['mean_irradiance_lux']:.6f} lux")
    r = summary['results']
    print(f"  球面总光通量:      {r['total_flux_on_sphere_lm']:.4f} lm")
    print(f"  峰值总光强:        {r['peak_total_intensity_cd']:.2f} cd")
    print(f"  单灯光强峰值:      {r['single_lamp_peak_intensity_cd']:.2f} cd")
    print(f"  6× 理想叠加:       {r['n_lamp_peak_intensity_cd']:.2f} cd")
    if r['half_max_angle_deg_FWHM']:
        print(f"  半高角 (FWHM):      {r['half_max_angle_deg_FWHM']:.1f}°")
    pr = plane_summary['results']
    print(f"  ──────── 平面 (Z={Z_PLANE:.0f} mm) ────────")
    print(f"  峰值照度 (平面):   {pr['peak_irradiance_lux']:.1f} lux")
    print(f"  平均照度 (平面):   {pr['mean_irradiance_lux']:.1f} lux")
    if pr['spot_diameter_mm_FWHM']:
        print(f"  光斑直径 (FWHM):    {pr['spot_diameter_mm_FWHM']:.1f} mm")
    print(f"  平面总光通量:      {pr['total_flux_on_plane_lm']:.2f} lm")
    print("=" * 64)
    print("\n输出文件列表:")
    for f in ['sphere_irradiance_3d.png',
              'sphere_irradiance_2d.png',
              'ray_tracing_3d.png',
              'candela_distribution.png',
              'simulation_data_irradiance.csv',
              'simulation_data_candela.csv',
              'simulation_data_summary.json',
              'plane_irradiance_2d.png',
              'plane_irradiance_3d.png',
              'plane_data_irradiance.csv',
              'plane_data_cross_section.csv',
              'plane_data_summary.json']:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✔ {f:45s} ({size/1024:.1f} KB)")
        else:
            print(f"  ✘ {f:45s} (未找到)")

    return summary


if __name__ == '__main__':
    main()
