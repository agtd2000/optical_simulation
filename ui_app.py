#!/usr/bin/env python3
"""
光学仿真 UI 应用 — Gradio 现代界面
支持: IES文件上传, 自定义光源(位置+角度), 球面/平面接收屏, 交互式可视化
"""

import os, sys, json, traceback
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px

# ─── 为确保matplotlib不出缓存问题 ───
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = '/tmp/mplcache'

# ============================================================
# 1.  IES 解析
# ============================================================
def parse_ies(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.strip().split('\n')
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'TILT=NONE':
            data_start = i
            break
    data_lines = []
    for line in lines[data_start:]:
        s = line.strip()
        if s and s != 'TILT=NONE':
            data_lines.append(s)
    tokens = ' '.join(data_lines).split()
    header = tokens[0:10]
    nV = int(header[3])
    nH = int(header[4])
    data_tokens = [float(t) for t in tokens[13:]]
    v_ang = np.array(data_tokens[:nV])
    h_ang = np.array(data_tokens[nV:nV+nH])
    raw = np.array(data_tokens[nV+nH:])
    intensity = raw.reshape(nH, nV) * float(header[2])
    return {
        'vAngles': v_ang, 'hAngles': h_ang,
        'intensity': intensity,
        'candelaMult': float(header[2]),
        'lumensPerLamp': float(header[1]),
        'maxIntensity': float(np.max(intensity)),
        'isSymmetric': all((intensity[i]==intensity[0]).all() for i in range(nH)),
    }

# ============================================================
# 2.  带光源朝向的照度计算
# ============================================================
def _source_local_z(pitch_deg, yaw_deg):
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    return np.array([
        np.sin(p) * np.cos(y),
        np.sin(p) * np.sin(y),
        np.cos(p)
    ])

def _build_ies_lut(ies):
    v = ies['vAngles']
    Iv = ies['intensity'][0, :]
    lut_theta = np.linspace(0, 180, 1801)
    lut_I = np.interp(lut_theta, v, Iv)
    return lut_theta, lut_I

# ============================================================
# 3.  核心仿真函数
# ============================================================
def run_simulation(ies_path, sources_df, n_rays_per_source,
                   receiver_type, sphere_radius, n_theta, n_phi,
                   plane_z, plane_range, plane_n, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    ies = parse_ies(ies_path)
    lut_th, lut_I = _build_ies_lut(ies)

    sources = []
    for _, row in sources_df.iterrows():
        pos = np.array([float(row['x']), float(row['y']), float(row['z'])])
        local_z = _source_local_z(float(row['pitch_deg']), float(row['yaw_deg']))
        sources.append({'pos': pos, 'local_z': local_z})

    if receiver_type == 'sphere':
        return _run_sphere(ies, lut_th, lut_I, sources, n_rays_per_source,
                          sphere_radius, n_theta, n_phi, out_dir)
    else:
        return _run_plane(ies, lut_th, lut_I, sources, n_rays_per_source,
                         plane_z, plane_range, plane_n, out_dir)

# ─── 球面仿真 ───
def _run_sphere(ies, lut_th, lut_I, sources, n_rays,
                radius, n_theta, n_phi, out_dir):
    R = float(radius)
    R_m = R / 1000.0
    n_src = len(sources)

    th1 = np.linspace(0, np.pi, n_theta)
    ph1 = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(th1, ph1, indexing='ij')
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # 照度计算
    E = np.zeros((n_theta, n_phi))
    for src in sources:
        dx = (X - src['pos'][0]) / 1000.0
        dy = (Y - src['pos'][1]) / 1000.0
        dz = (Z - src['pos'][2]) / 1000.0
        r_sq = dx*dx + dy*dy + dz*dz
        r = np.sqrt(r_sq)

        cos_theta = (dx*src['local_z'][0] + dy*src['local_z'][1] + dz*src['local_z'][2]) / r
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_ies = np.degrees(np.arccos(cos_theta))

        idx = np.clip((theta_ies / 0.1).astype(int), 0, len(lut_I)-1)
        I = lut_I[idx]

        Px, Py, Pz = X/1000.0, Y/1000.0, Z/1000.0
        cos_alpha = (Px*dx + Py*dy + Pz*dz) / (R_m * r)
        cos_alpha = np.clip(cos_alpha, 0.0, 1.0)
        E += I / r_sq * cos_alpha

    # ── 静态图 (用于文件输出) ──
    # 3D 球面 (matplotlib)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(vmin=0, vmax=np.max(E))
    colors = cm.viridis(norm(E))
    stride = max(1, min(n_theta//60, n_phi//120))
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=stride, cstride=stride,
                    alpha=0.95, linewidth=0, antialiased=True)
    for s in sources:
        ax.scatter(*s['pos'], color='red', s=80, marker='o', edgecolors='white', zorder=5)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array(E)
    fig.colorbar(mappable, ax=ax, shrink=0.55, label='Illuminance (lux)')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    lim = R*1.08; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect([1,1,1]); ax.view_init(25, 45)
    plt.tight_layout(); plt.savefig(f'{out_dir}/irradiance_3d.png', dpi=150, bbox_inches='tight'); plt.close()
    # 2D 投影
    fig, ax = plt.subplots(figsize=(16, 6))
    extent = [0, 360, 180, 0]
    ax.imshow(E, extent=extent, aspect='auto', origin='upper', cmap='viridis', interpolation='bilinear')
    ax.set_xlabel('Azimuth φ (°)'); ax.set_ylabel('Polar angle θ (°)')
    ax.set_title('Sphere Irradiance – Cylindrical Projection')
    plt.tight_layout(); plt.savefig(f'{out_dir}/irradiance_2d.png', dpi=150, bbox_inches='tight'); plt.close()
    # 配光曲线
    I_avg = np.mean(E, axis=1) * R_m**2
    th_deg = np.degrees(th1)
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.plot(np.radians(th_deg), I_avg, 'b-', lw=2)
    ax1.fill(np.radians(th_deg), I_avg, alpha=0.25, color='steelblue')
    ax1.set_title('Polar Candela Distribution', pad=15)
    ax2 = fig.add_subplot(122)
    ax2.plot(th_deg, I_avg, 'b-', lw=2); ax2.fill_between(th_deg, I_avg, alpha=0.25, color='steelblue')
    ax2.set_xlabel('θ (°)'); ax2.set_ylabel('Intensity (cd)'); ax2.grid(True, alpha=0.25); ax2.set_xlim(0, 180)
    hm = np.max(I_avg)/2
    ax2.axhline(y=hm, color='gray', ls=':', alpha=0.5)
    plt.tight_layout(); plt.savefig(f'{out_dir}/candela_distribution.png', dpi=150, bbox_inches='tight'); plt.close()
    # 光线追迹
    _gen_ray_trace(sources, R, n_rays, f'{out_dir}/ray_tracing_3d.png')

    # 总光通量
    d_th = np.pi/(n_theta-1); d_ph = 2*np.pi/n_phi
    flux = 0.0
    for i in range(n_theta):
        flux += np.sum(E[i,:]) * R_m**2 * np.sin(th1[i]) * d_th * d_ph

    summary = {
        'receiver': 'sphere', 'sphere_radius_mm': R, 'n_sources': n_src,
        'peak_irradiance_lux': float(np.max(E)),
        'mean_irradiance_lux': float(np.mean(E)),
        'total_flux_lm': float(flux),
        'peak_intensity_cd': float(np.max(I_avg)),
        'single_lamp_peak_cd': float(ies['maxIntensity']),
    }
    hm = np.max(I_avg)/2
    for i in range(len(th_deg)-1):
        if (I_avg[i]-hm)*(I_avg[i+1]-hm) <= 0:
            t0,t1=th_deg[i],th_deg[i+1]; I0,I1=I_avg[i],I_avg[i+1]
            summary['half_max_angle_deg'] = float(t0 + (hm-I0)*(t1-t0)/(I1-I0)) if I1 != I0 else float(t0)
            break
    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── 交互式 Plotly 图 ──
    plots = {
        'irradiance_3d': f'{out_dir}/irradiance_3d.png',
        'irradiance_2d': f'{out_dir}/irradiance_2d.png',
        'ray_tracing_3d': f'{out_dir}/ray_tracing_3d.png',
        'candela_distribution': f'{out_dir}/candela_distribution.png',
    }
    fig_3d = _plotly_sphere_3d(X, Y, Z, E, sources, R)
    fig_2d = _plotly_sphere_2d(E, th1, ph1)
    fig_ray = None  # 光线追迹保持静态
    fig_ray = _plotly_ray_sphere(sources, R, n_rays)
    fig_candela = _plotly_candela(I_avg, th_deg)
    plotly_figs = {'3d': fig_3d, '2d': fig_2d, 'ray': fig_ray, 'candela': fig_candela}
    fig_3d.write_html(f'{out_dir}/irradiance_3d_interactive.html')
    fig_2d.write_html(f'{out_dir}/irradiance_2d_interactive.html')

    return plots, plotly_figs, summary


# ─── 平面仿真 ───
def _run_plane(ies, lut_th, lut_I, sources, n_rays,
               z_plane, extent, n, out_dir):
    Z_mm = float(z_plane)

    x1 = np.linspace(-extent, extent, n)
    y1 = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x1, y1)
    Z = np.full_like(X, Z_mm)

    E = np.zeros((n, n))
    for src in sources:
        dx = (X - src['pos'][0]) / 1000.0
        dy = (Y - src['pos'][1]) / 1000.0
        dz = (Z - src['pos'][2]) / 1000.0
        r_sq = dx*dx + dy*dy + dz*dz
        r = np.sqrt(r_sq)

        cos_theta = (dx*src['local_z'][0] + dy*src['local_z'][1] + dz*src['local_z'][2]) / r
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_ies = np.degrees(np.arccos(cos_theta))

        idx = np.clip((theta_ies / 0.1).astype(int), 0, len(lut_I)-1)
        I = lut_I[idx]

        cos_alpha = np.clip(dz / r, 0.0, 1.0)
        E += I / r_sq * cos_alpha

    # ── 静态图 (用于文件输出) ──
    # 2D 伪彩 + 截面
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1 = axes[0]
    extent_mm = [-extent, extent, -extent, extent]
    im = ax1.imshow(E, extent=extent_mm, origin='lower', cmap='viridis',
                    aspect='equal', interpolation='bilinear')
    fig.colorbar(im, ax=ax1, shrink=0.85, label='Illuminance (lux)')
    levels = np.linspace(0, np.max(E), 11)
    cs = ax1.contour(X, Y, E, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    ax1.clabel(cs, fmt='%.0f', fontsize=7, colors='white')
    for s in sources:
        ax1.scatter(*s['pos'][:2], color='red', marker='o', s=40, edgecolors='white', zorder=5)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_title(f'Plane at Z={Z_mm:.0f} mm'); ax1.set_aspect('equal')
    ax2 = axes[1]; mid = n // 2
    ax2.plot(x1, E[mid,:], 'b-', lw=1.5, label='X-axis (Y=0)')
    ax2.plot(y1, E[:,mid], 'r--', lw=1.5, label='Y-axis (X=0)')
    hm = np.max(E)/2; ax2.axhline(y=hm, color='gray', ls=':', alpha=0.5)
    ax2.set_xlabel('Position (mm)'); ax2.set_ylabel('Illuminance (lux)')
    ax2.set_title('Cross-section'); ax2.grid(True, alpha=0.25); ax2.legend()
    plt.tight_layout(); plt.savefig(f'{out_dir}/irradiance_2d.png', dpi=150, bbox_inches='tight'); plt.close()
    # 3D 曲面
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    stride = max(1, n//80)
    surf = ax.plot_surface(X[::stride,::stride], Y[::stride,::stride], E[::stride,::stride],
                           cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.95)
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Illuminance (lux)')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Illuminance (lux)')
    ax.set_title(f'3D Plane (Z={Z_mm:.0f} mm)')
    plt.tight_layout(); plt.savefig(f'{out_dir}/irradiance_3d.png', dpi=150, bbox_inches='tight'); plt.close()
    # 光线追迹
    _gen_ray_trace_plane(sources, Z_mm, extent, n_rays, f'{out_dir}/ray_tracing_3d.png')

    # 总通量
    dA = ((2*extent/1000)/(n-1))**2
    flux = float(np.sum(E) * dA)
    peak_intensity = float(np.max(E) * (Z_mm/1000)**2)

    # 光斑直径
    e_mid = E[n//2, :]; hm = np.max(E)/2
    left_i = np.argmax(e_mid[:n//2] >= hm) if np.any(e_mid[:n//2] >= hm) else 0
    right_i = n//2 + np.argmax(e_mid[n//2:] < hm) if np.any(e_mid[n//2:] < hm) else n-1
    spot_d = float(2 * extent * (right_i - left_i) / n) if left_i > 0 and right_i > n//2 else None

    summary = {
        'receiver': 'plane', 'plane_z_mm': Z_mm, 'plane_range_mm': extent,
        'n_sources': len(sources),
        'peak_irradiance_lux': float(np.max(E)),
        'mean_irradiance_lux': float(np.mean(E)),
        'total_flux_on_plane_lm': flux,
        'spot_diameter_mm_FWHM': spot_d,
        'approx_peak_intensity_cd': peak_intensity,
    }
    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    plots = {
        'irradiance_3d': f'{out_dir}/irradiance_3d.png',
        'irradiance_2d': f'{out_dir}/irradiance_2d.png',
        'ray_tracing_3d': f'{out_dir}/ray_tracing_3d.png',
        'candela_distribution': f'{out_dir}/irradiance_2d.png',  # 平面复用2D
    }
    fig_3d = _plotly_plane_3d(X, Y, E, sources)
    fig_2d = _plotly_plane_2d(X, Y, E, sources, extent)
    fig_ray = None
    fig_ray = _plotly_ray_plane(sources, Z_mm, extent, n_rays)
    fig_candela = _plotly_candela_plane(X, Y, E, sources, extent)
    plotly_figs = {'3d': fig_3d, '2d': fig_2d, 'ray': fig_ray, 'candela': fig_candela}
    fig_3d.write_html(f'{out_dir}/irradiance_3d_interactive.html')
    fig_2d.write_html(f'{out_dir}/irradiance_2d_interactive.html')

    return plots, plotly_figs, summary


# ─── 光线追迹 (球面/平面) ───
def _gen_ray_trace(sources, R, n_rays, out_path):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2*np.pi, 36); v = np.linspace(0, np.pi, 18)
    ax.plot_wireframe(R*np.outer(np.cos(u), np.sin(v)),
                      R*np.outer(np.sin(u), np.sin(v)),
                      R*np.outer(np.ones_like(u), np.cos(v)),
                      alpha=0.08, color='gray', linewidth=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))
    ray_list = []
    for si, src in enumerate(sources):
        cnt = 0; att = 0
        while cnt < n_rays and att < n_rays*200:
            att += 1
            th = np.random.uniform(0, 90); ph = np.random.uniform(0, 360)
            th_r = np.radians(th); ph_r = np.radians(ph)
            d_local = np.array([np.sin(th_r)*np.cos(ph_r), np.sin(th_r)*np.sin(ph_r), np.cos(th_r)])
            if np.random.random() < 1.0 - th/90.0:
                lz = src['local_z']; ly = np.array([-lz[1], lz[0], 0.0])
                if np.linalg.norm(ly) < 1e-10: ly = np.array([0.0, 1.0, 0.0])
                lx = np.cross(ly, lz)
                ly = ly/np.linalg.norm(ly); lx = lx/np.linalg.norm(lx)
                d_world = d_local[0]*lx + d_local[1]*ly + d_local[2]*lz
                cnt += 1
                s = src['pos']; b = 2*np.dot(s, d_world); c = np.dot(s,s) - R**2; disc = b*b - 4*c
                if disc >= 0:
                    t = min(t for t in [(-b+np.sqrt(disc))/2, (-b-np.sqrt(disc))/2] if t > 1e-8)
                    ray_list.append((si, s, s + t*d_world))
    step = max(1, len(ray_list)//400)
    for i, (si, s, hit) in enumerate(ray_list[::step]):
        ax.plot([s[0], hit[0]], [s[1], hit[1]], [s[2], hit[2]],
                color=colors[si%len(sources)], alpha=0.12, linewidth=0.6)
    for si, src in enumerate(sources):
        ax.scatter(*src['pos'], color=colors[si], s=100, marker='o', edgecolors='white', zorder=5)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    lim = R*1.05; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_box_aspect([1,1,1]); ax.view_init(20, 50)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()

def _gen_ray_trace_plane(sources, Z_mm, extent, n_rays, out_path):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    xp = np.linspace(-extent, extent, 20); yp = np.linspace(-extent, extent, 20)
    Xp, Yp = np.meshgrid(xp, yp)
    ax.plot_wireframe(Xp, Yp, np.full_like(Xp, Z_mm), alpha=0.12, color='gray', linewidth=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))
    ray_list = []
    for si, src in enumerate(sources):
        cnt = 0; att = 0
        while cnt < n_rays and att < n_rays*200:
            att += 1
            th = np.random.uniform(0, 90); ph = np.random.uniform(0, 360)
            th_r = np.radians(th); ph_r = np.radians(ph)
            d_local = np.array([np.sin(th_r)*np.cos(ph_r), np.sin(th_r)*np.sin(ph_r), np.cos(th_r)])
            if np.random.random() < 1.0 - th/90.0:
                lz = src['local_z']; ly = np.array([-lz[1], lz[0], 0.0])
                if np.linalg.norm(ly) < 1e-10: ly = np.array([0.0, 1.0, 0.0])
                lx = np.cross(ly, lz); ly = ly/np.linalg.norm(ly); lx = lx/np.linalg.norm(lx)
                d_world = d_local[0]*lx + d_local[1]*ly + d_local[2]*lz
                cnt += 1
                s = src['pos']
                if abs(d_world[2]) > 1e-10:
                    t = (Z_mm - s[2]) / d_world[2]
                    if t > 0:
                        hit = s + t*d_world
                        if abs(hit[0]) <= extent and abs(hit[1]) <= extent:
                            ray_list.append((si, s, hit))
    step = max(1, len(ray_list)//400)
    for i, (si, s, hit) in enumerate(ray_list[::step]):
        ax.plot([s[0], hit[0]], [s[1], hit[1]], [s[2], hit[2]],
                color=colors[si%len(sources)], alpha=0.12, linewidth=0.6)
    for si, src in enumerate(sources):
        ax.scatter(*src['pos'], color=colors[si], s=100, marker='o', edgecolors='white', zorder=5)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    lim = max(extent, Z_mm)*1.1
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(min(0,Z_mm-100), max(Z_mm+100,100))
    ax.set_box_aspect([1,1,0.8]); ax.view_init(20, 50)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


# ============================================================
# 4.  Plotly 交互式可视化 (所有类型)
# ============================================================
def _fig_defaults(fig, title='', height=550):
    """为 Plotly 图设置统一外观：隐藏模式栏(去掉分享按钮)、设置边距"""
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode='closest',
        modebar=dict(bgcolor='rgba(0,0,0,0)', color='rgba(0,0,0,0)',
                     activecolor='rgba(0,0,0,0)'),
    )
    return fig

def _plotly_sphere_3d(X, Y, Z, E, sources, R):
    stride = max(1, min(X.shape[0]//60, X.shape[1]//120))
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X[::stride,::stride], y=Y[::stride,::stride], z=Z[::stride,::stride],
        surfacecolor=E[::stride,::stride],
        colorscale='Viridis', showscale=True,
        colorbar=dict(title='lux', x=1.02, len=0.8),
    ))
    for si, s in enumerate(sources):
        fig.add_trace(go.Scatter3d(
            x=[s['pos'][0]], y=[s['pos'][1]], z=[s['pos'][2]],
            mode='markers', marker=dict(size=8, color='red', symbol='circle',
                                         line=dict(color='white', width=1)),
            name=f'S{si+1}', showlegend=(si == 0)))
    lim = R*1.12
    fig.update_layout(
        scene=dict(xaxis=dict(range=[-lim,lim], title='X (mm)'),
                   yaxis=dict(range=[-lim,lim], title='Y (mm)'),
                   zaxis=dict(range=[-lim,lim], title='Z (mm)'),
                   aspectmode='cube',
                   camera=dict(eye=dict(x=1.8,y=1.5,z=1.2))))
    return _fig_defaults(fig, 'Sphere 3D — 拖拽旋转 · 滚轮缩放', 600)

def _plotly_sphere_2d(E, th1, ph1):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=E, x=np.degrees(ph1), y=np.degrees(th1),
        colorscale='Viridis', colorbar=dict(title='lux'),
        hovertemplate='θ=%{y:.1f}°<br>φ=%{x:.1f}°<br>E=%{z:.1f} lux<extra></extra>',
    ))
    fig.update_layout(
        xaxis=dict(title='Azimuth φ (°)', range=[0, 360]),
        yaxis=dict(title='Polar angle θ (°)', range=[180, 0], autorange='reversed'),
        dragmode='zoom',
    )
    return _fig_defaults(fig, 'Sphere 2D Projection — 拖拽平移 · 滚轮缩放')

def _plotly_plane_3d(X, Y, E, sources):
    stride = max(1, X.shape[0]//80)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X[::stride,::stride], y=Y[::stride,::stride],
        z=E[::stride,::stride],
        colorscale='Viridis', showscale=True, colorbar=dict(title='lux'),
    ))
    for s in sources:
        fig.add_trace(go.Scatter3d(
            x=[s['pos'][0]], y=[s['pos'][1]], z=[0],
            mode='markers', marker=dict(size=6, color='red'),
            name='Source', showlegend=False))
    fig.update_layout(
        scene=dict(xaxis=dict(title='X (mm)'), yaxis=dict(title='Y (mm)'),
                   zaxis=dict(title='lux'), aspectmode='manual',
                   aspectratio=dict(x=1, y=1, z=0.6),
                   camera=dict(eye=dict(x=1.8,y=1.5,z=1.0))))
    return _fig_defaults(fig, 'Plane 3D — 拖拽旋转 · 滚轮缩放', 600)

def _plotly_plane_2d(X, Y, E, sources, extent):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=E, x=X[0,:], y=Y[:,0],
        colorscale='Viridis', colorbar=dict(title='lux'),
        hovertemplate='X=%{x:.1f} mm<br>Y=%{y:.1f} mm<br>E=%{z:.1f} lux<extra></extra>',
    ))
    # 等照度线 (用 contour trace)
    levels = np.linspace(np.min(E)*0.1, np.max(E), 10)
    fig.add_trace(go.Contour(
        z=E, x=X[0,:], y=Y[:,0],
        contours=dict(start=levels[0], end=levels[-1], size=levels[1]-levels[0]),
        line=dict(width=0.5, color='white'),
        showscale=False, opacity=0.4,
        hovertemplate='X=%{x:.1f}<br>Y=%{y:.1f}<br>E=%{z:.1f}<extra></extra>',
    ))
    for s in sources:
        fig.add_trace(go.Scatter(
            x=[s['pos'][0]], y=[s['pos'][1]],
            mode='markers', marker=dict(size=10, color='red', symbol='circle',
                                         line=dict(color='white', width=1)),
            name='Source', showlegend=False))
    fig.update_layout(
        xaxis=dict(title='X (mm)', scaleanchor='y'),
        yaxis=dict(title='Y (mm)'),
        dragmode='zoom',
    )
    return _fig_defaults(fig, f'Plane 2D (±{extent:.0f} mm) — 拖拽选框缩放 · 滚轮缩放')


# ─── 光线追迹 (Plotly) ───
def _plotly_ray_sphere(sources, R, n_rays):
    fig = go.Figure()
    # 半透明球面线框
    u = np.linspace(0, 2*np.pi, 24); v = np.linspace(0, np.pi, 12)
    for i in range(len(u)-1):
        for j in range(len(v)-1):
            x = [[R*np.sin(v[j])*np.cos(u[i]), R*np.sin(v[j+1])*np.cos(u[i])],
                 [R*np.sin(v[j])*np.cos(u[i+1]), R*np.sin(v[j+1])*np.cos(u[i+1])]]
            y = [[R*np.sin(v[j])*np.sin(u[i]), R*np.sin(v[j+1])*np.sin(u[i])],
                 [R*np.sin(v[j])*np.sin(u[i+1]), R*np.sin(v[j+1])*np.sin(u[i+1])]]
            z = [[R*np.cos(v[j]), R*np.cos(v[j+1])],
                 [R*np.cos(v[j]), R*np.cos(v[j+1])]]
            fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.04,
                                     showscale=False, hoverinfo='skip',
                                     colorscale=[[0,'gray'],[1,'gray']]))
    # 生成光线
    ray_data = []
    for si, src in enumerate(sources):
        cnt = 0; att = 0
        while cnt < n_rays and att < n_rays*200:
            att += 1
            th = np.random.uniform(0, 90); ph = np.random.uniform(0, 360)
            th_r = np.radians(th); ph_r = np.radians(ph)
            d_local = np.array([np.sin(th_r)*np.cos(ph_r),
                                np.sin(th_r)*np.sin(ph_r), np.cos(th_r)])
            if np.random.random() < 1.0 - th/90.0:
                lz = src['local_z']; ly = np.array([-lz[1], lz[0], 0.0])
                if np.linalg.norm(ly) < 1e-10: ly = np.array([0.0, 1.0, 0.0])
                lx = np.cross(ly, lz); ly /= np.linalg.norm(ly); lx /= np.linalg.norm(lx)
                d_world = d_local[0]*lx + d_local[1]*ly + d_local[2]*lz
                cnt += 1
                s = src['pos']; b = 2*np.dot(s, d_world); c = np.dot(s,s)-R**2; disc = b*b-4*c
                if disc >= 0:
                    t = min(t for t in [(-b+np.sqrt(disc))/2, (-b-np.sqrt(disc))/2] if t > 1e-8)
                    hit = s + t*d_world
                    ray_data.append((si, s, hit))
    colors = px.colors.qualitative.Plotly
    # 按光源分组，每组用一条 Scatter3d (None 分隔) 避免数千个独立 trace
    step = max(1, len(ray_data)//400)
    sampled = ray_data[::step]
    for si_idx in range(len(sources)):
        xs, ys, zs = [], [], []
        for (rsi, s, hit) in sampled:
            if rsi == si_idx:
                xs += [s[0], hit[0], None]
                ys += [s[1], hit[1], None]
                zs += [s[2], hit[2], None]
        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines',
                line=dict(color=colors[si_idx % len(colors)], width=1),
                opacity=0.15, hoverinfo='skip', showlegend=False))
    for si, src in enumerate(sources):
        fig.add_trace(go.Scatter3d(
            x=[src['pos'][0]], y=[src['pos'][1]], z=[src['pos'][2]],
            mode='markers', marker=dict(size=8, color='red',
                                         line=dict(color='white', width=1)),
            showlegend=False))
    lim = R*1.05
    fig.update_layout(scene=dict(xaxis=dict(range=[-lim,lim]), yaxis=dict(range=[-lim,lim]),
                                 zaxis=dict(range=[-lim,lim]), aspectmode='cube',
                                 camera=dict(eye=dict(x=1.5,y=1.2,z=1.0))))
    return _fig_defaults(fig, f'Ray Tracing ({len(ray_data)} rays) — 拖拽旋转', 600)

def _plotly_ray_plane(sources, Z_mm, extent, n_rays):
    fig = go.Figure()
    # 平面
    fig.add_trace(go.Surface(
        x=[[-extent, extent],[ -extent, extent]],
        y=[[-extent, -extent],[extent, extent]],
        z=[[Z_mm, Z_mm],[Z_mm, Z_mm]],
        opacity=0.08, showscale=False, hoverinfo='skip',
        colorscale=[[0,'gray'],[1,'gray']]))
    ray_data = []
    for si, src in enumerate(sources):
        cnt = 0; att = 0
        while cnt < n_rays and att < n_rays*200:
            att += 1
            th = np.random.uniform(0, 90); ph = np.random.uniform(0, 360)
            th_r = np.radians(th); ph_r = np.radians(ph)
            d_local = np.array([np.sin(th_r)*np.cos(ph_r),
                                np.sin(th_r)*np.sin(ph_r), np.cos(th_r)])
            if np.random.random() < 1.0 - th/90.0:
                lz = src['local_z']; ly = np.array([-lz[1], lz[0], 0.0])
                if np.linalg.norm(ly) < 1e-10: ly = np.array([0.0, 1.0, 0.0])
                lx = np.cross(ly, lz); ly /= np.linalg.norm(ly); lx /= np.linalg.norm(lx)
                d_world = d_local[0]*lx + d_local[1]*ly + d_local[2]*lz
                cnt += 1
                s = src['pos']
                if abs(d_world[2]) > 1e-10:
                    t = (Z_mm - s[2]) / d_world[2]
                    if t > 0:
                        hit = s + t*d_world
                        if abs(hit[0]) <= extent and abs(hit[1]) <= extent:
                            ray_data.append((si, s, hit))
    colors = px.colors.qualitative.Plotly
    # 按光源分组
    step = max(1, len(ray_data)//400)
    sampled = ray_data[::step]
    for si_idx in range(len(sources)):
        xs, ys, zs = [], [], []
        for (rsi, s, hit) in sampled:
            if rsi == si_idx:
                xs += [s[0], hit[0], None]
                ys += [s[1], hit[1], None]
                zs += [s[2], hit[2], None]
        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines',
                line=dict(color=colors[si_idx % len(colors)], width=1),
                opacity=0.15, hoverinfo='skip', showlegend=False))
    for si, src in enumerate(sources):
        fig.add_trace(go.Scatter3d(
            x=[src['pos'][0]], y=[src['pos'][1]], z=[src['pos'][2]],
            mode='markers', marker=dict(size=8, color='red',
                                         line=dict(color='white', width=1)),
            showlegend=False))
    lim = max(extent, Z_mm)*1.1
    fig.update_layout(scene=dict(xaxis=dict(range=[-lim,lim]), yaxis=dict(range=[-lim,lim]),
                                 zaxis=dict(range=[-lim,lim]), aspectmode='manual',
                                 aspectratio=dict(x=1,y=1,z=0.6),
                                 camera=dict(eye=dict(x=1.5,y=1.2,z=1.0))))
    return _fig_defaults(fig, f'Ray Tracing to Plane ({len(ray_data)} rays) — 拖拽旋转', 600)

def _plotly_candela(I_avg, th_deg):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'polar'}, {'type': 'xy'}]],
                        subplot_titles=('Polar', 'Cartesian'))
    fig.add_trace(go.Scatterpolar(r=I_avg, theta=th_deg, mode='lines',
                                  fill='toself', line=dict(color='blue', width=2),
                                  name='Intensity'), row=1, col=1)
    fig.add_trace(go.Scatter(x=th_deg, y=I_avg, mode='lines',
                             fill='tozeroy', line=dict(color='blue', width=2),
                             name='Intensity'), row=1, col=2)
    hm = np.max(I_avg)/2
    # 用 shape 代替 add_hline (避免 subplot 兼容问题)
    fig.add_shape(type='line', x0=0, x1=180, y0=hm, y1=hm,
                  line=dict(color='gray', dash='dot', width=1),
                  row=1, col=2)
    fig.update_xaxes(title='θ (°)', range=[0, 180], row=1, col=2)
    fig.update_yaxes(title='Intensity (cd)', row=1, col=2)
    fig.update_layout(showlegend=False)
    return _fig_defaults(fig, 'Candela Distribution — 拖拽平移 · 滚轮缩放')


def _plotly_candela_plane(X, Y, E, sources, extent):
    """平面模式下的'配光曲线': 显示沿 X 和 Y 轴的截面照度曲线"""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('X-axis (Y=0)', 'Y-axis (X=0)'))
    mid = X.shape[0] // 2
    x1 = X[0, :]; y1 = Y[:, 0]
    fig.add_trace(go.Scatter(x=x1, y=E[mid, :], mode='lines',
                              fill='tozeroy', line=dict(color='blue', width=2),
                              name='X截面'), row=1, col=1)
    fig.add_trace(go.Scatter(x=y1, y=E[:, mid], mode='lines',
                              fill='tozeroy', line=dict(color='red', width=2),
                              name='Y截面'), row=1, col=2)
    hm = np.max(E)/2
    for col in [1, 2]:
        fig.add_shape(type='line', x0=-extent, x1=extent,
                      y0=hm, y1=hm, line=dict(color='gray', dash='dot', width=1),
                      row=1, col=col)
    fig.update_xaxes(title='Position (mm)', row=1, col=1)
    fig.update_xaxes(title='Position (mm)', row=1, col=2)
    fig.update_yaxes(title='Illuminance (lux)', row=1, col=1)
    fig.update_yaxes(title='Illuminance (lux)', row=1, col=2)
    fig.update_layout(showlegend=True)
    return _fig_defaults(fig, f'Plane Cross-section — 拖拽缩放 · FWHM={hm*2:.0f} lux')
def build_default_sources(n):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return [[float(round(10*np.cos(a), 2)), float(round(10*np.sin(a), 2)), 0.0, 0.0, 0.0]
            for a in angles]

def on_n_sources_change(n):
    n = int(n); data = build_default_sources(n)
    return gr.Dataframe(value=data, headers=['x','y','z','pitch_deg','yaw_deg'],
                        datatype=['number']*5, row_count=n, column_count=(5,'fixed'))

def on_receiver_change(rtype):
    vis_s = rtype == 'sphere'; vis_p = rtype == 'plane'
    return (gr.update(visible=vis_s),)*3 + (gr.update(visible=vis_p),)*3

def run_click(ies_file, n_sources, src_df, n_rays,
              recv_type, sphere_r, n_th, n_ph,
              plane_z, plane_range, plane_n):
    if ies_file is None:
        return [None]*4 + [{k: None for k in ['3d','2d','ray','candela']}, "请上传 IES 文件"]

    try:
        import pandas as pd
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = f'outputs/{timestamp}'
        os.makedirs(out_dir, exist_ok=True)
        expected_cols = ['x','y','z','pitch_deg','yaw_deg']

        # 统一 Dataframe 输入格式
        if isinstance(src_df, pd.DataFrame):
            df = src_df.copy()
        elif isinstance(src_df, dict):
            if 'data' in src_df:
                h = src_df.get('headers', expected_cols)
                df = pd.DataFrame(src_df['data'], columns=h)
            else:
                df = pd.DataFrame(src_df)
                if list(df.columns) != expected_cols and df.shape[1] == 5:
                    df.columns = expected_cols
        elif isinstance(src_df, (list, tuple)):
            df = pd.DataFrame(src_df, columns=expected_cols)
        else:
            raise ValueError(f"不支持的光源数据格式: {type(src_df).__name__}")

        if list(df.columns) != expected_cols and df.shape[1] == 5:
            df.columns = expected_cols
        for col in expected_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        plots, plotly_figs, summary = run_simulation(
            ies_path=ies_file.name, sources_df=df, n_rays_per_source=int(n_rays),
            receiver_type=recv_type, sphere_radius=float(sphere_r),
            n_theta=int(n_th), n_phi=int(n_ph),
            plane_z=float(plane_z), plane_range=float(plane_range), plane_n=int(plane_n),
            out_dir=out_dir,
        )
        summary['output_dir'] = os.path.abspath(out_dir)
        summary_str = json.dumps(summary, indent=2, ensure_ascii=False)

        return (
            plots.get('irradiance_3d'),
            plots.get('irradiance_2d'),
            plots.get('ray_tracing_3d'),
            plots.get('candela_distribution'),
            plotly_figs,  # dict: {'3d': fig, '2d': fig, 'ray': None, 'candela': fig}
            summary_str,
        )
    except Exception as e:
        traceback.print_exc()
        return [None]*4 + [{k: None for k in ['3d','2d','ray','candela']}, f"Error: {str(e)}"]


# ─── 构建 UI ───
def build_ui():
    with gr.Blocks(title="Optical Simulation UI") as demo:

        gr.Markdown("""
        # 🔦 光学仿真系统
        <p style="font-size:0.95rem; color:#666;">
        支持自定义 IES 光源、球面/平面接收屏、多光源排列与朝向 · 所有图形支持鼠标拖拽 / 滚轮缩放
        </p>
        """)

        with gr.Row(equal_height=False):
            # ─── 左侧: 配置 ───
            with gr.Column(scale=2, variant='panel'):
                gr.Markdown("### 📂 配置面板")

                with gr.Accordion("📄 IES 光源文件", open=True):
                    ies_file = gr.File(label="上传 IES 文件", file_types=['.ies'],
                                       value='LTE-C1726-ZH-GL.ies')

                with gr.Accordion("💡 光源配置", open=True):
                    n_sources = gr.Slider(minimum=1, maximum=24, value=6, step=1, label="光源数量")
                    src_df = gr.Dataframe(
                        headers=['x', 'y', 'z', 'pitch_deg', 'yaw_deg'],
                        datatype=['number']*5,
                        value=build_default_sources(6),
                        row_count=6, column_count=(5, 'fixed'),
                        label="光源位置与朝向 (pitch=0 → +Z)", interactive=True)
                    n_sources.change(fn=on_n_sources_change, inputs=n_sources, outputs=src_df)
                    n_rays = gr.Number(value=200, label="每光源光线数", minimum=10, maximum=2000, step=10)

                with gr.Accordion("📐 接收屏", open=True):
                    recv_type = gr.Radio(choices=['sphere', 'plane'], value='sphere', label="接收屏类型")
                    with gr.Column(visible=True) as sphere_group:
                        sphere_r = gr.Number(value=500, label="球面半径 (mm)", minimum=10, maximum=10000, step=10)
                        with gr.Row():
                            n_th = gr.Number(value=181, label="极角采样数", minimum=10, maximum=721, step=1)
                            n_ph = gr.Number(value=361, label="方位角采样数", minimum=10, maximum=720, step=1)
                    with gr.Column(visible=False) as plane_group:
                        plane_z = gr.Number(value=500, label="平面 Z 距离 (mm)", minimum=1, maximum=10000, step=10)
                        plane_range = gr.Number(value=1500, label="XY 采样范围 ± (mm)", minimum=10, maximum=10000, step=10)
                        plane_n = gr.Number(value=301, label="XY 每轴采样点数", minimum=10, maximum=1001, step=1)

                recv_type.change(fn=on_receiver_change, inputs=recv_type,
                                 outputs=[sphere_group, n_th, n_ph, plane_group, plane_range, plane_n])
                run_btn = gr.Button("🚀 运行仿真", variant='primary', size='lg')

            # ─── 右侧: 结果 ───
            with gr.Column(scale=3, variant='panel'):
                gr.Markdown("### 📊 结果")
                result_selector = gr.Radio(
                    choices=[
                        ('3D 照度分布', 'irradiance_3d'),
                        ('2D 照度分布', 'irradiance_2d'),
                        ('光线追迹',     'ray_tracing_3d'),
                        ('配光曲线',     'candela_distribution'),
                    ],
                    value='irradiance_3d', label="选择可视化",
                )

                result_plot = gr.Plot(label="仿真结果 (交互式)", visible=True)
                result_img = gr.Image(label="静态图片 (存档)", visible=False)

                # 所有可视化都用 Plotly, 通过 state 切换不同图形
                plotly_state = gr.State({'3d': None, '2d': None, 'ray': None, 'candela': None})

                def update_plot(choice, pstate):
                    key_map = {'irradiance_3d': '3d', 'irradiance_2d': '2d',
                               'ray_tracing_3d': 'ray', 'candela_distribution': 'candela'}
                    return pstate.get(key_map.get(choice))

                result_selector.change(fn=update_plot,
                                       inputs=[result_selector, plotly_state],
                                       outputs=result_plot)

                summary_out = gr.Textbox(label="结果摘要", lines=10, max_lines=20)

                # ─── 运行回调 ───
                def run_and_update(ies, n_src, df, n_r, rtype,
                                   sr, nt, np_, pz, pr, pn):
                    i3d, i2d, ray, candela, plotly_dict, summary = run_click(
                        ies, n_src, df, n_r, rtype, sr, nt, np_, pz, pr, pn)
                    first_fig = (plotly_dict or {}).get('3d')
                    return plotly_dict, first_fig, summary if summary else ""

                run_btn.click(
                    fn=run_and_update,
                    inputs=[ies_file, n_sources, src_df, n_rays,
                            recv_type, sphere_r, n_th, n_ph,
                            plane_z, plane_range, plane_n],
                    outputs=[plotly_state, result_plot, summary_out],
                )

        gr.Markdown("""
        ---
        <p style="text-align:center;color:#999;font-size:0.85rem;">
        Optical Simulation System · 3D/2D 图形支持鼠标拖拽旋转/平移 · 右键可保存图片
        </p>
        """)

    return demo


if __name__ == '__main__':
    demo = build_ui()
    demo.launch(server_port=7860, share=False, show_error=True,
                theme=gr.themes.Soft(
                    primary_hue="blue", secondary_hue="slate",
                    neutral_hue="slate", font=gr.themes.GoogleFont('Inter'),
                ))
