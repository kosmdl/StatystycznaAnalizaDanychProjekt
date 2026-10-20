# -*- coding: utf-8 -*-
"""
Wykresy do sprawozdania „Przestrzenne zróżnicowanie poziomu życia w województwie małopolskim (2011)”.
Uruchomienie:  python wykresy.py   (w folderze z analiza.py i raw.json)
Wymagane biblioteki:  pip install numpy pandas scipy matplotlib adjustText
Wynik: pliki rys1_eda.png ... rys6_bezrobocie.png w tym samym folderze.
"""
import json, math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text
import analiza as A

HERE = A.HERE
df, Z, X, XALL = A.df, A.Z, A.X, A.XALL
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": False, "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.facecolor": "white"})
BLUE = "#2a78d6"; DARK = "#0d366b"; GREY = "#52514e"; LIGHT = "#e6e6e3"; ORANGE = "#eb6834"
GROUP_COL = {1: "#0d366b", 2: "#2a78d6", 3: "#86b6ef", 4: "#cde2fb"}
GROUP_LBL = {1: "grupa I (najwyższy)", 2: "grupa II", 3: "grupa III", 4: "grupa IV (najniższy)"}
short = {"m. Kraków": "Kraków", "m. Nowy Sącz": "Nowy Sącz", "m. Tarnów": "Tarnów"}
lab = lambda p: short.get(p, p)

# ---------- Rys. 1: EDA – współczynniki zmienności i rozkłady ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
cv = A.desc.loc[XALL, "V (%)"]
cols = [BLUE for x in XALL]
ax.bar(XALL, cv.values, color=cols, width=0.65, edgecolor="none")
ax.axhline(10, color="#e34948", lw=1, ls="--"); ax.text(-0.45, 14, "próg V = 10%", color="#e34948", ha="left", fontsize=7.5)
for x, v in zip(XALL, cv.values):
    ax.text(x, v + 2, f"{v:.0f}", ha="center", fontsize=7.5, color=GREY)
ax.set_ylabel("klasyczny współczynnik zmienności V (%)"); ax.set_ylim(0, 115)
ax.set_title("a) Zmienność cech diagnostycznych", loc="left", fontsize=10, fontweight="bold")
ax.tick_params(axis="x", labelsize=8)
ax = axes[1]
Zall = A.unitaryzacja(df, XALL); Zall.index = df["powiat"]
ax.boxplot([Zall[x].values for x in XALL], widths=0.5, patch_artist=True, showfliers=False,
           medianprops=dict(color=DARK, lw=1.4), boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE, lw=1),
           whiskerprops=dict(color=BLUE, lw=1), capprops=dict(color=BLUE, lw=1))
rng = np.random.default_rng(1)
for i, x in enumerate(XALL, start=1):
    xs = i + rng.uniform(-0.16, 0.16, len(Zall))
    ax.scatter(xs, Zall[x].values, s=9, color=GREY, alpha=0.55, zorder=3, lw=0)
    for city in ["m. Kraków", "m. Nowy Sącz", "m. Tarnów"]:
        j = list(Zall.index).index(city)
        ax.scatter(xs[j], Zall.loc[city, x], s=22, color=ORANGE, zorder=4, lw=0)
ax.set_xticks(range(1, len(XALL)+1)); ax.set_xticklabels(XALL, fontsize=8); ax.set_ylim(-0.03, 1.03)
ax.set_ylabel("wartość zunitaryzowana z$_{ij}$")
ax.set_title("b) Rozkłady wartości zunitaryzowanych (pomarańczowe – miasta na prawach powiatu)", loc="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{HERE}/rys1_eda.png"); plt.close(fig)

# ---------- Rys. 2: macierz korelacji ----------
C = A.corr_p.loc[X, X]
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c0392b"])
fig, ax = plt.subplots(figsize=(5.6, 4.7))
im = ax.imshow(C.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(X))); ax.set_yticks(range(len(X))); ax.set_xticklabels(X, fontsize=8.5); ax.set_yticklabels(X, fontsize=8.5)
for i in range(len(X)):
    for j in range(len(X)):
        v = C.values[i, j]
        if i != j:
            ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "#0b0b0b", fontweight="bold" if abs(v) > 0.5 else "normal")
        else:
            ax.text(j, i, "1", ha="center", va="center", fontsize=7, color="white")
ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.set_label("współczynnik korelacji Pearsona r", fontsize=8); cb.ax.tick_params(labelsize=7.5)
ax.set_title("Macierz korelacji zmiennych diagnostycznych (wartości surowe, n = 22)", loc="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{HERE}/rys2_korelacje.png"); plt.close(fig)
X = A.X

# ---------- Rys. 3: wykres radarowy MR ----------
order_sub = ["krakowski", "m. Kraków", "nowosądecki", "oświęcimski", "tarnowski"]
dd = df.copy(); dd["sub_ord"] = dd["podregion"].map({s: i for i, s in enumerate(order_sub)})
dd = dd.sort_values(["sub_ord", "powiat"])
vals = dd["MR"].values; labels = [lab(p) for p in dd["powiat"]]
n = len(vals); ang = np.linspace(0, 2*np.pi, n, endpoint=False)
fig = plt.figure(figsize=(7.2, 6.6)); ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
ax.plot(np.r_[ang, ang[0]], np.r_[vals, vals[0]], color=DARK, lw=1.8, marker="o", ms=4, zorder=3, label="MR powiatu")
ax.fill(np.r_[ang, ang[0]], np.r_[vals, vals[0]], color=BLUE, alpha=0.12)
m = df["MR"].mean()
ax.plot(np.linspace(0, 2*np.pi, 200), [m]*200, color=ORANGE, lw=1.1, ls="--", zorder=2, label=f"średnia MR dla województwa = {m:.3f}")
ax.set_xticks(ang); ax.set_xticklabels([])
for a, l, v in zip(ang, labels, vals):
    ax.text(a, 0.84, l, ha="center", va="center", fontsize=8.3, color="#0b0b0b")
    ax.text(a, v + 0.04, f"{v:.2f}", ha="center", va="center", fontsize=6.8, color=GREY)
ax.set_ylim(0, 0.9); ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]); ax.set_yticklabels(["0,1", "0,2", "0,3", "0,4", "0,5", "0,6", "0,7"], fontsize=7, color=GREY)
ax.yaxis.grid(color=LIGHT, lw=0.8); ax.xaxis.grid(color=LIGHT, lw=0.6); ax.spines["polar"].set_visible(False)
ax.legend(loc="lower left", bbox_to_anchor=(-0.12, -0.08), fontsize=8, frameon=False)
fig.savefig(f"{HERE}/rys3_radar.png"); plt.close(fig)

# ---------- Rys. 4: ranking MS z progami trzech średnich + wykres zmian pozycji ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 6.0), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
ds = df.sort_values("MS", ascending=True)
ax.barh([lab(p) for p in ds["powiat"]], ds["MS"], color=[GROUP_COL[g] for g in ds["grupa_MS"]], height=0.7, edgecolor="white", lw=0.5)
for i, (v, r) in enumerate(zip(ds["MS"], ds["rank_MR"])):
    ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7.3, color=GREY)
mean_, up_, lo_ = A.thr_MS
for t, name, yy in [(mean_, "średnia", -0.95), (up_, "średnia górna", -0.95), (lo_, "średnia dolna", -2.35)]:
    ax.axvline(t, color=ORANGE, lw=1, ls="--"); ax.text(t, yy, f"{name}\n{t:.3f}", color=ORANGE, fontsize=7, ha="center", va="top")
ax.set_xlim(0, 0.95); ax.set_ylim(-3.8, 21.7); ax.set_xlabel("miernik syntetyczny MS (bezwzorcowy)"); ax.tick_params(axis="y", labelsize=8)
ax.set_title("a) Ranking według MS i podział metodą trzech średnich", loc="left", fontsize=10, fontweight="bold")
ax.legend(handles=[Patch(color=GROUP_COL[g], label=GROUP_LBL[g]) for g in [1, 2, 3, 4]], loc="lower right", fontsize=7.5, frameon=False)
ax = axes[1]
for _, r in df.iterrows():
    shift = r["rank_MS"] - r["rank_MR"]
    col = ORANGE if abs(shift) >= 3 else ("#9a9a95" if shift == 0 else BLUE)
    ax.plot([0, 1], [r["rank_MS"], r["rank_MR"]], color=col, lw=1.6 if abs(shift) >= 3 else 1.1, zorder=2)
    ax.scatter([0, 1], [r["rank_MS"], r["rank_MR"]], s=14, color=col, zorder=3)
    ax.text(-0.04, r["rank_MS"], f"{lab(r['powiat'])}  {r['rank_MS']}.", ha="right", va="center", fontsize=7.6)
    ax.text(1.04, r["rank_MR"], f"{r['rank_MR']}.  {lab(r['powiat'])}", ha="left", va="center", fontsize=7.6)
ax.set_ylim(23.6, -1.7); ax.set_xlim(-0.6, 1.6); ax.axis("off")
ax.text(0, -0.25, "ranking MS\n(bezwzorcowy)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.text(1, -0.25, "ranking MR\n(wzorcowy)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.text(0.5, 23.6, f"τ Kendalla = {A.tau_MS_MR[0]:.3f};  ρ Spearmana = {A.rho_MS_MR[0]:.3f}\npomarańczowe – zmiana pozycji o 3 i więcej miejsc", ha="center", va="top", fontsize=8, color="#0b0b0b")
ax.set_title("b) Zmiany pozycji między rankingami MS i MR", loc="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{HERE}/rys4_ranking.png"); plt.close(fig)

# ---------- Rys. 5: kartogramy ----------
GEO = os.path.join(HERE, "powiaty-medium.geojson")
if not os.path.exists(GEO):   # granice powiatów (GUGiK) – pobierane przy pierwszym uruchomieniu
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/powiaty/powiaty-medium.geojson", GEO)
g = json.load(open(GEO, encoding="utf-8"))
def all_rings(geom):
    if geom["type"] == "Polygon": return [geom["coordinates"][0]]
    return [poly[0] for poly in geom["coordinates"]]
feats = {}
for f in g["features"]:
    nm = f["properties"]["nazwa"].replace("powiat ", "")
    rings = all_rings(f["geometry"]); cx = np.mean([np.mean([p[0] for p in r]) for r in rings])
    key = {"Kraków": "m. Kraków", "Nowy Sącz": "m. Nowy Sącz", "Tarnów": "m. Tarnów"}.get(nm, nm)
    if key in set(df["powiat"]):
        if key == "brzeski" and cx < 19: continue   # brzeski w woj. opolskim
        feats[key] = rings
assert len(feats) == 22, len(feats)
def draw_map(ax, gcol, title):
    for _, r in df.iterrows():
        for ring in feats[r["powiat"]]:
            ax.add_patch(Polygon(ring, closed=True, facecolor=GROUP_COL[r[gcol]], edgecolor="white", lw=0.7))
    for _, r in df.iterrows():
        rings = feats[r["powiat"]]; big = max(rings, key=len); xs = [p[0] for p in big]; ys = [p[1] for p in big]
        cx, cy = np.mean(xs), np.mean(ys)
        col = "white" if r[gcol] in (1, 2) else "#0b0b0b"
        if r["powiat"] in ("m. Tarnów", "m. Nowy Sącz"):
            tx, ty = (cx + 0.22, cy + 0.17) if r["powiat"] == "m. Tarnów" else (cx - 0.33, cy - 0.12)
            ax.annotate(lab(r["powiat"]), xy=(cx, cy), xytext=(tx, ty), fontsize=6.8, ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color="#0b0b0b", lw=0.6), color="#0b0b0b")
            continue
        if r["powiat"] == "krakowski": cx += 0.02; cy += 0.2
        if r["powiat"] == "nowosądecki": cx += 0.02; cy -= 0.14
        if r["powiat"] == "tarnowski": cx += 0.1; cy -= 0.08
        if r["powiat"] == "m. Kraków": cy -= 0.01
        ax.text(cx, cy, lab(r["powiat"]), ha="center", va="center", fontsize=6.8, color=col)
    ax.set_aspect(1/np.cos(np.radians(50))); ax.set_xlim(19.05, 21.5); ax.set_ylim(49.15, 50.55); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
draw_map(axes[0], "grupa_MS", "a) grupy według MS (bezwzorcowy)")
draw_map(axes[1], "grupa_MR", "b) grupy według MR (wzorcowy)")
fig.legend(handles=[Patch(color=GROUP_COL[gg], label=GROUP_LBL[gg]) for gg in [1, 2, 3, 4]], loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(); fig.savefig(f"{HERE}/rys5_kartogramy.png"); plt.close(fig)

# ---------- Rys. 6: MS a stopa bezrobocia ----------
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.scatter(df["stopa_bezrobocia"], df["MS"], s=38, color=[GROUP_COL[g] for g in df["grupa_MS"]], edgecolor=DARK, lw=0.6, zorder=3)
texts = [ax.text(r["stopa_bezrobocia"], r["MS"], lab(r["powiat"]), fontsize=7.2) for _, r in df.iterrows()]
adjust_text(texts, x=df["stopa_bezrobocia"].values, y=df["MS"].values, ax=ax, expand=(1.25, 1.5),
            arrowprops=dict(arrowstyle="-", color="#9a9a95", lw=0.5), force_text=(0.4, 0.6), force_points=(0.3, 0.5))
ax.set_xlabel("stopa bezrobocia rejestrowanego, stan na 31 XII 2011 (%)"); ax.set_ylabel("miernik syntetyczny MS")
ax.text(14.2, 0.76, f"ρ Spearmana = {A.rho_MS_bezr[0]:.3f} (p = {A.rho_MS_bezr[1]:.4f})", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=LIGHT))
ax.set_title("Miernik MS a stopa bezrobocia rejestrowanego w powiatach (2011)", loc="left", fontsize=10, fontweight="bold")
ax.set_xlim(3.5, 21); ax.set_ylim(0.12, 0.86)
fig.tight_layout(); fig.savefig(f"{HERE}/rys6_bezrobocie.png"); plt.close(fig)
print("figures done")
