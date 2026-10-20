# -*- coding: utf-8 -*-
"""Porządkowanie liniowe powiatów woj. małopolskiego (2011) – obliczenia."""
import json
import numpy as np, pandas as pd
from scipy import stats

import os
HERE = os.path.dirname(os.path.abspath(__file__))   # folder ze skryptami i raw.json
raw = json.load(open(f"{HERE}/raw.json"))
D = raw["data"]; names = raw["names"]; units = raw["units"]

podregion = {  # podział NTS 3 obowiązujący w 2011 r.
 "bocheński":"krakowski","krakowski":"krakowski","miechowski":"krakowski","myślenicki":"krakowski",
 "proszowicki":"krakowski","wielicki":"krakowski","m. Kraków":"m. Kraków",
 "gorlicki":"nowosądecki","limanowski":"nowosądecki","nowosądecki":"nowosądecki","nowotarski":"nowosądecki",
 "tatrzański":"nowosądecki","m. Nowy Sącz":"nowosądecki",
 "chrzanowski":"oświęcimski","olkuski":"oświęcimski","oświęcimski":"oświęcimski","suski":"oświęcimski","wadowicki":"oświęcimski",
 "brzeski":"tarnowski","dąbrowski":"tarnowski","tarnowski":"tarnowski","m. Tarnów":"tarnowski"}

df = pd.DataFrame({"powiat": names, "unit": units})
df["podregion"] = df["powiat"].map(podregion)
pop = np.array(D["pop"], float)
df["ludnosc"] = pop
df["ludnosc_wies"] = np.array(D["pop_wies_60633"], float)
df["stopa_bezrobocia"] = D["stopa_bezrobocia_60270"]
df["bezrobotni"] = D["bezrob_ogolem_453628"]
df["bezrobotni_dlugotrwale"] = D["bezrob_dlugotrw_453616"]

df["X1"] = np.array(D["lekarze_74067"]) / pop * 1e4                       # lekarze na 10 tys.
df["X2"] = np.array(D["miejsca_10412"]) / pop * 1e3                       # miejsca w stacj. pomocy społ. na 1000
df["X3"] = D["X5_kanal_79130"]                                            # % ludności z kanalizacją
df["X4"] = D["X7_zielen_m2_1724791"]                                      # zieleń m2 / mieszkańca
df["X5"] = np.array(D["bezrob_dlugotrw_453616"]) / np.array(D["bezrob_ogolem_453628"]) * 100
df["X6"] = D["X9_bezrob_na_oferte_217839"]                                # bezrobotni na 1 ofertę
df["X7"] = D["X10_regon_60530"]                                           # REGON na 10 tys.
df["X8"] = np.array(D["uczestnicy_imprez_58506"]) / pop * 1e3             # uczestnicy imprez na 1000

XALL = [f"X{i}" for i in range(1, 9)]
STIM = {"X1","X2","X3","X4","X7","X8"}
DEST = {"X5","X6"}

# ---------- EDA ----------
desc = pd.DataFrame({
    "średnia": df[XALL].mean(), "mediana": df[XALL].median(), "odch. std": df[XALL].std(ddof=1),
    "V (%)": df[XALL].std(ddof=1)/df[XALL].mean()*100, "min": df[XALL].min(), "max": df[XALL].max(),
    "skośność": df[XALL].skew(), "kurtoza": df[XALL].kurt()})
desc["min – powiat"] = [df.loc[df[x].idxmin(),"powiat"] for x in XALL]
desc["max – powiat"] = [df.loc[df[x].idxmax(),"powiat"] for x in XALL]
corr_p = df[XALL].corr(method="pearson"); corr_s = df[XALL].corr(method="spearman")

# kryterium zmienności: V < 10% -> zmienna quasi-stała
quasi = [x for x in XALL if desc.loc[x, "V (%)"] < 10]
X = list(XALL)

def hellwig_param(R, rstar):
    """metoda parametryczna Hellwiga – zmienne centralne / satelitarne / izolowane"""
    A = R.to_numpy().copy(); np.fill_diagonal(A, 0)
    R = pd.DataFrame(A, index=R.index, columns=R.columns)
    central, satellite, isolated = [], {}, []
    remaining = list(R.index)
    while remaining:
        sub = R.loc[remaining, remaining].abs()
        c = sub.sum(axis=1).idxmax()
        sats = [v for v in remaining if v != c and sub.loc[c, v] > rstar]
        if sats: central.append(c); satellite[c] = sats
        else: isolated.append(c)
        remaining = [v for v in remaining if v != c and v not in sats]
    return central, satellite, isolated

# ---------- metody porządkowania ----------
def unitaryzacja(frame, cols):
    Z = pd.DataFrame(index=frame.index)
    for x in cols:
        v = frame[x].astype(float); rng = v.max() - v.min()
        Z[x] = (v - v.min())/rng if x in STIM else (v.max() - v)/rng
    return Z

def order(frame, cols):
    Z = unitaryzacja(frame, cols)
    out = pd.DataFrame(index=frame.index)
    out["MS"] = Z.mean(axis=1)                                   # bezwzorcowa
    pattern = Z.max(axis=0)                                      # wzorzec = max (wektor jedynek)
    d = np.sqrt(((Z - pattern)**2).sum(axis=1))
    d0 = d.mean() + 2*d.std(ddof=1)
    out["d"] = d; out["MR"] = 1 - d/d0                           # wzorcowa (Hellwig)
    dm = np.sqrt(((Z - Z.min(axis=0))**2).sum(axis=1))
    out["TOPSIS"] = dm/(d + dm)                                  # uzupełniająco
    for m in ["MS","MR","TOPSIS"]:
        out[f"rank_{m}"] = out[m].rank(ascending=False, method="min").astype(int)
    return Z, out, d0

def three_means(s):
    m = s.mean(); mu = s[s >= m].mean(); ml = s[s < m].mean()
    g = np.where(s >= mu, 1, np.where(s >= m, 2, np.where(s >= ml, 3, 4)))
    return g, (m, mu, ml)

Z, res_main, d0 = order(df, X)
for c in res_main.columns: df[c] = res_main[c].values
Z.index = df["powiat"]
df["grupa_MS"], thr_MS = three_means(df["MS"])
df["grupa_MR"], thr_MR = three_means(df["MR"])
df["grupa_TOPSIS"], thr_TOP = three_means(df["TOPSIS"])

# ---------- zgodność rankingów ----------
tau_MS_MR = stats.kendalltau(df["MS"], df["MR"]); rho_MS_MR = stats.spearmanr(df["MS"], df["MR"])
tau_MS_T = stats.kendalltau(df["MS"], df["TOPSIS"]); tau_MR_T = stats.kendalltau(df["MR"], df["TOPSIS"])
rho_MS_bezr = stats.spearmanr(df["MS"], df["stopa_bezrobocia"]); rho_MR_bezr = stats.spearmanr(df["MR"], df["stopa_bezrobocia"])

# ---------- analiza wrażliwości ----------
sens = {}
dfl = df.copy(); dfl["X6"] = np.log(dfl["X6"])
_, rlog, _ = order(dfl, X); sens["log(X6)"] = rlog
for x in X:
    _, r_, _ = order(df, [v for v in X if v != x]); sens[f"bez {x}"] = r_
sens_tab = []
for k, r in sens.items():
    sens_tab.append(dict(wariant=k,
        tau_MS=stats.kendalltau(df["MS"], r["MS"])[0], rho_MS=stats.spearmanr(df["MS"], r["MS"])[0],
        tau_MR=stats.kendalltau(df["MR"], r["MR"])[0], rho_MR=stats.spearmanr(df["MR"], r["MR"])[0],
        top3_MS=", ".join(df.loc[r["rank_MS"].sort_values().index[:3], "powiat"]),
        bottom3_MS=", ".join(df.loc[r["rank_MS"].sort_values().index[-3:], "powiat"]),
        maxshift_MS=int((df["rank_MS"] - r["rank_MS"]).abs().max())))
sens_tab = pd.DataFrame(sens_tab)

# ---------- udziały ludności w grupach ----------
def shares(gcol, mcol):
    out = []
    for g in [1,2,3,4]:
        sub = df[df[gcol]==g]
        out.append(dict(grupa=g, n=len(sub), ludnosc=sub["ludnosc"].sum(),
            udzial_ludnosci=sub["ludnosc"].sum()/df["ludnosc"].sum()*100,
            udzial_wsi=sub["ludnosc_wies"].sum()/max(sub["ludnosc"].sum(),1)*100,
            bezrobotni=sub["bezrobotni"].sum()/df["bezrobotni"].sum()*100,
            dlugotrwale=sub["bezrobotni_dlugotrwale"].sum()/df["bezrobotni_dlugotrwale"].sum()*100,
            stopa=sub["bezrobotni"].sum() and np.average(sub["stopa_bezrobocia"], weights=sub["ludnosc"]),
            powiaty=", ".join(sub.sort_values(mcol, ascending=False)["powiat"])))
    return pd.DataFrame(out)
sh_MS = shares("grupa_MS","MS"); sh_MR = shares("grupa_MR","MR")

res = dict(desc=desc, corr_p=corr_p, corr_s=corr_s, Z=Z, df=df, X=X, XALL=XALL, quasi=quasi,
           thr_MS=thr_MS, thr_MR=thr_MR, thr_TOP=thr_TOP, d0=d0,
           tau_MS_MR=tau_MS_MR, rho_MS_MR=rho_MS_MR, tau_MS_T=tau_MS_T, tau_MR_T=tau_MR_T,
           rho_MS_bezr=rho_MS_bezr, rho_MR_bezr=rho_MR_bezr, sh_MS=sh_MS, sh_MR=sh_MR,
           hellwig=hellwig_param(corr_p.loc[X, X], 0.5), sens=sens_tab)

if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(desc.round(2)); print("quasi-stałe:", quasi, " finalne:", X)
    print("\nKorelacje Pearsona |r|>0.5 (finalny zbiór):")
    for i,a in enumerate(X):
        for b in X[i+1:]:
            if abs(corr_p.loc[a,b])>0.5: print(f"  {a}-{b}: {corr_p.loc[a,b]:.2f}")
    print("\nHellwig r*=0.5:", res["hellwig"])
    cols = ["powiat","podregion","MS","rank_MS","MR","rank_MR","TOPSIS","rank_TOPSIS","grupa_MS","grupa_MR","grupa_TOPSIS","d","stopa_bezrobocia"]
    print(df.sort_values("MS", ascending=False)[cols].round(3).to_string())
    print("\nMS: mean %.3f upper %.3f lower %.3f" % thr_MS); print("MR: mean %.3f upper %.3f lower %.3f" % thr_MR, " d0=%.3f" % d0)
    print("TOPSIS: mean %.3f upper %.3f lower %.3f" % thr_TOP)
    print("tau MS-MR %.4f p=%.2e" % (tau_MS_MR[0], tau_MS_MR[1]), "| rho MS-MR %.4f" % rho_MS_MR[0])
    print("tau MS-TOPSIS %.4f | tau MR-TOPSIS %.4f" % (tau_MS_T[0], tau_MR_T[0]))
    print("rho MS-bezrobocie %.3f p=%.4f | rho MR-bezrobocie %.3f p=%.4f" % (rho_MS_bezr[0], rho_MS_bezr[1], rho_MR_bezr[0], rho_MR_bezr[1]))
    print(sh_MS.round(1).to_string()); print(sh_MR.round(1).to_string())
    print(sens_tab.round(3).to_string())
    print("MS min/max", df["MS"].min().round(3), df["MS"].max().round(3), " MR min/max", df["MR"].min().round(3), df["MR"].max().round(3))
    print("udział wsi woj: %.1f" % (df["ludnosc_wies"].sum()/df["ludnosc"].sum()*100))
    print(Z.round(3).to_string())
