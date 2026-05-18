import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pysr import PySRRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import spearmanr
import time


CUTOFF = pd.Timestamp("2025-01-01")
MIN_TRAIN = 400
RETRAIN_EVERY = 10
PYSR_RETRAIN_EVERY = 30
PYSR_WINDOW = 720
ALPHA_SPLIT = 0.65
SHRINK_SCALE = 500
ALPHA_WIN = 90
IC_THR = 0.02
BLEND_SCALE = 2.0
TOP_K = 3
SEEDS = [42, 137]
IX_BASE = ["rsi_14", "bb_pos", "atr_norm", "mom_5_20"]


CB_CLF_P = {'iterations': 600, 'learning_rate': 0.04, 'depth': 5,
            'l2_leaf_reg': 5, 'subsample': 0.75, 'colsample_bylevel': 0.75,
            'auto_class_weights': "Balanced", 'eval_metric': "Accuracy",
            'early_stopping_rounds': 40, 'verbose': False,
            'random_seed': 42, 'allow_writing_files': False}

CB_REG_P = {'iterations': 800, 'learning_rate': 0.03, 'depth': 6,
            'l2_leaf_reg': 5, 'subsample': 0.75, 'colsample_bylevel': 0.75,
            'loss_function': "RMSE", 'eval_metric': "RMSE",
            'early_stopping_rounds': 50, 'verbose': False,
            'random_seed': 42, 'allow_writing_files': False}

XGB_REG_P = {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 6,
             'reg_lambda': 5, 'subsample': 0.75, 'colsample_bytree': 0.75,
             'objective': "reg:squarederror", 'eval_metric': "rmse",
             'early_stopping_rounds': 50, 'verbosity': 0,
             'random_seed': 42, 'n_jobs': -1}

XGB_CLF_P = {'n_estimators': 600, 'learning_rate': 0.04, 'max_depth': 5,
             'reg_lambda': 5, 'subsample': 0.75, 'colsample_bytree': 0.75,
             'objective': "binary:logistic", 'eval_metric': "logloss",
             'early_stopping_rounds': 40, 'verbosity': 0,
             'random_seed': 42, 'n_jobs': -1}
PYSR_P = {'niterations': 120, 'populations': 30, 'population_size': 50,
          'maxsize': 20, 'binary_operators': ["+", "-", "*", "/"],
          'unary_operators': ["abs", "neg", "square", "tanh", "sin", "exp", "log"],
          'elementwise_loss': "L2DistLoss()", 'parsimony': 0.003, 'verbosity': 0,
          'temp_equation_file': True, 'delete_tempfiles': True,
          'select_k_features': 12, 'turbo': True, 'bumper': True}
PYSR_RESID_P = {'niterations': 80, 'populations': 20, 'population_size': 40,
                'maxsize': 14, 'binary_operators': ["+", "-", "*", "/"],
                'unary_operators': ["abs", "neg", "square", "tanh"],
                'elementwise_loss': "L2DistLoss()", 'parsimony': 0.005, 'verbosity': 0,
                'temp_equation_file': True, 'delete_tempfiles': True,
                'select_k_features': 10, 'turbo': True, 'bumper': True}


CB_FEATS = ["close", "ret_1d","ret_2d","ret_3d","ret_5d","ret_10d","ret_15d","ret_20d","ret_30d","ret_60d",
            "body","upper_shadow","lower_shadow","hl_range","gap","close_vs_sma5","close_vs_sma10","close_vs_sma20",
            "close_vs_sma50","close_vs_sma100","close_vs_sma200","sma5_vs_20","sma20_vs_50","sma50_vs_200",
            "vol_5d","vol_10d","vol_20d","vol_60d","vol_ratio_5_20","vol_ratio_20_60","atr_norm","vol_change",
            "vol_vs_ma5","vol_vs_ma20","vol_vs_ma60","vol_price_pressure","rsi_7","rsi_14","rsi_21","rsi_diff_7_14",
            "macd_norm","macd_hist","macd_cross","bb_pos","bb_width","bb_squeeze","stoch_k","stoch_d","stoch_diff",
            "williams_r","dist_52w_high","dist_52w_low","mom_5_20","mom_10_60","streak"]

PYSR_FEATS = ["ret_5d","ret_10d","ret_20d", "rsi_14","stoch_k","williams_r","bb_pos", 
              "vol_ratio_5_20","bb_squeeze","atr_norm","sma5_vs_20","sma20_vs_50",
              "dist_52w_high","dist_52w_low","vol_price_pressure","rsi_x_bb","mom_regime",
              "vol_adj_mom","trend_vs_rev","squeeze_mom","rsi_stoch_div","vol_mom_pressure","dist_ratio"]


def make_features(df):
    d = df.copy()
    for n in [1,2,3,5,10,15,20,30,60]:
        d[f"ret_{n}d"] = d["close"].pct_change(n)
    d["body"] = (d["close"]-d["open"])/(d["open"]+1e-9)
    d["upper_shadow"] = (d["high"]-d[["open","close"]].max(axis=1))/(d["open"]+1e-9)
    d["lower_shadow"] = (d[["open","close"]].min(axis=1)-d["low"])/(d["open"]+1e-9)
    d["hl_range"] = (d["high"]-d["low"])/(d["open"]+1e-9)
    d["gap"] = (d["open"]-d["close"].shift(1))/(d["close"].shift(1)+1e-9)
    for w in [5,10,20,50,100,200]:
        sma = d["close"].rolling(w).mean()
        d[f"close_vs_sma{w}"] = d["close"]/(sma+1e-9)-1
    d["sma5_vs_20"] = d["close"].rolling(5).mean()/(d["close"].rolling(20).mean()+1e-9)-1
    d["sma20_vs_50"] = d["close"].rolling(20).mean()/(d["close"].rolling(50).mean()+1e-9)-1
    d["sma50_vs_200"] = d["close"].rolling(50).mean()/(d["close"].rolling(200).mean()+1e-9)-1
    for w in [5,10,20,60]:
        d[f"vol_{w}d"] = d["ret_1d"].rolling(w).std()
    d["vol_ratio_5_20"] = d["vol_5d"]/(d["vol_20d"]+1e-9)
    d["vol_ratio_20_60"] = d["vol_20d"]/(d["vol_60d"]+1e-9)
    tr = pd.concat([d["high"]-d["low"], (d["high"]-d["close"].shift(1)).abs(),
                    (d["low"]-d["close"].shift(1)).abs()], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    d["atr_norm"] = d["atr_14"]/(d["close"]+1e-9)
    d["vol_change"] = d["volume"].pct_change(1)
    d["vol_vs_ma5"] = d["volume"]/(d["volume"].rolling(5).mean()+1e-9)-1
    d["vol_vs_ma20"] = d["volume"]/(d["volume"].rolling(20).mean()+1e-9)-1
    d["vol_vs_ma60"] = d["volume"]/(d["volume"].rolling(60).mean()+1e-9)-1
    d["vol_price_pressure"] = d["ret_1d"]*d["vol_vs_ma20"]
    for p in [7,14,21]:
        delta = d["close"].diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        d[f"rsi_{p}"] = 100-100/(1+gain/(loss+1e-9))
    d["rsi_diff_7_14"] = d["rsi_7"]-d["rsi_14"]
    ema12 = d["close"].ewm(span=12,adjust=False).mean()
    ema26 = d["close"].ewm(span=26,adjust=False).mean()
    d["macd"] = ema12-ema26
    d["macd_signal"] = d["macd"].ewm(span=9,adjust=False).mean()
    d["macd_hist"] = d["macd"]-d["macd_signal"]
    d["macd_norm"] = d["macd"]/(d["close"]+1e-9)
    d["macd_cross"] = np.sign(d["macd_hist"])-np.sign(d["macd_hist"].shift(1))
    mid = d["close"].rolling(20).mean()
    std20 = d["close"].rolling(20).std()
    d["bb_upper"] = mid+2*std20
    d["bb_lower"] = mid-2*std20
    d["bb_pos"] = (d["close"]-d["bb_lower"])/(d["bb_upper"]-d["bb_lower"]+1e-9)
    d["bb_width"] = (d["bb_upper"]-d["bb_lower"])/(mid+1e-9)
    d["bb_squeeze"] = d["bb_width"]/(d["bb_width"].rolling(20).mean()+1e-9)
    lo14 = d["low"].rolling(14).min()
    hi14 = d["high"].rolling(14).max()
    d["stoch_k"] = (d["close"]-lo14)/(hi14-lo14+1e-9)*100
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    d["stoch_diff"] = d["stoch_k"]-d["stoch_d"]
    d["williams_r"] = (hi14-d["close"])/(hi14-lo14+1e-9)*-100
    d["dist_52w_high"] = d["close"]/(d["high"].rolling(252).max()+1e-9)-1
    d["dist_52w_low"]  = d["close"]/(d["low"].rolling(252).min()+1e-9)-1
    d["mom_5_20"] = d["ret_5d"]/(d["vol_20d"]+1e-9)
    d["mom_10_60"] = d["ret_10d"]/(d["vol_60d"]+1e-9)
    d["up_day"] = (d["close"]>d["close"].shift(1)).astype(int)
    d["streak"] = (d["up_day"].groupby((d["up_day"]!=d["up_day"].shift()).cumsum()).cumcount()+1)*(2*d["up_day"]-1)
    d["rsi_x_bb"] = d["rsi_14"]*d["bb_pos"]
    d["mom_regime"] = d["mom_5_20"]/(d["atr_norm"]**2+1e-9)
    d["vol_adj_mom"] = d["ret_5d"]/(d["vol_5d"]**2+1e-9)
    d["trend_vs_rev"] = d["sma5_vs_20"]*np.sign(d["rsi_14"]-50)
    d["squeeze_mom"] = d["bb_squeeze"]*d["mom_5_20"]
    d["rsi_stoch_div"] = d["rsi_14"]-d["stoch_k"]
    d["vol_mom_pressure"] = d["vol_price_pressure"]*d["mom_10_60"]
    d["dist_ratio"] = d["dist_52w_high"]/((d["dist_52w_low"]-d["dist_52w_high"]).abs()+1e-9)
    d["target"] = (d["close"].shift(-1)>d["close"]).astype(int)
    d["close_next"] = d["close"].shift(-1)
    d["log_ret_next"] = np.log(d["close"].shift(-1)/(d["close"]+1e-9))
    return d


def da(yt, yp):
    return np.mean(np.array(yt)==np.array(yp))

def smape(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    return float(np.mean(2*np.abs(yt-yp)/(np.abs(yt)+np.abs(yp)+1e-9))*100)

def mape(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    m = np.abs(yt)>1e-9
    if m.sum()==0: return float("nan")
    return float(np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100)

def log_cosh(yt, yp):
    return float(np.mean(np.log(np.cosh(np.asarray(yp,float)-np.asarray(yt,float)+1e-12))))

def ic(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    m = np.isfinite(yt)&np.isfinite(yp)
    if m.sum()<10: return 0.0
    c,_ = spearmanr(yt[m],yp[m])
    return float(c) if np.isfinite(c) else 0.0

def dir_acc(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    m = np.abs(yt)>1e-12
    if m.sum()==0: return 0.5
    return float(np.mean(np.sign(yt[m])==np.sign(yp[m])))

def qhr(yt, yp, q=0.2):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    k = max(1,int(len(yt)*q))
    tr=set(np.argsort(yt)[-k:]); tp=set(np.argsort(yp)[-k:])
    br=set(np.argsort(yt)[:k]);  bp=set(np.argsort(yp)[:k])
    return float((len(tr&tp)+len(br&bp))/(2*k))

def r2(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    m = np.isfinite(yt)&np.isfinite(yp)
    if m.sum()<2: return float("nan")
    yt, yp = yt[m], yp[m]
    return 1.0 - float(np.sum((yt-yp)**2))/(float(np.sum((yt-yt.mean())**2))+1e-12)

def reg_metrics(actual_next, pred, curr):
    yt = np.asarray(actual_next, float)
    yp = np.asarray(pred, float)
    cc = np.asarray(curr, float)
    alr = np.log(yt/(cc+1e-9))
    plr = np.log(np.maximum(yp,1e-9)/(cc+1e-9))
    return {"r2": r2(yt, yp),
            "rmse": float(np.sqrt(np.mean((yt-yp)**2))),
            "mae": float(np.mean(np.abs(yt-yp))),
            "mape": mape(yt, yp),
            "smape": smape(yt, yp),
            "log_cosh": log_cosh(yt, yp),
            "ic": ic(alr, plr),
            "da": dir_acc(alr, plr),
            "qhr_20": qhr(alr, plr, 0.2)}


def fit_cb_clf(Xt,yt,Xv,yv):
    m = CatBoostClassifier(**CB_CLF_P)
    m.fit(Pool(Xt,label=yt), eval_set=Pool(Xv,label=yv), use_best_model=True, verbose=False)
    return m

def fit_cb_reg(Xt,yt,Xv,yv):
    m = CatBoostRegressor(**CB_REG_P)
    m.fit(Pool(Xt,label=yt), eval_set=Pool(Xv,label=yv), use_best_model=True, verbose=False)
    return m

def fit_xgb_reg(Xt,yt,Xv,yv):
    m = XGBRegressor(**XGB_REG_P)
    m.fit(Xt.astype(float), yt, eval_set=[(Xv.astype(float),yv)], verbose=False)
    return m

def fit_xgb_clf(Xt,yt,Xv,yv):
    m = XGBClassifier(**XGB_CLF_P)
    m.fit(Xt.astype(float), yt, eval_set=[(Xv.astype(float),yv)], verbose=False)
    return m


def get_sel(model):
    for attr in ["selection_","best_idx_","selected_idx_"]:
        if hasattr(model,attr): 
            return getattr(model,attr)
    try:
        eqs = model.equations_
        if isinstance(eqs,list): 
            eqs = eqs[0]
        return len(eqs)-1
    except: 
        return 0

def set_sel(model, idx):
    for attr in ["selection_","best_idx_","selected_idx_"]:
        if hasattr(model, attr): setattr(model, attr, idx); 
        return
    try: 
        model.selection_ = idx
    except: 
        pass

def safe_pred(model, X, feats):
    try:
        p = np.asarray(model.predict(X[feats]),dtype=float)
        p = np.nan_to_num(p,nan=0.0,posinf=0.0,neginf=0.0)
        nz = p[p!=0]
        cv = np.nanpercentile(np.abs(nz),99)*2 if len(nz)>0 else 1.0
        return np.clip(p, -cv, cv)
    except: 
        return np.zeros(len(X))

def safe_pred_eq(model, X, feats, eq_idx):
    try:
        old = get_sel(model); set_sel(model,eq_idx)
        p = np.asarray(model.predict(X[feats]),dtype=float)
        p = np.nan_to_num(p,nan=0.0,posinf=0.0,neginf=0.0)
        set_sel(model,old)
        nz = p[p!=0]
        cv = np.nanpercentile(np.abs(nz),99)*2 if len(nz)>0 else 1.0
        return np.clip(p, -cv, cv)
    except:
        try: 
            set_sel(model, old)
        except: 
            pass
        return np.zeros(len(X))

def best_eq(model, Xv, yv):
    try:
        eqs = model.equations_
        if isinstance(eqs,list): 
            eqs = eqs[0]
        bs, bi = -np.inf, get_sel(model)
        for i,row in eqs.iterrows():
            try:
                old = get_sel(model)
                set_sel(model,i)
                p = np.nan_to_num(model.predict(Xv),nan=0.0,posinf=0.0,neginf=0.0)
                if np.std(p)<1e-12: 
                    set_sel(model,old)
                    continue
                s=abs(ic(yv,p))-row["complexity"]*0.003
                if s > bs: 
                    bs, bi = s, i
                set_sel(model, old)
            except: 
                continue
        set_sel(model,bi)
    except: 
        pass
    return model

def top_k_idxs(model, Xv, yv, k=TOP_K):
    try:
        eqs=model.equations_
        if isinstance(eqs,list): 
            eqs=eqs[0]
        scores=[]
        for i,row in eqs.iterrows():
            try:
                old = get_sel(model)
                set_sel(model, i)
                p = np.nan_to_num(model.predict(Xv), nan=0.0, posinf=0.0, neginf=0.0)
                set_sel(model,old)
                if np.std(p)<1e-12:
                    continue
                scores.append((i,abs(ic(yv,p)),p))
            except:
                try: 
                    set_sel(model,old)
                except: 
                    pass
        if not scores: 
            return [get_sel(model)]
        scores.sort(key=lambda x:x[1], reverse=True)
        sel = [scores[0]]
        for sc in scores[1:]:
            if len(sel) >= k: 
                break
            ok = True
            for s2 in sel:
                if abs(np.corrcoef(sc[2],s2[2])[0,1]) > 0.85: 
                    ok = False
                    break
            if ok: 
                sel.append(sc)
        return [s[0] for s in sel]
    except: 
        return [get_sel(model)]

def run_pysr(Xt, yt, Xv, yv, params=None, seed=42):
    if params is None: 
        params=PYSR_P
    p = params.copy()
    p['random_state'] = seed
    m = PySRRegressor(**p)
    cv = float(np.nanpercentile(np.abs(yt),98))
    m.fit(Xt, np.clip(yt,-cv,cv))
    return best_eq(m, Xv, yv)

def pysr_targets(df):
    lr = df["log_ret_next"].values
    return {"ret": lr, "vol": np.abs(lr), "mom": np.sign(df["ret_5d"].values)*np.abs(lr)}

def train_pysr_all(Xtr, Xvl, dftr, dfvl, pfx=""):
    ttr = pysr_targets(dftr)
    tvl = pysr_targets(dfvl)
    out = {}
    for tname, ytr in ttr.items():
        yvl = tvl[tname]
        for seed in SEEDS:
            key = f"{tname}_s{seed}"
            prm = PYSR_P if tname=="ret" else PYSR_RESID_P
            out[key] = run_pysr(Xtr, ytr, Xvl, yvl, prm, seed)
    return out


def extract_pf(models, X, Xvl, dfvl, pfx="pf"):
    tvl = pysr_targets(dfvl)
    raw = {}
    ics = {}
    pvl = {}

    for key, model in models.items():
        tname = key.split("_s")[0]
        yvl = tvl.get(tname, tvl["ret"])
        idxs = top_k_idxs(model, Xvl, yvl, k=TOP_K)

        for ki, eq_idx in enumerate(idxs):
            fn = f"{pfx}_{key}_e{ki}"
            pvl_ = safe_pred_eq(model, Xvl, PYSR_FEATS, eq_idx)
            pall = safe_pred_eq(model, X, PYSR_FEATS, eq_idx)
            dup = any(len(pvl_)==len(ep) and abs(np.corrcoef(pvl_,ep)[0,1])>0.90 for ep in pvl.values())
            if dup or np.std(pvl_) < 1e-12: 
                continue
            raw[fn] = pall
            ics[fn] = ic(yvl, pvl_)
            pvl[fn] = pvl_

    kept = {k:v for k,v in ics.items() if abs(v) >= IC_THR}
    if not kept:
        return pd.DataFrame({f"{pfx}_fallback": np.zeros(len(X))}, index=X.index if hasattr(X,'index') else range(len(X)))

    out={}
    for fn, p in raw.items():
        if fn not in kept: continue
        bw = float(np.clip(abs(ics[fn])/IC_THR*BLEND_SCALE, 0.0, 1.0))
        out[fn] = p * bw

    bfn = max(kept, key=lambda k: abs(kept[k]))
    bp = raw[bfn]
    bic = ics[bfn]

    for bf in IX_BASE:
        if bf in X.columns:
            ix = bp * X[bf].values.astype(float)
            if np.std(ix)>1e-12:
                out[f"{pfx}_ix_{bf}"] = ix * float(np.clip(abs(bic)/IC_THR,0.0,1.0))
    return pd.DataFrame(out, index=X.index if hasattr(X,'index') else range(len(X)))


def opt_w(prob_clf, sig, ytrue, n=None):
    vote = 0.5+0.5*np.sign(sig)
    n = n or len(ytrue)
    shrk = min(1.0, n/SHRINK_SCALE)
    bd, bw = 0.0, 0.5
    for wc in np.arange(0.3, 0.85, 0.05):
        prob = np.clip(wc*prob_clf + (1-wc)*vote, 0, 1)
        d = float(np.mean((prob>=0.5).astype(int)==ytrue))
        if d > bd: 
            bd, bw = d, round(wc,2)
    return round(shrk*bw + (1-shrk)*0.5, 3)

def hyb_pred(prob_clf, sig, wc):
    return float(np.clip(wc*prob_clf+(1-wc)*(0.5+0.5*np.sign(sig)),0,1))

# ================================================================================================================
df_raw = pd.read_csv("data/ETHUSDT_daily_interpolated_from_2021.csv")
df_raw["date"] = pd.to_datetime(df_raw["date"])
df_raw = df_raw.sort_values("date").reset_index(drop=True)
df = make_features(df_raw)

need = list(set(CB_FEATS+PYSR_FEATS+["target","close_next","log_ret_next"]))
dfc = df.dropna(subset=need).copy().reset_index(drop=True)
trn = dfc[dfc["date"] < CUTOFF].reset_index(drop=True)
tst = dfc[dfc["date"] >= CUTOFF].reset_index(drop=True)

sn = int(len(trn)*ALPHA_SPLIT)
psr_tr = trn.iloc[:sn].reset_index(drop=True)
sv = trn.iloc[sn:].reset_index(drop=True)

pw = (psr_tr.iloc[-PYSR_WINDOW:].reset_index(drop=True)
        if len(psr_tr)>PYSR_WINDOW else psr_tr.reset_index(drop=True))
ws = int(len(pw)*0.8)
pw_tr, pw_vl = pw.iloc[:ws], pw.iloc[ws:]

pmdl = train_pysr_all(pw_tr[PYSR_FEATS], pw_vl[PYSR_FEATS], pw_tr, pw_vl)

pf_tr = extract_pf(pmdl, psr_tr, pw_vl[PYSR_FEATS], pw_vl)
pf_sv = extract_pf(pmdl, sv, pw_vl[PYSR_FEATS], pw_vl)

pf_names = list(pf_tr.columns)
ENH= CB_FEATS + pf_names

enh_tr = pd.concat([psr_tr.reset_index(drop=True), pf_tr.reset_index(drop=True)], axis=1)
enh_sv = pd.concat([sv.reset_index(drop=True), pf_sv.reset_index(drop=True)], axis=1)

es = int(len(enh_tr)*0.85)
tp, vp = enh_tr.iloc[:es], enh_tr.iloc[es:]

cb_c0 = fit_cb_clf(tp[CB_FEATS], tp["target"], vp[CB_FEATS], vp["target"])
cb_r0 = fit_cb_reg(tp[CB_FEATS], tp["close_next"], vp[CB_FEATS], vp["close_next"])
cb_ce0 = fit_cb_clf(tp[ENH], tp["target"], vp[ENH], vp["target"])
cb_re0 = fit_cb_reg(tp[ENH], tp["close_next"], vp[ENH], vp["close_next"])
xb_c0 = fit_xgb_clf(tp[CB_FEATS].astype(float), tp["target"], vp[CB_FEATS].astype(float), vp["target"])
xb_r0 = fit_xgb_reg(tp[CB_FEATS].astype(float), tp["close_next"], vp[CB_FEATS].astype(float), vp["close_next"])
xb_ce0 = fit_xgb_clf(tp[ENH].astype(float), tp["target"], vp[ENH].astype(float), vp["target"])
xb_re0 = fit_xgb_reg(tp[ENH].astype(float), tp["close_next"], vp[ENH].astype(float), vp["close_next"])

svf = enh_sv[CB_FEATS]
svef = enh_sv[ENH]
sv_cc = sv["close"].values.astype(float)
sv_cn = sv["close_next"].values.astype(float)

cb_resid = sv_cn - cb_r0.predict(svf)
xb_resid = sv_cn - xb_r0.predict(svf.astype(float))

rs = int(len(sv)*0.7)
rtr = sv.iloc[:rs].reset_index(drop=True)
rvl = sv.iloc[rs:].reset_index(drop=True)
psr_rcb = run_pysr(rtr[PYSR_FEATS], cb_resid[:rs], rvl[PYSR_FEATS], cb_resid[rs:], PYSR_RESID_P, 42)
psr_rxb = run_pysr(rtr[PYSR_FEATS], xb_resid[:rs], rvl[PYSR_FEATS], xb_resid[rs:], PYSR_RESID_P, 42)

w_cbe = opt_w(cb_ce0.predict_proba(svef)[:,1], cb_re0.predict(svef).astype(float)-sv_cc, sv["target"].values, len(sv))
w_cb0 = opt_w(cb_c0.predict_proba(svf)[:,1], cb_r0.predict(svf).astype(float)-sv_cc, sv["target"].values, len(sv))
w_xbe = opt_w(xb_ce0.predict_proba(svef.astype(float))[:,1], xb_re0.predict(svef.astype(float)).astype(float)-sv_cc, sv["target"].values, len(sv))
w_xb0 = opt_w(xb_c0.predict_proba(svf.astype(float))[:,1], xb_r0.predict(svf.astype(float)).astype(float)-sv_cc, sv["target"].values, len(sv))

rows=[]
t0=time.time()
cur={}

for step, (_, row) in enumerate(tst.iterrows()):
    cd = row["date"]
    trdf = dfc[dfc["date"] < cd]

    if not cur or step%RETRAIN_EVERY==0:
        if len(trdf) >= MIN_TRAIN:
            pft = extract_pf(pmdl, trdf, pw_vl[PYSR_FEATS], pw_vl)
            cenh = CB_FEATS + list(pft.columns)
            te = pd.concat([trdf.reset_index(drop=True), pft.reset_index(drop=True)], axis=1)
            si = int(len(te)*0.85)
            t2, v2 = te.iloc[:si], te.iloc[si:]

            cur["cb_c"] = fit_cb_clf(t2[CB_FEATS].astype(float) if False else t2[CB_FEATS], t2["target"], v2[CB_FEATS], v2["target"])
            cur["cb_r"] = fit_cb_reg(t2[CB_FEATS], t2["close_next"], v2[CB_FEATS], v2["close_next"])
            cur["cb_ce"] = fit_cb_clf(t2[cenh], t2["target"], v2[cenh], v2["target"])
            cur["cb_re"] = fit_cb_reg(t2[cenh], t2["close_next"], v2[cenh], v2["close_next"])
            cur["xb_c"] = fit_xgb_clf(t2[CB_FEATS].astype(float), t2["target"], v2[CB_FEATS].astype(float), v2["target"])
            cur["xb_r"] = fit_xgb_reg(t2[CB_FEATS].astype(float), t2["close_next"], v2[CB_FEATS].astype(float), v2["close_next"])
            cur["xb_ce"] = fit_xgb_clf(t2[cenh].astype(float), t2["target"], v2[cenh].astype(float), v2["target"])
            cur["xb_re"] = fit_xgb_reg(t2[cenh].astype(float), t2["close_next"], v2[cenh].astype(float), v2["close_next"])
            cur["enh"] = cenh

            rw = te.iloc[-ALPHA_WIN:].reset_index(drop=True)
            if len(rw) >= 30:
                rwf = rw[CB_FEATS]; rwef = rw[cenh]
                rwc = rw["close"].values.astype(float)
                cur["w_cbe"] = opt_w(cur["cb_ce"].predict_proba(rwef)[:,1], cur["cb_re"].predict(rwef).astype(float)-rwc, rw["target"].values, len(rw))
                cur["w_xbe"] = opt_w(cur["xb_ce"].predict_proba(rwef.astype(float))[:,1], cur["xb_re"].predict(rwef.astype(float)).astype(float)-rwc, rw["target"].values, len(rw))
                cur["w_cb0"] = opt_w(cur["cb_c"].predict_proba(rwf)[:,1], cur["cb_r"].predict(rwf).astype(float)-rwc, rw["target"].values, len(rw))
                cur["w_xb0"] = opt_w(cur["xb_c"].predict_proba(rwf.astype(float))[:,1], cur["xb_r"].predict(rwf.astype(float)).astype(float)-rwc, rw["target"].values, len(rw))
            else:
                cur["w_cbe"] = w_cbe
                cur["w_xbe"] = w_xbe
                cur["w_cb0"] = w_cb0
                cur["w_xb0"] = w_xb0

    if step > 0 and step%PYSR_RETRAIN_EVERY == 0 and len(trdf) >= MIN_TRAIN:
        w2  = (trdf.iloc[-PYSR_WINDOW:].reset_index(drop=True) if len(trdf)>PYSR_WINDOW else trdf.reset_index(drop=True))
        ws2 = int(len(w2)*0.8)
        wt, wv = w2.iloc[:ws2], w2.iloc[ws2:]
        pmdl = train_pysr_all(wt[PYSR_FEATS], wv[PYSR_FEATS], wt, wv)
        pw_vl = wv

        rec = trdf.iloc[-len(sv):].reset_index(drop=True)
        rcn = rec["close_next"].values.astype(float)
        cbp = (cur["cb_r"] if cur else cb_r0).predict(rec[CB_FEATS])
        xbp = (cur["xb_r"] if cur else xb_r0).predict(rec[CB_FEATS].astype(float))
        rrs = int(len(rec)*0.7)
        rt = rec.iloc[:rrs].reset_index(drop=True)
        rv = rec.iloc[rrs:].reset_index(drop=True)
        psr_rcb = run_pysr(rt[PYSR_FEATS], (rcn-cbp)[:rrs], rv[PYSR_FEATS], (rcn-cbp)[rrs:], PYSR_RESID_P, 42)
        psr_rxb = run_pysr(rt[PYSR_FEATS], (rcn-xbp)[:rrs], rv[PYSR_FEATS], (rcn-xbp)[rrs:], PYSR_RESID_P, 42)

    if cur:
        xr = row.to_frame().T
        xc = xr[CB_FEATS]
        cenh = cur.get("enh", ENH)
        cc = float(row["close"])

        pfs = extract_pf(pmdl, xr, pw_vl[PYSR_FEATS], pw_vl)
        xe = pd.concat([xc.reset_index(drop=True), pfs.reset_index(drop=True)], axis=1)
        for col in cenh:
            if col not in xe.columns: 
                xe[col]=0.0

        pc = float(cur["cb_c"].predict_proba(xc)[0][1])
        lc = float(cur["cb_r"].predict(xc)[0])
        px = float(cur["xb_c"].predict_proba(xc.astype(float))[0][1])
        lx = float(cur["xb_r"].predict(xc.astype(float))[0])

        pce = float(cur["cb_ce"].predict_proba(xe[cenh])[0][1])
        lce = float(cur["cb_re"].predict(xe[cenh])[0])
        pxe = float(cur["xb_ce"].predict_proba(xe[cenh].astype(float))[0][1])
        lxe = float(cur["xb_re"].predict(xe[cenh].astype(float))[0])

        rc = float(safe_pred(psr_rcb, xr, PYSR_FEATS)[0])
        rx = float(safe_pred(psr_rxb, xr, PYSR_FEATS)[0])
        lc_r = lc + rc
        lx_r = lx + rx

        hcb = hyb_pred(pc, lc-cc, cur.get("w_cb0", w_cb0))
        hcbe = hyb_pred(pce, lce-cc, cur.get("w_cbe", w_cbe))
        hxb = hyb_pred(px, lx-cc, cur.get("w_xb0", w_xb0))
        hxbe = hyb_pred(pxe, lxe-cc, cur.get("w_xbe", w_xbe))
    else:
        pc = pce = px = pxe = hcb = hcbe = hxb = hxbe = 0.5
        cc = float(row["close"])
        lc = lce = lx = lxe = lc_r = lx_r = cc
        rc = rx = 0.0

    a = int(row["target"])
    def d(p, c, a): 
        return int(int(p > c)==a)

    rows.append({"date": row["date"], "close": row["close"], "actual": a,
                 "close_next_actual": float(row["close_next"]),
                 "pc": pc, "lc": lc, "c_cb": int(int(pc >= 0.5) == a), 
                 "c_cb_r":  d(lc, cc, a), "px": px, "lx": lx,
                 "c_xb": int(int(px >= 0.5) == a), "c_xb_r": d(lx, cc, a),
                 "pce": pce, "lce": lce, "c_cbe": int(int(pce >= 0.5) == a),
                 "c_cbe_r": d(lce, cc, a), "pxe": pxe, "lxe": lxe, 
                 "c_xbe": int(int(pxe >= 0.5) == a), "c_xbe_r": d(lxe, cc, a),
                 "hcb": hcb, "c_hcb": int(int(hcb >= 0.5) == a),
                 "hcbe": hcbe, "c_hcbe": int(int(hcbe >= 0.5) == a),
                 "hxb": hxb, "c_hxb": int(int(hxb >= 0.5) == a),
                 "hxbe": hxbe, "c_hxbe": int(int(hxbe >= 0.5) == a),
                 "lc_r": lc_r, "lx_r": lx_r, "c_cbr": d(lc_r, cc, a), "c_xbr": d(lx_r, cc, a),
                 "regime": "bull" if float(row["sma20_vs_50"]) > 0 else "bear"})

    if (step+1)%50 == 0 or step == len(tst) - 1:
        r = rows
        el = time.time() - t0
        print(f"[{step+1:>3}/{len(tst)}]"
              f"CB={np.mean([x['c_cb']  for x in r]):.3f}"
              f"CB+={np.mean([x['c_cbe'] for x in r]):.3f}"
              f"XB={np.mean([x['c_xb']  for x in r]):.3f}"
              f"XB+={np.mean([x['c_xbe'] for x in r]):.3f}"
              f"HCBe={np.mean([x['c_hcbe'] for x in r]):.3f}"
              f"HXBe={np.mean([x['c_hxbe'] for x in r]):.3f}"
              f"[{el:.0f}s]")

res = pd.DataFrame(rows)
rr = res.dropna(subset=["close_next_actual"]).copy()
acn = rr["close_next_actual"].values.astype(float)
cca = rr["close"].values.astype(float)

mcols = {"CB": "lc", "CB+PySR": "lce", "XB": "lx",
         "XB+PySR": "lxe", "CB+res": "lc_r", "XB+res": "lx_r"}
arm = {n: reg_metrics(acn, rr[c].values.astype(float), cca) for n,c in mcols.items()}

dv = {"CB": np.mean(res["c_cb"]), "CB+PySR": np.mean(res["c_cbe"]), "XB": np.mean(res["c_xb"]), 
      "XB+PySR": np.mean(res["c_xbe"]), "Hyb CB": np.mean(res["c_hcb"]), "Hyb CB+": np.mean(res["c_hcbe"]), 
      "Hyb XB": np.mean(res["c_hxb"]), "Hyb XB+": np.mean(res["c_hxbe"]), "CB+res":  np.mean(res["c_cbr"]),
      "XB+res":  np.mean(res["c_xbr"])}

d0cb = dv["CB"]; d0xb = dv["XB"]
print(f"\n  {'Model':<18} {'DA':>8} {'vs CB':>8} {'vs XB':>8}")
print("  "+"-"*46)
for n, d in dv.items():
    dc = f"{(d-d0cb)*100:+.2f}" if n!="CB"  else "---"
    dx = f"{(d-d0xb)*100:+.2f}" if n!="XB"  else "---"
    mk = " ★" if d==max(dv.values()) else ""
    print(f"  {n:<18} {d*100:>7.2f}% {dc:>7}pp {dx:>7}pp{mk}")

mks = ["r2","rmse","mae","mape","smape","ic","da","qhr_20"]
hdr = f"  {'Metric':<10}"
for n in mcols: hdr += f"  {n:>10}"
print(f"\n{hdr}"); print("  "+"-"*(10+12*len(mcols)))
for mk in mks:
    ln = f"  {mk.upper():<10}"
    u  = "%" if mk in ("mape","smape") else ""
    for n in mcols:
        v = arm[n].get(mk, float('nan'))
        ln += f"  {v:>9.4f}{u} "
    print(ln)

best = max(dv, key=dv.get)
print(f"\n  Best DA: {best} = {dv[best]*100:.2f}%")