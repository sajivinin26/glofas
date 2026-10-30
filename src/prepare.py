"""Build the modelling table from GloFAS NetCDF files (flood-onset early warning)."""
import glob, numpy as np, pandas as pd, xarray as xr
import pathlib; ROOT=str(pathlib.Path(__file__).resolve().parent.parent)
H=3            # forecast horizon (days)
PCT=0.90       # high-flow level = 90th percentile of a point's TRAINING-period flow
TRAIN_END, VALID_END = "2025-06-30", "2025-12-31"

parts=[xr.open_dataset(f).avg_dis.load() for f in sorted(glob.glob(f"{ROOT}/raw/*.nc"))]
q=xr.concat(parts,dim="valid_time").sortby("valid_time")
q=q.isel(valid_time=~pd.Index(q.valid_time.values).duplicated())
q=q.rename({"valid_time":"time","latitude":"lat","longitude":"lon"})
t=pd.to_datetime(q.time.values); full=pd.date_range(t.min(),t.max()); print("days",len(t),"expected",len(full),"missing days (interpolated):",len(full)-len(t))
mean_q=q.mean("time"); cand=mean_q.where(mean_q>=25).to_series().dropna().sort_values(ascending=False)
chosen=[]
for (la,lo),v in cand.items():                       # greedy: strongest rivers first, >=0.15 deg apart
    if all(max(abs(la-a),abs(lo-b))>=0.15 for a,b,_ in chosen): chosen.append((la,lo,v))
print("river points selected:",len(chosen))
pts=pd.DataFrame(chosen,columns=["lat","lon","mean_q"]); pts["point_id"]=[f"P{i+1:02d}" for i in range(len(pts))]
pts.to_csv(f"{ROOT}/outputs/points.csv",index=False)

rows=[]
for _,p in pts.iterrows():
    s=pd.Series(q.sel(lat=p.lat,lon=p.lon).values,index=t).astype(float).sort_index().reindex(full).interpolate(limit=1)  # fill the single missing day
    thr=s[:TRAIN_END].quantile(PCT)
    d=pd.DataFrame({"q":s})
    for k in (1,2,3,7): d[f"q_lag{k}"]=s.shift(k)
    d["q_mean7"]=s.rolling(7).mean(); d["q_mean30"]=s.rolling(30).mean(); d["q_max7"]=s.rolling(7).max()
    d["rise_1d"]=(s-s.shift(1))/(s.shift(1)+1e-6); d["rise_3d"]=(s-s.shift(3))/(s.shift(3)+1e-6)
    d["q_over_mean30"]=s/(d["q_mean30"]+1e-6)
    d["q_over_thr"]=s/thr
    doy=s.index.dayofyear.values; d["season_sin"]=np.sin(2*np.pi*doy/365.25); d["season_cos"]=np.cos(2*np.pi*doy/365.25)
    d["mean_q"]=p.mean_q; d["lat"]=p.lat; d["lon"]=p.lon
    fut=pd.concat([s.shift(-k) for k in range(1,H+1)],axis=1).max(axis=1); fut[s.index[-H:]]=np.nan
    d["thr"]=thr; d["high_now"]=(s>=thr).astype(int); d["y"]=(fut>=thr).astype(float); d.loc[fut.isna(),"y"]=np.nan
    d["point_id"]=p.point_id; d["date"]=s.index; rows.append(d)
df=pd.concat(rows,ignore_index=True).dropna(subset=["q_lag7","q_mean30","y"])
onset=df[df.high_now==0].copy()                       # onset task: only days that are NOT already high
onset["y"]=onset.y.astype(int)
onset["split"]=np.where(onset.date<=TRAIN_END,"train",np.where(onset.date<=VALID_END,"valid","test"))
onset.to_csv(f"{ROOT}/data/onset_table.csv",index=False)
print(onset.groupby("split").agg(rows=("y","size"),positives=("y","sum"),rate=("y","mean")).round(4))
print("dates:",onset.groupby("split").date.agg(["min","max"]))
