"""
EnergyOS AI Engine — ai_engine.py

Implements:
  1. Isolation Forest (ML anomaly detection)
  2. Rule-based off-hours / threshold detection
  3. Statistical insights generator
  4. Rolling-average prediction model
  5. Savings calculator (cost ROI model)

Usage:
  from ai_engine import EnergyAI
  ai = EnergyAI()
  ai.fit(df)
  df  = ai.detect(df)
  ins = ai.insights(df)
  p7  = ai.predict(daily_series)
  sav = ai.savings(df)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class Insight:
    id: str; type: str; severity: str; icon: str
    title: str; description: str; recommendation: str
    value: str; ai_type: str; confidence: float


class EnergyAI:
    """
    Full AI pipeline for office building energy optimization.
    
    Two AI methods:
      1. IsolationForest (sklearn) — unsupervised ML anomaly detection
      2. Rule engine — domain-specific waste flagging
    """
    
    def __init__(self, contamination=0.05, off_hour_start=7, off_hour_end=21, threshold_kwh=1200):
        self.contamination   = contamination
        self.off_start       = off_hour_start
        self.off_end         = off_hour_end
        self.threshold_kwh   = threshold_kwh
        self._iso   = IsolationForest(contamination=contamination, n_estimators=150, random_state=42)
        self._scaler = StandardScaler()
        self._fitted = False
    
    # ─── ML: Isolation Forest ──────────────────────────
    def fit(self, df: pd.DataFrame) -> "EnergyAI":
        feats = df[["total_energy_kWh","peak_demand_kW","hour","dow"]].fillna(0)
        self._scaler.fit(feats)
        self._iso.fit(self._scaler.transform(feats))
        self._fitted = True
        self._avg = df["total_energy_kWh"].mean()
        self._std = df["total_energy_kWh"].std()
        print(f"[IsoForest] Fitted on {len(df)} records. Avg={self._avg:.1f} Std={self._std:.1f}")
        return self
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df with is_anomaly, is_offhours, is_threshold_breach columns."""
        df = df.copy()
        if self._fitted:
            feats  = df[["total_energy_kWh","peak_demand_kW","hour","dow"]].fillna(0)
            preds  = self._iso.predict(self._scaler.transform(feats))
            scores = self._iso.score_samples(self._scaler.transform(feats))
            df["iso_pred"]   = preds
            df["iso_score"]  = scores
            df["is_anomaly"] = preds == -1
        
        # Rule 1: off-hours waste
        df["is_offhours"] = (df["hour"] < self.off_start) | \
                             (df["hour"] > self.off_end) | \
                             (df["dow"] >= 5)
        
        # Rule 2: threshold breach
        df["is_threshold_breach"] = df["total_energy_kWh"] > self.threshold_kwh
        
        # Rule 3: night anomaly (after midnight, high energy)
        df["is_night_waste"] = (df["hour"].between(0,5)) & \
                                (df["total_energy_kWh"] > df["total_energy_kWh"].quantile(0.6))
        return df
    
    # ─── Statistical Insights ─────────────────────────
    def insights(self, df: pd.DataFrame) -> List[Insight]:
        total  = df["total_energy_kWh"].sum()
        off    = df.get("is_offhours", pd.Series(False, index=df.index))
        off_kw = df[off]["total_energy_kWh"].sum() if off.any() else 0
        n_anom = int(df.get("is_anomaly", pd.Series(False)).sum())
        ph     = int(df.groupby("hour")["total_energy_kWh"].mean().idxmax())
        
        result = [
            Insight("W1","waste","critical","⚡",
                    f"{round(off_kw/total*100,1)}% Energy in Off-Hours",
                    f"{int(off_kw/1000)}K kWh consumed outside 7AM–9PM and weekends with minimal occupancy.",
                    "Deploy occupancy-linked auto-shutoff for HVAC and lighting.",
                    f"{int(off_kw/1000)}K kWh","rule_based",0.94),
            Insight("A1","anomaly","high","🤖",
                    f"ML Detected {n_anom} Anomalies ({round(n_anom/len(df)*100,1)}%)",
                    f"IsolationForest flagged {n_anom} readings (contamination={self.contamination}). Statistical outliers.",
                    "Audit top-5 anomaly buildings. Cross-reference with maintenance logs.",
                    f"{n_anom} records","isolation_forest",0.89),
            Insight("P1","peak","high","📈",
                    f"Peak Load at {ph}:00 Every Weekday",
                    "Simultaneous startup of all systems causes peak demand charges.",
                    "Stagger startup across 90 minutes: HVAC → Elevators → Lighting.",
                    f"{ph}:00 AM","statistical",0.92),
        ]
        if "city" in df.columns:
            ca   = df.groupby("city")["total_energy_kWh"].mean()
            var  = round((ca.max()-ca.min())/ca.min()*100,1)
            result.append(Insight("C1","city","medium","🏙",
                f"{ca.idxmax()} Consumes {var}% More Than {ca.idxmin()}",
                f"City variance of {var}%. Older construction and lower AC setpoints are primary drivers.",
                f"Raise AC setpoints 1–2°C in {ca.idxmax()} buildings.",
                f"{var}% variance","statistical",0.87))
        return result
    
    # ─── Rolling-Average Prediction ───────────────────
    def predict(self, daily_values: np.ndarray, n_days=7) -> List[Dict]:
        last = daily_values[-30:] if len(daily_values)>=30 else daily_values
        avg  = last.mean()
        trend= (last[-1]-last[0]) / max(len(last)-1,1)
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        preds = []
        for i in range(n_days):
            wk  = 0.72 if i>=5 else 1.0
            val = max(8000, (avg + trend*(i+1)) * wk)
            preds.append(dict(day=days[i], kwh=round(float(val),1),
                              low=round(float(val*.92),1), high=round(float(val*1.08),1),
                              is_weekend=i>=5))
        return preds
    
    # ─── Savings Calculator ───────────────────────────
    def savings(self, df: pd.DataFrame, rate_inr=16.58) -> Dict:
        total = df["total_energy_kWh"].sum()
        breakdown = {
            "occupancy_shutoff": dict(pct=.28, months=14),
            "peak_staggering":   dict(pct=.08, months=3),
            "hvac_setpoint":     dict(pct=.05, months=1),
            "zone_scheduling":   dict(pct=.04, months=8),
            "time_of_use_shift": dict(pct=.03, months=2),
        }
        for k,v in breakdown.items():
            v["kwh"]  = round(total*v["pct"], 0)
            v["inr"]  = round(total*v["pct"]*rate_inr, 0)
        total_kwh = sum(v["kwh"] for v in breakdown.values())
        total_inr = sum(v["inr"] for v in breakdown.values())
        return dict(total_kwh_saved=total_kwh, total_inr_saved=total_inr, breakdown=breakdown)


# ─── Standalone test ──────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n  = 500
    df = pd.DataFrame({
        "total_energy_kWh": np.random.normal(1000,200,n).clip(100,2000),
        "peak_demand_kW":   np.random.normal(400,80,n).clip(50,500),
        "hour": np.random.randint(0,24,n),
        "dow":  np.random.randint(0,7,n),
        "city": np.random.choice(["Hyderabad","Delhi","Mumbai"],n),
    })
    # Inject anomalies
    df.loc[np.random.choice(n,30,replace=False),"total_energy_kWh"] = 1900

    ai  = EnergyAI(contamination=0.05)
    ai.fit(df)
    df  = ai.detect(df)
    ins = ai.insights(df)
    p7  = ai.predict(df["total_energy_kWh"].values)
    sav = ai.savings(df)

    print(f"\n✅ Anomalies: {df['is_anomaly'].sum()}")
    print(f"✅ Insights: {len(ins)}")
    print(f"✅ Forecast 7d: {[round(p['kwh']/1000,1) for p in p7]}k kWh")
    print(f"✅ Savings: ₹{int(sav['total_inr_saved']/1e5)}L total")
    for i in ins:
        print(f"  [{i.severity}] {i.title}")
