"""Builds notebooks/spend_cube_analysis.ipynb from the cells below."""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Procurement Spend Cube & Auto-Classification
**Client:** Indonesian upstream oil & gas operator (name withheld) · **Scope:** 5 years of PO lines (2021-2025), ~855k rows, ~USD 6.2B spend

**Problem.** Category managers could not answer basic questions — *how much do we spend on valves? with how many vendors?* — because
only ~48% of PO lines carried a category, about 1 in 10 of those labels was wrong, vendor records were duplicated, and item descriptions
were free text in mixed English/Bahasa Indonesia.

**What this notebook does**
1. Profiles and cleans the raw SAP PO-line extract
2. Normalises the vendor master (duplicate records under one tax ID)
3. Trains a text classifier that assigns every PO line to a 3-level category taxonomy (UNSPSC-aligned)
4. Builds the spend cube (category × vendor × site × year) and the analyses category managers asked for
5. Exports the tables behind the dashboard (`docs/index.html`)

> Data note: the original client data is confidential. This repository runs on a synthetic replica generated to match the
> structure, volumes and data-quality issues of the original extract (`src/generate_data.py`).
""")

md("## 1. Load & profile")
code("""
import re, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30, "display.width", 160, "display.float_format", "{:,.2f}".format)

RAW, OUT, FIG = Path("../data/raw"), Path("../data/processed"), Path("../outputs/figures")
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.facecolor": "#f6f4ef", "axes.facecolor": "#f6f4ef", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold"})
ACC = "#1f5f5b"

# full extract: one CSV (from src/generate_data.py) or the yearly zipped CSVs shipped in the repo
full = RAW / "po_lines_2021_2025.csv"
files = [full] if full.exists() else sorted(RAW.glob("po_lines_20??.csv.zip"))
po = pd.concat([pd.read_csv(f, dtype={"po_number": str, "material_code": str}) for f in files], ignore_index=True)
vendors = pd.read_csv(RAW / "vendor_master.csv")
taxo = pd.read_csv(RAW / "category_taxonomy.csv")
qa = pd.read_csv(RAW / "qa_validated_sample.csv")
print(f"PO lines: {len(po):,} | POs: {po.po_number.nunique():,} | vendors in master: {len(vendors):,} | categories (L3): {len(taxo)}")
po.head()
""")
code("""
profile = pd.DataFrame({"dtype": po.dtypes.astype(str), "null_%": po.isna().mean().mul(100).round(2), "unique": po.nunique()})
profile
""")
code("""
print("Exact duplicate lines   :", po.drop(columns="line_id").duplicated().sum())
print("Plant codes             :", sorted(po.plant.dropna().unique()))
print("Currency codes          :", po.currency.value_counts().to_dict())
print("Zero-value lines        :", (po.net_value_usd == 0).sum())
valid = set(taxo.category_l3)
cm = po.category_manual
print(f"Lines with a category   : {cm.notna().mean():.1%}  (of which not in taxonomy: {(~cm.dropna().isin(valid)).sum():,})")
""")

md("## 2. Cleaning")
code("""
df = po.copy()
n0 = len(df)
df = df[~df.drop(columns="line_id").duplicated()]                 # exact duplicates from the extract
df["plant"] = df["plant"].str.upper()
# plant missing -> recover from cost centre prefix
cc_map = {"HOJKT": "HO-JKT", "ONSSKT": "ONS-SKT", "OFFMRB": "OFF-MRB", "SBBPN": "SB-BPN", "SBPLM": "SB-PLM"}
df["plant"] = df["plant"].fillna(df["cost_center"].str.split("-").str[0].map(cc_map))
df["currency"] = df["currency"].replace({"Rp": "IDR"})
zero = df["net_value_usd"].eq(0)
df = df[~zero]
df["po_date"] = pd.to_datetime(df["po_date"])
df["year"] = df.po_date.dt.year
# reconcile USD value with currency and monthly FX
chk = np.where(df.currency.eq("USD"), df.net_value, df.net_value / df.fx_rate_idr_usd)
print(f"Removed duplicates: {n0 - len(df) - zero.sum():,} | removed zero-value lines: {zero.sum():,} | rows now: {len(df):,}")
print("Max USD reconciliation gap:", round(np.abs(chk - df.net_value_usd).max(), 2))
print("Plant nulls after fix:", df.plant.isna().sum())
""")

md("## 3. Vendor normalisation\nSame legal entity, several vendor IDs (e.g. `PT Sinar Baja Teknik` vs `SINAR BAJA TEKNIK, PT`). Tax ID is the anchor; the earliest record becomes the golden record.")
code("""
def clean_name(s):
    s = s.upper()
    s = re.sub(r"\\b(PT|P\\.T|CV)\\b\\.?", " ", s)
    s = re.sub(r"\\b(PTE\\.? LTD\\.?|SDN\\.? BHD\\.?|INC\\.?|LIMITED|LTD\\.?|CO\\.?,? LTD\\.?|GMBH|FZE)\\b", " ", s)
    s = re.sub(r"\\(BRANCH\\)", " ", s)
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", s).split())

vendors["name_key"] = vendors.vendor_name.map(clean_name)
golden = (vendors.sort_values("created_date").groupby("tax_id")
          .agg(master_vendor_id=("vendor_id", "first"), master_vendor_name=("vendor_name", "first"), n_records=("vendor_id", "size")))
vmap = vendors.merge(golden, on="tax_id")
dup = vmap[vmap.n_records > 1]
print(f"Vendor IDs: {len(vendors):,} -> unique legal entities: {len(golden):,} ({len(dup) - dup.tax_id.nunique():,} duplicate records)")
print("Name-key check: duplicates whose cleaned names also match:", (dup.groupby('tax_id').name_key.nunique() == 1).mean().round(3))
dup.sort_values("tax_id")[["tax_id", "vendor_id", "vendor_name", "created_date"]].head(8)
""")
code("""
df = df.merge(vmap[["vendor_id", "master_vendor_id", "master_vendor_name", "country", "vendor_type"]], on="vendor_id", how="left")
df["vendor_origin"] = np.where(df.country.eq("Indonesia"), "Local (Indonesia)", "Foreign")
spend_on_dups = df[df.vendor_id != df.master_vendor_id].net_value_usd.sum()
print(f"Spend booked on duplicate vendor IDs: USD {spend_on_dups/1e6:,.1f}M — invisible in vendor-level reports before the merge")
""")

md("## 4. Auto-classification of PO lines\n"
   "Training labels = buyer-assigned categories that exist in the taxonomy (noisy). "
   "Evaluation uses the **2,000-line sample validated by category managers**, which the model never sees.")
code("""
def norm_desc(s):
    s = str(s).lower()
    s = re.sub(r"\\bex pr\\d+", " ", s)
    s = s.replace("urgent -", " ")
    s = re.sub(r"(\\d+)\\s*(\\")", r"\\1in", s)
    s = re.sub(r"[^a-z0-9/#\\.\\- ]", " ", s)
    return re.sub(r"\\s+", " ", s).strip()

df["desc_norm"] = df.item_description.map(norm_desc)
df["text"] = df.desc_norm + " uom_" + df.uom.str.lower() + " type_" + df.po_type.str.lower()
lab = df[df.category_manual.isin(valid)]
print(f"Labelled lines usable for training: {len(lab):,} ({len(lab)/len(df):.1%})")

qa_ids = set(qa.line_id)
train = lab[~lab.line_id.isin(qa_ids)].drop_duplicates(["text", "category_manual"])
print(f"Unique (text, label) training pairs: {len(train):,}")
""")
code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

vec = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=250_000),
                 TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))
clf = CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)   # linear SVM + calibrated probabilities for the confidence score
model = make_pipeline(vec, clf)
model.fit(train.text, train.category_manual)

# predict once per unique text (many lines share a description)
u = df[["text"]].drop_duplicates()
proba = model.predict_proba(u.text)
u["pred_l3"] = model.classes_[proba.argmax(1)]
u["confidence"] = proba.max(1)
df = df.merge(u, on="text", how="left")
print(f"Unique texts scored: {len(u):,}")
""")
code("""
ev = qa.merge(df[["line_id", "category_manual", "pred_l3", "confidence"]], on="line_id")
manual_acc = (ev.category_manual == ev.category_l3_validated)[ev.category_manual.notna()].mean()
model_acc = accuracy_score(ev.category_l3_validated, ev.pred_l3)
print(f"QA sample: {len(ev):,} validated lines")
print(f"  Buyer labels (where present, {ev.category_manual.notna().mean():.0%} of sample) accuracy : {manual_acc:.1%}")
print(f"  Model accuracy, all lines                            : {model_acc:.1%}")
bands = pd.DataFrame([{"min_confidence": t, "share_of_lines_%": (ev.confidence >= t).mean() * 100,
                       "accuracy_%": (ev.pred_l3 == ev.category_l3_validated)[ev.confidence >= t].mean() * 100} for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
bands.round(1)
""")
code("""
rep = pd.DataFrame(classification_report(ev.category_l3_validated, ev.pred_l3, output_dict=True, zero_division=0)).T
rep = rep.drop(["accuracy", "macro avg", "weighted avg"]).sort_values("f1-score")
rep.head(8)
""")
md("### Final category rule\n"
   "* Model confidence ≥ 0.20 (see accuracy-by-confidence table above) → model category (also overrides a buyer label that disagrees — those are the mislabels)\n"
   "* Below 0.20 → keep the buyer label if it exists, otherwise send to the **manual review queue**")
code("""
THR = 0.20
has_manual = df.category_manual.isin(valid)
df["category_l3"] = np.where(df.confidence >= THR, df.pred_l3, np.where(has_manual, df.category_manual, "Unclassified (review)"))
df["class_source"] = np.where(df.confidence >= THR, "model", np.where(has_manual, "buyer", "review queue"))
relabel = has_manual & (df.confidence >= 0.6) & (df.pred_l3 != df.category_manual)

ev = qa.merge(df[["line_id", "category_l3"]], on="line_id")
auto = ev.category_l3 != "Unclassified (review)"
final_acc = (ev.category_l3 == ev.category_l3_validated)[auto].mean()
cov_before = df.category_manual.isin(valid).mean()
cov_after = (df.category_l3 != "Unclassified (review)").mean()
review = df[df.category_l3 == "Unclassified (review)"]
print(f"Coverage: {cov_before:.1%} of lines -> {cov_after:.1%} | spend coverage -> {1 - review.net_value_usd.sum()/df.net_value_usd.sum():.2%}")
print(f"Final category accuracy on QA sample (lines not sent to review, {auto.mean():.1%} of sample): {final_acc:.1%}")
print(f"Buyer labels overridden as likely mislabels (confidence >= 0.6): {relabel.sum():,} lines, USD {df[relabel].net_value_usd.sum()/1e6:,.0f}M")
print(f"Manual review queue: {len(review):,} lines, USD {review.net_value_usd.sum()/1e6:,.1f}M")
""")
code("""
df = df.merge(taxo.rename(columns={"po_type": "tax_po_type"}), on="category_l3", how="left")
for c in ["category_l1", "category_l2"]:
    df[c] = df[c].fillna("Unclassified (review)")
df[["line_id", "item_description", "category_manual", "pred_l3", "confidence", "category_l3", "class_source"]].sample(8, random_state=3)
""")

md("## 5. Spend cube")
code("""
cube = (df.groupby(["year", "plant", "category_l1", "category_l2", "category_l3", "master_vendor_id", "master_vendor_name", "vendor_origin"], dropna=False)
          .agg(spend_usd=("net_value_usd", "sum"), po_lines=("line_id", "size"), pos=("po_number", "nunique")).reset_index())
cube.to_csv(OUT / "spend_cube.csv", index=False)
print(f"Spend cube rows: {len(cube):,}")
tot = df.net_value_usd.sum()
yr = df.groupby("year").net_value_usd.sum().div(1e6)
print(f"Total spend USD {tot/1e9:,.2f}B | by year (USD M):", yr.round(0).to_dict())
""")
code("""
l1 = df.groupby("category_l1").agg(spend=("net_value_usd", "sum"), lines=("line_id", "size"),
                                   vendors=("master_vendor_id", "nunique")).sort_values("spend", ascending=False)
l1["spend_%"] = l1.spend / tot * 100
l1["lines_%"] = l1.lines / len(df) * 100
l1["usd_per_line"] = l1.spend / l1.lines
l1.assign(spend=l1.spend / 1e6).round(1)
""")
code("""
fig, ax = plt.subplots(figsize=(9, 5))
s = l1.spend.div(1e6).sort_values()
ax.barh(s.index, s.values, color=ACC)
for i, v in enumerate(s.values):
    ax.text(v, i, f"  {v:,.0f}", va="center", fontsize=9)
ax.set_title("Spend by category L1, 2021-2025 (USD M)"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(FIG / "spend_by_l1.png", dpi=130); plt.show()
""")

md("### 5.1 Pareto & tail spend")
code("""
vs = df.groupby("master_vendor_id").net_value_usd.sum().sort_values(ascending=False)
cum = vs.cumsum() / vs.sum()
n80 = int((cum < 0.8).sum() + 1)
tail = vs[cum > 0.95]
print(f"Active vendors: {len(vs):,} | vendors making 80% of spend: {n80} ({n80/len(vs):.1%})")
print(f"Tail (last 5% of spend): {len(tail):,} vendors, USD {tail.sum()/1e6:,.0f}M, "
      f"{df[df.master_vendor_id.isin(tail.index)].po_number.nunique():,} POs")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values * 100, color=ACC, lw=2)
ax.axhline(80, ls="--", c="grey", lw=.8); ax.axvline(n80 / len(vs) * 100, ls="--", c="grey", lw=.8)
ax.set_xlabel("% of vendors"); ax.set_ylabel("% of spend"); ax.set_title("Vendor Pareto curve")
plt.tight_layout(); plt.savefig(FIG / "pareto.png", dpi=130); plt.show()
""")

md("### 5.2 Vendor fragmentation by category\nHow many vendors does each L3 category use, and how many would cover 80% of its spend?")
code("""
def frag(g):
    s = g.groupby("master_vendor_id").net_value_usd.sum().sort_values(ascending=False)
    c = s.cumsum() / s.sum()
    return pd.Series({"spend_usd_m": s.sum() / 1e6, "vendors": len(s), "vendors_for_80pct": int((c < .8).sum() + 1),
                      "top1_share_%": s.iloc[0] / s.sum() * 100, "po_lines": len(g)})
fr = df[df.category_l3 != "Unclassified (review)"].groupby(["category_l1", "category_l3"]).apply(frag).reset_index()
fr["frag_ratio"] = fr.vendors / fr.vendors_for_80pct
fr.sort_values("vendors", ascending=False).head(12).round(1)
""")

md("### 5.3 Contract coverage, local vs foreign, currency")
code("""
df["on_contract"] = df.contract_no.notna()
cc = df.groupby("category_l1").apply(lambda g: pd.Series({
        "spend_usd_m": g.net_value_usd.sum() / 1e6,
        "contract_cov_%": g.loc[g.on_contract, "net_value_usd"].sum() / g.net_value_usd.sum() * 100,
        "local_vendor_%": g.loc[g.vendor_origin.str.startswith("Local"), "net_value_usd"].sum() / g.net_value_usd.sum() * 100,
        "usd_po_%": g.loc[g.currency.eq("USD"), "net_value_usd"].sum() / g.net_value_usd.sum() * 100})).sort_values("spend_usd_m", ascending=False)
print(f"Overall contract coverage: {df.loc[df.on_contract,'net_value_usd'].sum()/tot:.1%} | local vendor share: "
      f"{df.loc[df.vendor_origin.str.startswith('Local'),'net_value_usd'].sum()/tot:.1%}")
cc.round(1)
""")

md("### 5.4 Trend & seasonality")
code("""
m = df.set_index("po_date").resample("MS").net_value_usd.sum().div(1e6)
drill = df[df.category_l1.isin(["Drilling & Completion Materials", "Well & Drilling Services"])].set_index("po_date").resample("MS").net_value_usd.sum().div(1e6)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(m.index, m.values, color=ACC, lw=1.8, label="Total")
ax.plot(drill.index, drill.values, color="#c07a2c", lw=1.4, label="Drilling-related")
ax.set_title("Monthly spend (USD M)"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(FIG / "monthly_trend.png", dpi=130); plt.show()
ga = df[df.category_l1.isin(["Corporate & G&A", "IT & Telecom", "MRO & Consumables", "HSE & Safety"])]
nd = ga[ga.po_date.dt.month.isin([11, 12])].net_value_usd.sum() / ga.net_value_usd.sum()
print(f"Indirect/MRO spend placed in Nov-Dec: {nd:.1%} (2 of 12 months = 16.7%) -> year-end budget flush")
""")

md("### 5.5 Price dispersion on identical items\nSame material code, same year, different vendors — how far apart are the unit prices (in USD)?")
code("""
mat = df[df.material_code.notna() & df.unit_price.gt(0)].copy()
mat["unit_usd"] = mat.net_value_usd / mat.quantity
g = mat.groupby(["material_code", "year"]).agg(n=("unit_usd", "size"), vendors=("master_vendor_id", "nunique"),
                                              p25=("unit_usd", lambda x: x.quantile(.25)), med=("unit_usd", "median"))
g = g[(g.n >= 5) & (g.vendors >= 2)]
mat = mat.merge(g[["med", "p25"]], left_on=["material_code", "year"], right_index=True)
above = mat[mat.unit_usd > mat.p25]
saving = ((above.unit_usd - above.p25) * above.quantity).sum()
print(f"Material codes bought from 2+ vendors in the same year: {g.reset_index().material_code.nunique():,}")
print(f"Spend on those lines: USD {mat.net_value_usd.sum()/1e6:,.0f}M | gap to each item's P25 price: USD {saving/1e6:,.1f}M "
      f"({saving/mat.net_value_usd.sum():.1%})")
""")

md("## 6. Export for dashboard")
code("""
def top(s, n): return s.sort_values(ascending=False).head(n)
by_year_l1 = df.pivot_table(index="year", columns="category_l1", values="net_value_usd", aggfunc="sum").div(1e6).round(2)
top_vendors = df.groupby(["master_vendor_name", "vendor_origin"]).agg(spend=("net_value_usd", "sum"), l3=("category_l3", lambda x: x.value_counts().index[0]),
                                                                    lines=("line_id", "size")).reset_index().sort_values("spend", ascending=False).head(15)
l3 = df.groupby(["category_l1", "category_l3"]).net_value_usd.sum().div(1e6).reset_index()
plant = df.groupby("plant").net_value_usd.sum().div(1e6)
monthly = m.round(2)
dash = {
  "kpi": {"spend_b": round(tot / 1e9, 2), "lines": int(len(df)), "pos": int(df.po_number.nunique()), "vendors": int(len(vs)),
          "vendor_ids_raw": int(len(vendors)), "cov_before": round(cov_before * 100, 1), "cov_after": round(cov_after * 100, 1),
          "spend_cov_after": round((1 - review.net_value_usd.sum() / tot) * 100, 1), "qa_acc": round(final_acc * 100, 1),
          "buyer_acc": round(manual_acc * 100, 1), "relabel_lines": int(relabel.sum()), "relabel_usd_m": round(df[relabel].net_value_usd.sum() / 1e6, 1),
          "review_lines": int(len(review)), "n80": n80, "tail_vendors": int(len(tail)), "tail_usd_m": round(tail.sum() / 1e6, 1),
          "contract_cov": round(df.loc[df.on_contract, "net_value_usd"].sum() / tot * 100, 1),
          "local_share": round(df.loc[df.vendor_origin.str.startswith("Local"), "net_value_usd"].sum() / tot * 100, 1),
          "price_gap_usd_m": round(saving / 1e6, 1), "dup_vendor_records": int(len(dup) - dup.tax_id.nunique()),
          "dup_spend_usd_m": round(spend_on_dups / 1e6, 1), "yearend_share": round(nd * 100, 1)},
  "years": [int(y) for y in by_year_l1.index], "by_year_l1": {c: by_year_l1[c].fillna(0).tolist() for c in by_year_l1.columns},
  "l1": [{"name": k, "spend": round(v.spend / 1e6, 1), "lines": int(v.lines), "vendors": int(v.vendors)} for k, v in l1.iterrows()],
  "l3": [{"l1": r.category_l1, "l3": r.category_l3, "spend": round(r.net_value_usd, 1)} for r in l3.itertuples()],
  "frag": fr.sort_values("vendors", ascending=False).head(15).round(1).to_dict("records"),
  "top_vendors": [{"name": r.master_vendor_name, "origin": r.vendor_origin, "spend": round(r.spend / 1e6, 1), "l3": r.l3, "lines": int(r.lines)} for r in top_vendors.itertuples()],
  "plant": {k: round(v, 1) for k, v in plant.items()},
  "monthly": {"labels": [d.strftime("%Y-%m") for d in monthly.index], "total": monthly.tolist(), "drilling": drill.round(2).tolist()},
  "contract": cc.round(1).reset_index().to_dict("records"),
  "pareto": {"x": [round(x, 2) for x in (np.arange(1, len(cum) + 1) / len(cum) * 100)[::5]], "y": [round(y * 100, 2) for y in cum.values[::5]]},
}
Path("../docs").mkdir(exist_ok=True)
(Path("../docs") / "dashboard_data.json").write_text(json.dumps(dash))
df[["line_id", "po_number", "po_line", "po_date", "plant", "master_vendor_id", "master_vendor_name", "item_description", "net_value_usd",
    "category_l1", "category_l2", "category_l3", "class_source", "confidence"]].to_parquet(OUT / "po_lines_classified.parquet", index=False)
review.sort_values("net_value_usd", ascending=False)[["line_id", "po_number", "item_description", "net_value_usd", "pred_l3", "confidence"]] \\
      .to_csv(OUT / "manual_review_queue.csv", index=False)
print(json.dumps(dash["kpi"], indent=1))
""")

code("""
from IPython.display import Markdown
k = dash["kpi"]
mat_cov = cc.loc[["Production & Process Equipment", "MRO & Consumables", "Drilling & Completion Materials", "Production Chemicals"], "contract_cov_%"]
srv_cov = cc.loc[["Well & Drilling Services", "Marine, Aviation & Logistics", "Engineering, Construction & Maintenance"], "contract_cov_%"].mean()
fv = fr[fr.category_l1.isin(["Production & Process Equipment", "MRO & Consumables"])]
Markdown(f\"\"\"## 7. Findings & recommendations

| # | Finding | Recommendation |
|---|---------|----------------|
| 1 | Category coverage rose from **{k['cov_before']}% to {k['cov_after']}% of PO lines ({k['spend_cov_after']}% of spend)**. On the validated sample, auto-assigned categories are **{k['qa_acc']}%** correct vs **{k['buyer_acc']}%** for buyer labels. | Suggest the category at PR creation; buyers confirm instead of typing. |
| 2 | **{k['relabel_lines']:,} buyer labels (USD {k['relabel_usd_m']}M)** contradict a high-confidence prediction and were corrected; **{k['review_lines']:,} lines** have vague descriptions ("sesuai penawaran", "lihat lampiran") and sit in the review queue. | Make the item description a structured, required field; block "as per attachment" descriptions above a value threshold. |
| 3 | **{k['dup_vendor_records']} duplicate vendor records** (same tax ID) held **USD {k['dup_spend_usd_m']}M** that vendor-level reports split in two. | Merge the records; block vendor creation when the tax ID already exists. |
| 4 | **{k['n80']} of {k['vendors']:,} vendors cover 80% of spend**; the last 5% of spend is spread over **{k['tail_vendors']} vendors** (USD {k['tail_usd_m']}M). | Move tail MRO/indirect buying to catalogues and a short distributor panel. |
| 5 | Process-equipment and MRO categories use **{int(fv.vendors.min())}–{int(fv.vendors.max())} vendors** each while **{int(fv.vendors_for_80pct.min())}–{int(fv.vendors_for_80pct.max())}** cover 80% of their spend. Material categories are **{mat_cov.min():.0f}–{mat_cov.max():.0f}%** on contract vs ~{srv_cov:.0f}% for major services. | Framework agreements for valves, pipe & fittings, instrumentation, pumps and MRO consumables. |
| 6 | On identical material codes bought from 2+ vendors in the same year, prices above each item's P25 add up to **USD {k['price_gap_usd_m']}M** (upper bound). | Price harmonisation in the framework tenders; realistic capture of 20–30% is **USD {k['price_gap_usd_m']*0.2:.0f}–{k['price_gap_usd_m']*0.3:.0f}M over 5 years**. |
| 7 | **{k['yearend_share']}%** of indirect/MRO spend lands in Nov–Dec (16.7% if spread evenly). | Quarterly budget release and a PR cut-off date for non-critical indirect items. |
| 8 | Spend grew from **USD {yr[2021]:,.0f}M (2021) to USD {yr.max():,.0f}M ({yr.idxmax()})** with the drilling campaign. | Plan OCTG, bits and rig services against the drilling schedule 12 months ahead. |
\"\"\")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
Path(__file__).resolve().parents[1].joinpath("notebooks", "spend_cube_analysis.ipynb").write_text(nbf.writes(nb))
print("notebook written")
