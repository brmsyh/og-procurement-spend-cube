"""Generate the synthetic replica of the client's PO-line extract (2021-2025).

The structure mirrors an SAP ME2N / EKPO extract from an Indonesian upstream
oil & gas operator. Values are synthetic; the original data is confidential.

Usage:  python src/generate_data.py            (writes to data/raw/)
Seeded, so every run produces the same files.
"""
import random
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

from taxonomy import TAXONOMY, PLANTS, L1_DEPT, GL

SEED = 2026
TARGET_LINES = 852_317          # PO lines in the client extract
TARGET_SPEND_USD = 6.18e9        # 5-year addressable spend (~USD 1.2B per year)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HIDDEN = ROOT.parent / "_hidden"  # ground truth, never shipped in the repo

rng = np.random.default_rng(SEED)
random.seed(SEED)

# --------------------------------------------------------------------------- calendar
months = pd.period_range("2021-01", "2025-12", freq="M")
year_drill = {2021: 0.62, 2022: 0.88, 2023: 1.25, 2024: 1.38, 2025: 1.02}
year_base = {2021: 0.90, 2022: 0.98, 2023: 1.06, 2024: 1.12, 2025: 1.10}
eid_month = {2021: 5, 2022: 5, 2023: 4, 2024: 4, 2025: 3}
inflation = {2021: 1.00, 2022: 1.07, 2023: 1.11, 2024: 1.14, 2025: 1.17}
DRILL_L1 = {"Drilling & Completion Materials", "Well & Drilling Services"}
FLUSH_L1 = {"Corporate & G&A", "IT & Telecom", "MRO & Consumables", "HSE & Safety"}


def month_weights(l1):
    w = []
    for p in months:
        y, m = p.year, p.month
        v = year_drill[y] if l1 in DRILL_L1 else year_base[y]
        if l1 in DRILL_L1:  # campaigns ramp up after Q1 budget approval
            v *= [0.7, 0.8, 0.95, 1.05, 1.1, 1.1, 1.1, 1.1, 1.05, 1.0, 1.0, 0.95][m - 1]
        if l1 in FLUSH_L1 and m in (11, 12):
            v *= 1.35       # year-end budget flush
        if m == eid_month[y]:
            v *= 0.82
        if m == 1:
            v *= 0.85       # slow January
        w.append(v)
    w = np.array(w)
    return w / w.sum()


# fx IDR per USD, monthly with drift and noise
anchor = {2021: 14_300, 2022: 14_850, 2023: 15_240, 2024: 15_850, 2025: 16_350}
fx = {}
for p in months:
    a, b = anchor[p.year], anchor.get(p.year + 1, anchor[p.year] + 300)
    fx[str(p)] = round(a + (b - a) * (p.month - 1) / 12 + rng.normal(0, 90), 0)

MONTH_ID = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
MONTH_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fmt_month(y, m):
    r = random.random()
    if r < 0.35:
        return f"{MONTH_EN[m-1]}-{str(y)[2:]}"
    if r < 0.6:
        return f"{MONTH_ID[m-1]} {y}"
    if r < 0.8:
        return f"{MONTH_EN[m-1].upper()} {y}"
    return f"{m:02d}/{y}"


# --------------------------------------------------------------------------- vendors
ID_WORDS = ["Sinar", "Bumi", "Mitra", "Karya", "Teknik", "Energi", "Samudra", "Nusa", "Jaya", "Prima", "Abadi", "Mandiri", "Perkasa", "Sentosa",
            "Andalan", "Cipta", "Daya", "Graha", "Inti", "Lestari", "Makmur", "Pratama", "Rekayasa", "Sarana", "Tirta", "Utama", "Bahari", "Petro",
            "Indo", "Baja", "Patra", "Garuda", "Rajawali", "Kencana", "Mulia", "Sejahtera", "Persada", "Arta", "Buana", "Citra", "Dirgantara",
            "Eka", "Gemilang", "Harapan", "Kurnia", "Lintas", "Megah", "Nusantara", "Pilar", "Sakti", "Tama", "Wahana", "Yasa", "Borneo", "Sriwijaya"]
ID_SUFFIX = ["Teknik", "Energi", "Engineering", "Supply", "Servis", "Logistik", "Industri", "Kimia", "Oilfield", "Marine", "Trading", "Solusi", "Kontraktor"]
EN_A = ["Meridian", "Apex", "Crestline", "Harbor", "Ironbridge", "Northstar", "Pacific", "Redwater", "Summit", "Trident", "Westgate", "Bluefin",
        "Cobalt", "Halcyon", "Keystone", "Orion", "Sterling", "Vanguard", "Atlas", "Corvex", "Delta Rho", "Falcon", "Granite", "Lumen", "Strata"]
EN_B = ["Oilfield Supply", "Energy Services", "Well Technologies", "Flow Control", "Drilling Products", "Process Systems", "Offshore",
        "Industrial", "Subsea", "Instruments", "Valve & Controls", "Engineering", "Chemicals", "Tubulars", "Marine Services"]
FOREIGN = [("Singapore", "Pte Ltd", 0.32), ("Malaysia", "Sdn Bhd", 0.14), ("United States", "Inc", 0.16), ("United Kingdom", "Ltd", 0.07),
           ("China", "Co., Ltd", 0.12), ("South Korea", "Co., Ltd", 0.05), ("Japan", "Co., Ltd", 0.04), ("Germany", "GmbH", 0.04),
           ("United Arab Emirates", "FZE", 0.06)]
ID_CITIES = [("Jakarta", 0.45), ("Balikpapan", 0.14), ("Palembang", 0.12), ("Surabaya", 0.09), ("Batam", 0.07), ("Bekasi", 0.06),
             ("Tangerang", 0.04), ("Samarinda", 0.03)]
used_names = set()


SUFFIX = {  # (Indonesian suffixes, foreign suffixes) by the vendor's main category family
    "Drilling & Completion Materials": (["Oilfield", "Tubular", "Drilling Supply", "Petro Supply", "Oilfield Supply"], ["Oilfield Supply", "Tubulars", "Drilling Products", "Well Technologies", "Completion Systems"]),
    "Production & Process Equipment": (["Teknik", "Valve Indonesia", "Instrumentasi", "Industri", "Engineering", "Pompa Indonesia"], ["Flow Control", "Valve & Controls", "Instruments", "Process Systems", "Industrial", "Pumps"]),
    "Production Chemicals": (["Kimia", "Chemical Indonesia", "Kimia Industri"], ["Chemicals", "Specialty Chemicals", "Production Chemicals"]),
    "MRO & Consumables": (["Teknik", "Supply", "Trading", "Industri", "Sarana Teknik"], ["Industrial Supply", "Bearings", "Industrial"]),
    "Fuel & Lubricants": (["Energi", "Niaga Energi", "Petro Niaga", "Lubrindo"], ["Lubricants", "Energy Trading"]),
    "HSE & Safety": (["Safety", "Proteksi", "Safety Indonesia"], ["Safety Systems", "Fire & Safety"]),
    "Well & Drilling Services": (["Drilling", "Oilfield Services", "Energi Servis", "Well Services"], ["Energy Services", "Well Services", "Drilling", "Oilfield Services"]),
    "Marine, Aviation & Logistics": (["Logistik", "Bahari", "Transport", "Pelayaran", "Ekspres Logistik"], ["Marine Services", "Offshore", "Logistics", "Aviation"]),
    "Engineering, Construction & Maintenance": (["Kontraktor", "Engineering", "Konstruksi", "Rekayasa", "Inspeksi"], ["Engineering", "Inspection", "Offshore Construction"]),
    "Facilities & Site Support": (["Katering", "Servis", "Jasa Utama", "Medika", "Lingkungan"], ["Facility Services"]),
    "IT & Telecom": (["Teknologi", "Data Solusi", "Telematika", "Komunikasi"], ["Technologies", "Software", "Networks"]),
    "Corporate & G&A": (["Konsultan", "Solusi", "Wisata", "Properti", "Kertas"], ["Advisory", "Consulting", "Partners"]),
}


def new_name(foreign, l1):
    id_suf, en_suf = SUFFIX[l1]
    while True:
        if foreign:
            ctry, suf, _ = random.choices(FOREIGN, weights=[f[2] for f in FOREIGN])[0]
            nm = f"{random.choice(EN_A)} {random.choice(en_suf)} {suf}"
        else:
            ctry = "Indonesia"
            k = random.random()
            if k < 0.8:
                nm = f"PT {random.choice(ID_WORDS)} {random.choice(ID_WORDS)} {random.choice(id_suf)}"
            else:
                nm = f"CV {random.choice(ID_WORDS)} {random.choice(id_suf)}"
            nm = re.sub(r"\b(\w+) \1\b", r"\1", nm)
        if nm not in used_names:
            used_names.add(nm)
            return nm, ctry


vendors = []        # dicts
l3_pool = {}        # l3 -> list of (vendor_id, weight)
vid = 100000


def add_vendor(foreign, l1):
    global vid
    vid += 1
    nm, ctry = new_name(foreign, l1)
    city = random.choices([c for c, _ in ID_CITIES], weights=[w for _, w in ID_CITIES])[0] if ctry == "Indonesia" else None
    v = dict(vendor_id=f"V{vid}", vendor_name=nm, country=ctry, city=city, primary_l1=l1,
             tax_id=("".join(random.choices(string.digits, k=15)) if ctry == "Indonesia" else "FRN-" + "".join(random.choices(string.digits, k=9))),
             price_factor=float(np.clip(rng.normal(1.0, 0.09), 0.78, 1.32)),
             vendor_type=None, created_date=None)
    vendors.append(v)
    return v


by_l1 = {}
for t in TAXONOMY:
    pool = []
    # long tail: pool size inflated so tail vendors exist
    n = int(t["pool"] * rng.uniform(2.2, 3.4))
    for _ in range(n):
        foreign = random.random() < t["foreign"]
        reuse = [v for v in by_l1.get(t["l1"], []) if (v["country"] != "Indonesia") == foreign]
        if reuse and random.random() < 0.28:
            v = random.choice(reuse)
        else:
            v = add_vendor(foreign, t["l1"])
            by_l1.setdefault(t["l1"], []).append(v)
        pool.append(v["vendor_id"])
    pool = list(dict.fromkeys(pool))
    w = 1 / np.arange(1, len(pool) + 1) ** 1.15      # Zipf-like: few vendors dominate
    rng.shuffle(w)
    l3_pool[t["l3"]] = (pool, w / w.sum())

vtype_map = {"Well & Drilling Services": "Service Contractor", "Marine, Aviation & Logistics": "Service Contractor",
             "Engineering, Construction & Maintenance": "Service Contractor", "Facilities & Site Support": "Service Contractor"}
for v in vendors:
    v["vendor_type"] = vtype_map.get(v["primary_l1"], random.choice(["Manufacturer", "Distributor", "Distributor", "Agent"]))
    v["created_date"] = str(pd.Timestamp("2008-01-01") + pd.Timedelta(days=int(rng.integers(0, 4600))))[:10]

# duplicate vendor records (same tax id, different spelling) -- a classic master-data issue
alias_of = {}
dups = random.sample(vendors, int(len(vendors) * 0.055))
for v in dups:
    nm = v["vendor_name"]
    r = random.random()
    if nm.startswith("PT "):
        alt = (nm[3:].upper() + ", PT") if r < 0.4 else ("PT. " + nm[3:]) if r < 0.75 else nm.replace("PT ", "P.T ").upper()
    elif nm.startswith("CV "):
        alt = nm[3:] + ", CV"
    else:
        alt = nm.upper() if r < 0.5 else nm.replace(" Pte Ltd", " Pte. Ltd.").replace(" Inc", ", Inc.").replace(" Sdn Bhd", " Sdn. Bhd.").replace(" Ltd", " Limited")
        if alt == nm:
            alt = nm + " (Branch)"
    vid += 1
    a = dict(v)
    a.update(vendor_id=f"V{vid}", vendor_name=alt,
             created_date=str(min(pd.Timestamp(v["created_date"]) + pd.Timedelta(days=int(rng.integers(200, 2500))), pd.Timestamp("2020-12-31")))[:10])
    vendors.append(a)
    alias_of[v["vendor_id"]] = a["vendor_id"]

vend_df = pd.DataFrame(vendors)
vend_price = dict(zip(vend_df.vendor_id, vend_df.price_factor))
vend_foreign = dict(zip(vend_df.vendor_id, vend_df.country != "Indonesia"))

# --------------------------------------------------------------------------- PO generation
tax = {t["l3"]: t for t in TAXONOMY}
l3_names = [t["l3"] for t in TAXONOMY]
share = np.array([t["share"] for t in TAXONOMY], float)
# lines per PO: material POs carry more lines than service POs
lpp = np.array([3.6 if t["po_type"] == "Material" else 1.35 for t in TAXONOMY])
po_w = share / lpp
po_w /= po_w.sum()
mw = {t["l1"]: month_weights(t["l1"]) for t in TAXONOMY}
siblings = {}
for t in TAXONOMY:
    siblings.setdefault(t["l1"], []).append(t["l3"])

PLANT_W = {
    "Drilling & Completion Materials": {"ONS-SKT": .35, "OFF-MRB": .25, "SB-BPN": .2, "SB-PLM": .2},
    "Well & Drilling Services": {"ONS-SKT": .5, "OFF-MRB": .5},
    "Production & Process Equipment": {"ONS-SKT": .3, "OFF-MRB": .3, "SB-BPN": .2, "SB-PLM": .15, "HO-JKT": .05},
    "Production Chemicals": {"ONS-SKT": .45, "OFF-MRB": .45, "SB-BPN": .1},
    "MRO & Consumables": {"ONS-SKT": .25, "OFF-MRB": .2, "SB-BPN": .25, "SB-PLM": .25, "HO-JKT": .05},
    "Fuel & Lubricants": {"ONS-SKT": .3, "OFF-MRB": .3, "SB-BPN": .25, "SB-PLM": .15},
    "HSE & Safety": {"ONS-SKT": .3, "OFF-MRB": .3, "SB-BPN": .15, "SB-PLM": .15, "HO-JKT": .1},
    "Marine, Aviation & Logistics": {"OFF-MRB": .45, "SB-BPN": .3, "SB-PLM": .15, "HO-JKT": .1},
    "Engineering, Construction & Maintenance": {"ONS-SKT": .35, "OFF-MRB": .35, "HO-JKT": .15, "SB-BPN": .1, "SB-PLM": .05},
    "Facilities & Site Support": {"ONS-SKT": .3, "OFF-MRB": .25, "SB-BPN": .15, "SB-PLM": .15, "HO-JKT": .15},
    "IT & Telecom": {"HO-JKT": .6, "ONS-SKT": .15, "OFF-MRB": .15, "SB-BPN": .1},
    "Corporate & G&A": {"HO-JKT": .8, "SB-BPN": .1, "SB-PLM": .1},
}
WELLS = {"ONS-SKT": [f"SKT-{i:02d}" for i in range(1, 49)], "OFF-MRB": [f"MRB-{c}{i}" for c in "ABC" for i in range(1, 15)]}
BUYERS = [f"BYR-{i:02d}" for i in range(1, 27)]
BUYER_L1 = {l1: random.sample(BUYERS, 4) for l1 in L1_DEPT}
VAGUE = ["material as per attached spec", "sesuai penawaran terlampir", "lihat lampiran", "item as per quotation", "pengadaan sesuai PR",
         "spare part as per list", "jasa sesuai kontrak", "misc items", "lain-lain", "as per scope of work"]

TYPO_P = 0.10


def typo(s):
    if len(s) < 6:
        return s
    i = random.randint(1, len(s) - 2)
    k = random.random()
    if k < 0.4:
        return s[:i] + s[i + 1:]
    if k < 0.75:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]


def describe(t, y, m, plant):
    tpl = random.choice(t["templates"])
    vals = {k: random.choice(v) for k, v in t["vars"].items()}
    vals["mon"] = fmt_month(y, m)
    vals["yr"] = str(y)
    vals["well"] = random.choice(WELLS.get(plant) or WELLS[random.choice(["ONS-SKT", "OFF-MRB"])])
    d = tpl.format(**vals)
    r = random.random()
    if r < 0.30:
        d = d.upper()
    elif r < 0.38:
        d = d.lower()
    if random.random() < TYPO_P:
        d = typo(d)
    if random.random() < 0.03:
        d = "URGENT - " + d
    if random.random() < 0.04:
        d = d + f" ex PR{random.randint(10000000, 10999999)}"
    if random.random() < 0.05:
        d = d.replace(" ", "  ", 1)
    if random.random() < 0.02:
        d = random.choice(VAGUE)
    return d


rows = []
truth = []
po_seq = []
line_count = 0
item_base = {}
mat_codes = {}
mcode = 20000000
while line_count < TARGET_LINES:
    i = rng.choice(len(TAXONOMY), p=po_w)
    t = TAXONOMY[i]
    mi = rng.choice(len(months), p=mw[t["l1"]])
    p = months[mi]
    day = int(rng.integers(1, p.days_in_month + 1))
    date = pd.Timestamp(p.year, p.month, day)
    if date.dayofweek >= 5:
        date -= pd.Timedelta(days=int(date.dayofweek - 4))
    pw = PLANT_W[t["l1"]]
    plant = random.choices(list(pw), weights=list(pw.values()))[0]
    pool, w = l3_pool[t["l3"]]
    vendor = pool[rng.choice(len(pool), p=w)]
    if vendor in alias_of and random.random() < 0.35:
        vendor = alias_of[vendor]
    n_lines = 1 + int(rng.geometric(1 / lpp[i])) - 1
    n_lines = int(min(max(n_lines, 1), 40))
    dept = random.choice(L1_DEPT[t["l1"]])
    buyer = random.choice(BUYER_L1[t["l1"]]) if random.random() < 0.85 else random.choice(BUYERS)
    contract = f"CTR-{p.year - random.choice([0, 0, 1, 2])}-{random.randint(1, 420):04d}" if (t["po_type"] == "Service" and random.random() < 0.7) or random.random() < 0.15 else None
    po_seq.append(date)
    po_idx = len(po_seq) - 1
    for ln in range(n_lines):
        tt = t
        if ln > 0 and random.random() < 0.10:
            tt = tax[random.choice(siblings[t["l1"]])]
        desc = describe(tt, p.year, p.month, plant)
        if tt["po_type"] == "Material":
            key = re.sub(r"\s+", " ", desc.upper()).strip()
            if key not in item_base:
                item_base[key] = float(rng.lognormal(np.log(tt["med"] / 5.5), tt["sigma"] * 0.85))
            q = max(1.0, float(np.round(rng.lognormal(1.45, 0.9))))
            usd = item_base[key] * q * inflation[p.year] * vend_price.get(vendor, 1.0) * float(rng.normal(1, 0.05))
        else:
            q = None
            usd = float(rng.lognormal(np.log(tt["med"]), tt["sigma"])) * inflation[p.year] * vend_price.get(vendor, 1.0)
        rows.append((po_idx, (ln + 1) * 10, date, plant, dept, buyer, vendor, contract, tt["po_type"], desc, usd, tt["l3"], tt["uoms"], q))
        line_count += 1

print("lines generated", line_count, "POs", len(po_seq))
df = pd.DataFrame(rows, columns=["po_idx", "po_line", "po_date", "plant", "requester_dept", "buyer_id", "vendor_id", "contract_no",
                                 "po_type", "item_description", "usd", "true_l3", "uoms", "q"])
df = df.iloc[:TARGET_LINES].copy()

# scale to target addressable spend
df["usd"] *= TARGET_SPEND_USD / df["usd"].sum()

# PO numbers in chronological order (SAP style)
order = np.argsort(np.array(po_seq, dtype="datetime64[ns]"), kind="stable")
po_num = np.empty(len(po_seq), dtype=np.int64)
po_num[order] = 4500100000 + np.arange(len(po_seq))
df["po_number"] = po_num[df["po_idx"].values].astype(str)

# quantity / uom / unit price / currency
n = len(df)
df["uom"] = [random.choice(u) for u in df["uoms"]]
qty = np.ones(n)
is_mat = (df["po_type"] == "Material").values
qty[is_mat] = df.loc[is_mat, "q"].values
day_uom = df["uom"].isin(["DAY"]).values
qty[day_uom] = rng.integers(10, 32, day_uom.sum())
pax = df["uom"].isin(["PAX", "MH", "HRS", "USER"]).values
qty[pax] = rng.integers(2, 60, pax.sum())
df["quantity"] = qty
foreign_v = df["vendor_id"].map(vend_foreign).values
usd_cur = np.where(foreign_v, rng.random(n) < 0.97, rng.random(n) < np.where(is_mat, 0.12, 0.38))
df["currency"] = np.where(usd_cur, "USD", "IDR")
df["ym"] = df["po_date"].dt.to_period("M").astype(str)
df["fx_rate_idr_usd"] = df["ym"].map(fx)
amt_local = np.where(usd_cur, np.round(df["usd"], 2), np.round(df["usd"] * df["fx_rate_idr_usd"], -2))
df["unit_price"] = np.where(usd_cur, np.round(amt_local / qty, 2), np.round(amt_local / qty, 0))
df["net_value"] = np.where(usd_cur, np.round(df["unit_price"] * qty, 2), df["unit_price"] * qty)
df["net_value_usd"] = np.where(usd_cur, df["net_value"], np.round(df["net_value"] / df["fx_rate_idr_usd"], 2))

# material codes for stock items (free-text lines have none)
codes = []
for desc, pt in zip(df["item_description"].values, df["po_type"].values):
    if pt != "Material" or random.random() < 0.22:
        codes.append(None)
        continue
    key = re.sub(r"\s+", " ", desc.upper()).strip()
    if key not in mat_codes:
        mcode += random.randint(1, 3)
        mat_codes[key] = str(mcode)
    codes.append(mat_codes[key])
df["material_code"] = codes

# buyer-assigned category (only partly filled, and noisy)
lab = np.where(df["material_code"].notna(), rng.random(n) < 0.62, rng.random(n) < 0.27)
manual = np.where(lab, df["true_l3"].values, None).astype(object)
noise = lab & (rng.random(n) < 0.065)
l1_of = {t["l3"]: t["l1"] for t in TAXONOMY}
for j in np.where(noise)[0]:
    true = df["true_l3"].iat[j]
    if random.random() < 0.6:
        manual[j] = random.choice([s for s in siblings[l1_of[true]] if s != true] or l3_names)
    else:
        manual[j] = random.choice(l3_names)
generic = lab & (rng.random(n) < 0.03)
manual[generic] = random.choices(["OTHERS", "Lain-lain", "General"], k=int(generic.sum()))
df["category_manual"] = manual

df["cost_center"] = df["plant"].str.replace("-", "") + "-" + df["requester_dept"].str[:3].str.upper()
gl_l1 = {"Well & Drilling Services": "5201300 - Well Services", "Corporate & G&A": "5301000 - G&A Services", "IT & Telecom": "5301000 - G&A Services"}
df["gl_account"] = [gl_l1.get(l1_of[l3]) if (pt == "Service" and l1_of[l3] in gl_l1) else
                    ("5201100 - Contract Services" if pt == "Service" else random.choice(GL["Material"]))
                    for l3, pt in zip(df["true_l3"].values, df["po_type"].values)]
df["delivery_date"] = df["po_date"] + pd.to_timedelta(np.where(is_mat, rng.integers(14, 120, n), rng.integers(1, 45, n)), unit="D")

# ---------------------------------------------------------------- realistic extract noise
out = df[["po_number", "po_line", "po_date", "delivery_date", "plant", "cost_center", "requester_dept", "buyer_id", "vendor_id",
          "contract_no", "po_type", "material_code", "item_description", "quantity", "uom", "unit_price", "currency",
          "fx_rate_idr_usd", "net_value", "net_value_usd", "gl_account", "category_manual"]].copy()
out["true_l3"] = df["true_l3"].values
out["po_date"] = out["po_date"].dt.strftime("%Y-%m-%d")
out["delivery_date"] = out["delivery_date"].dt.strftime("%Y-%m-%d")
# lower-case plant codes, Rp currency, a few blanks, zero-value lines, exact duplicates
k = rng.random(len(out))
out.loc[k < 0.004, "plant"] = out.loc[k < 0.004, "plant"].str.lower()
out.loc[(k >= 0.004) & (k < 0.006), "plant"] = None
k = rng.random(len(out))
out.loc[(k < 0.01) & (out["currency"] == "IDR"), "currency"] = "Rp"
k = rng.random(len(out))
zero = k < 0.0015
out.loc[zero, ["unit_price", "net_value", "net_value_usd"]] = 0.0
dupe = out.sample(frac=0.0035, random_state=SEED)
out = pd.concat([out, dupe]).sort_values(["po_number", "po_line"], kind="stable").reset_index(drop=True)
out.insert(0, "line_id", np.arange(1, len(out) + 1))

RAW.mkdir(parents=True, exist_ok=True)
HIDDEN.mkdir(parents=True, exist_ok=True)
out[["line_id", "true_l3"]].to_parquet(HIDDEN / "ground_truth.parquet", index=False)
out = out.drop(columns="true_l3")
out.to_csv(RAW / "po_lines_2021_2025.csv", index=False)

vend_out = vend_df[["vendor_id", "vendor_name", "tax_id", "country", "city", "vendor_type", "created_date"]].copy()
vend_out.to_csv(RAW / "vendor_master.csv", index=False)

tax_df = pd.DataFrame([{k: t[k] for k in ("l1", "l2", "l3", "unspsc", "po_type")} for t in TAXONOMY])
tax_df.columns = ["category_l1", "category_l2", "category_l3", "unspsc_code", "po_type"]
tax_df.to_csv(RAW / "category_taxonomy.csv", index=False)

pd.DataFrame({"month": list(fx), "idr_per_usd": list(fx.values())}).to_csv(RAW / "fx_rates_monthly.csv", index=False)

# QA sample: 2,000 lines reviewed manually by the category managers
gt = pd.read_parquet(HIDDEN / "ground_truth.parquet")
qa = gt.sample(2000, random_state=7).rename(columns={"true_l3": "category_l3_validated"})
qa["reviewed_by"] = rng.choice(["CM-Drilling", "CM-Ops", "CM-Services", "CM-Indirect"], len(qa))
qa.sort_values("line_id").to_csv(RAW / "qa_validated_sample.csv", index=False)

print(out.shape, "vendors", len(vend_out), "spend", round(out.net_value_usd.sum() / 1e9, 3), "B USD")
