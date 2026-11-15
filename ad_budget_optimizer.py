"""Digital Advertising Budget Optimizer  (script version of ad_budget_optimizer.ipynb)

Question: where should we spend the next Rs. 1 million across Google, Instagram, Facebook, YouTube and TikTok?

Run:    python ad_budget_optimizer.py
Needs:  pip install -r requirements.txt   (plus Java 17+ for the Spark part; without it Spark is skipped)
Output: tables in the terminal, charts and CSV files in ./results/
"""
import os, time, tempfile
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw charts into files instead of pop-up windows
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
pd.options.display.float_format = "{:,.2f}".format
pd.options.display.width = 200
pd.options.display.max_columns = 30

def display(x):                            # print a table in the terminal
    print(x.to_string() if hasattr(x, "to_string") else x, "\n")

def save_fig(name):                        # instead of showing a chart, save it as a PNG
    plt.savefig(f"results/{name}.png", dpi=120, bbox_inches="tight")
    plt.close("all")

# ======================================================================
# Step 1 · Simulate the ad data
# ======================================================================
# One record = one campaign, on one platform, for one day and one audience segment.
# The platform numbers below are assumptions (edit them to change the story).

BUDGET = 1_000_000                       # the Rs. 1 million to allocate
METRICS = ["impressions", "clicks", "spend", "conversions", "revenue"]   # columns we add up
AOV = 2800                               # average order value (Rs)

# ---- assumptions (edit freely) ----
PLATFORMS = ["Google", "Instagram", "Facebook", "YouTube", "TikTok"]
#                 impressions/record, CTR,   CPC (Rs), conversion rate
PROFILE = np.array([[ 500, 0.030, 30, 0.0450],     # Google
                    [1500, 0.010, 25, 0.0320],     # Instagram
                    [1500, 0.011, 22, 0.0240],     # Facebook
                    [2500, 0.006, 20, 0.0160],     # YouTube
                    [5000, 0.008, 14, 0.0075]])    # TikTok
IMPR, CTR, CPC, CVR = PROFILE.T

# category: (share of records, effect). Effect = conversion multiplier (for city: order-value multiplier)
CAMPAIGN = {"Retargeting": (.125, 1.60), "Flash Sale": (.125, 1.30), "Loyalty Rewards": (.125, 1.25),
            "Free Delivery": (.125, 1.15), "Festive Offer": (.125, 1.00), "New Customer Offer": (.125, .90),
            "Product Launch": (.125, .80), "Brand Awareness": (.125, .50)}
AGE     = {"18-24": (.25, .80), "25-34": (.35, 1.20), "35-44": (.22, 1.10), "45-54": (.12, .90), "55+": (.06, .70)}
GENDER  = {"female": (.50, 1.0), "male": (.50, 1.0)}
DEVICE  = {"mobile": (.70, .95), "desktop": (.22, 1.25), "tablet": (.08, 1.00)}
CITY    = {"Karachi": (.28, 1.00), "Lahore": (.25, 1.05), "Islamabad": (.15, 1.20),
           "Rawalpindi": (.10, .95), "Faisalabad": (.12, .90), "Peshawar": (.10, .90)}

def draw(rng, n, spec):
    """Pick a category for each of n records -> (category column, effect of each record)."""
    names, (share, effect) = list(spec), np.array(list(spec.values())).T
    codes = rng.choice(len(names), n, p=share).astype(np.int8)
    return pd.Categorical.from_codes(codes, names), effect[codes]

def make_data(n, seed=42):
    rng = np.random.default_rng(seed)
    code = rng.integers(0, len(PLATFORMS), n, dtype=np.int8)            # platform of each record
    campaign, c_eff = draw(rng, n, CAMPAIGN)
    age,      a_eff = draw(rng, n, AGE)
    gender,   _     = draw(rng, n, GENDER)
    device,   d_eff = draw(rng, n, DEVICE)
    city,  city_eff = draw(rng, n, CITY)

    impressions = (IMPR[code] * rng.lognormal(0, 0.5, n)).astype(np.int64)
    clicks      = rng.binomial(impressions, CTR[code])
    spend       = clicks * CPC[code] * rng.lognormal(0, 0.15, n)
    conversions = rng.binomial(clicks, CVR[code] * c_eff * a_eff * d_eff)
    revenue     = conversions * AOV * city_eff * rng.lognormal(0, 0.25, n)

    return pd.DataFrame({
        "date": np.datetime64("2025-01-01") + rng.integers(0, 365, n).astype("timedelta64[D]"),
        "campaign": campaign, "platform": pd.Categorical.from_codes(code, PLATFORMS),
        "impressions": impressions, "clicks": clicks, "spend": spend.round(2),
        "conversions": conversions, "revenue": revenue.round(2),
        "age": age, "gender": gender, "city": city, "device": device})

df = make_data(1_000_000)                # 1 million records for the main analysis
print(f"{len(df):,} records, {df.memory_usage().sum() / 1e6:.0f} MB")
print(df.head().to_string(), "\n")

# ======================================================================
# Step 2 · KPIs per platform
# ======================================================================
# CTR = clicks ÷ impressions · CPC = spend ÷ clicks · CPA = spend ÷ conversions · Conversion rate = conversions ÷ clicks · ROAS = revenue ÷ spend
#
# Always add up first, then divide (ratio of sums), never average the per-row ratios.

def add_metrics(t):
    """t = totals per group -> adds the KPIs."""
    t = t.copy()
    t["CTR %"] = 100 * t.clicks / t.impressions        # click-through rate
    t["CPC"]   = t.spend / t.clicks                     # cost per click (Rs)
    t["CPA"]   = t.spend / t.conversions                # cost per acquisition (Rs), a.k.a. CAC
    t["CVR %"] = 100 * t.conversions / t.clicks         # conversion rate
    t["ROAS"]  = t.revenue / t.spend                    # revenue earned per Rs 1 spent
    return t

def kpis(by):
    """KPI table for any grouping: kpis("platform"), kpis("city"), kpis(["campaign", "platform"]) ..."""
    return add_metrics(df.groupby(by, observed=True)[METRICS].sum())

by_platform = kpis("platform").sort_values("ROAS", ascending=False)
display(by_platform)
by_platform.to_csv("results/platform_kpis.csv")

view = by_platform.assign(**{"Spend (Rs M)": by_platform.spend / 1e6, "Revenue (Rs M)": by_platform.revenue / 1e6})
view[["Spend (Rs M)", "Revenue (Rs M)", "CTR %", "CPC", "CPA", "CVR %", "ROAS"]].plot.bar(
    subplots=True, layout=(2, 4), sharex=False, figsize=(14, 6), legend=False, rot=45)
plt.tight_layout(); save_fig("01_platform_kpis")

# ======================================================================
# Step 3 · Campaign performance
# ======================================================================

by_campaign = kpis("campaign").sort_values("ROAS", ascending=False)
display(by_campaign[["spend", "revenue", "CTR %", "CPC", "CPA", "CVR %", "ROAS"]])
by_campaign.to_csv("results/campaign_kpis.csv")

# ROAS of every campaign on every platform (green = good, red = bad)
roas_grid = (df.groupby(["campaign", "platform"], observed=True)[["spend", "revenue"]].sum()
               .eval("revenue / spend").unstack())
display(roas_grid.loc[by_campaign.index].round(1))

# ======================================================================
# Step 4 · Audience view (age, gender, city, device)
# ======================================================================

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
for ax, col in zip(axes, ["device", "age", "gender", "city"]):
    kpis(col).ROAS.plot.bar(ax=ax, title=f"ROAS by {col}", rot=45, xlabel="")
axes[0].set_ylabel("ROAS")
plt.tight_layout(); save_fig("02_audience_roas")

# ======================================================================
# Step 5 · Marketing Budget Recommendation Engine
# ======================================================================
# Rule: each platform gets a share of the budget proportional to its historical ROAS.
# Better platforms get more, but nobody gets 0% or 100%, because returns shrink when you push more money into one platform.

def recommend_budget(data, budget=BUDGET):
    """Budget Recommendation Engine: share of the budget = platform ROAS / sum of all ROAS."""
    p = add_metrics(data.groupby("platform", observed=True)[METRICS].sum())
    out = pd.DataFrame({"ROAS": p.ROAS,
                        "current spend %": 100 * p.spend / p.spend.sum(),
                        "recommended %": 100 * p.ROAS / p.ROAS.sum()})
    out["budget (Rs)"] = budget * out["recommended %"] / 100
    out["expected revenue (Rs)"] = out["budget (Rs)"] * out.ROAS
    return out.sort_values("recommended %", ascending=False)

rec = recommend_budget(df)
display(rec)
rec.to_csv("results/recommendation.csv")

print(f"\nWhere should we spend the next Rs. {BUDGET:,}?")
for name, r in rec.iterrows():
    print(f"  {name:<10} {r['recommended %']:5.1f}%    Rs. {r['budget (Rs)']:>9,.0f}")

keep_today = BUDGET * (rec["current spend %"] / 100 * rec.ROAS).sum()      # revenue if we keep today's split
use_engine = rec["expected revenue (Rs)"].sum()
print(f"\nExpected revenue, today's split: Rs. {keep_today:,.0f}   |   recommended split: Rs. {use_engine:,.0f}  ({use_engine / keep_today - 1:+.1%})")

rec[["current spend %", "recommended %"]].plot.bar(rot=0, figsize=(8, 4), title="Share of budget: today vs recommended")
plt.ylabel("%"); save_fig("03_recommendation")

# Limits: the data is simulated from the assumptions in Step 1, so this shows the *method*, not real-world truth.
# To use real data, replace df = make_data(...) with pd.read_csv("your_file.csv") (same 12 columns); Steps 2 to 5 then work unchanged.
# "Expected revenue" assumes each platform keeps its average ROAS when it gets more money. In real life returns shrink as spend grows, so re-run the engine after every campaign cycle.

# ======================================================================
# Step 6 · Big Data test: Pandas vs MapReduce vs Spark
# ======================================================================
# Same job for all three: load the file and add up impressions, clicks, spend, conversions and revenue per platform.
# * Pandas: one computer, everything in memory.
# * MapReduce: the classic model, written in plain Python (real Hadoop is not practical in Colab). *Map*: every chunk of rows becomes partial sums per platform → *Shuffle*: group the partial sums by platform → *Reduce*: add them up.
# * Spark: a distributed engine, run here on all CPU cores of this machine.

try:
    from pyspark.sql import SparkSession, functions as F
    spark = (SparkSession.builder.master("local[*]").appName("ads")
             .config("spark.driver.memory", "4g").config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
except Exception as e:           # no Java 17+ or no PySpark on this computer -> carry on with Pandas and MapReduce only
    spark = None
    print(f"Spark is not available ({type(e).__name__}), so it is skipped. Fix: install Java 17+ and `pip install pyspark`.")

# ---- 1) Pandas ----
def run_pandas(path):
    return pd.read_parquet(path).groupby("platform", observed=True)[METRICS].sum()

# ---- 2) MapReduce (plain Python) ----
def mapper(chunk):
    """MAP (+ local combine): one chunk of rows -> {platform: [partial sums]}"""
    out = defaultdict(lambda: [0, 0, 0.0, 0, 0.0])
    for p, i, c, s, v, r in zip(*(chunk[col].tolist() for col in ["platform"] + METRICS)):
        a = out[p]
        a[0] += i; a[1] += c; a[2] += s; a[3] += v; a[4] += r
    return out

def reducer(values):
    """REDUCE: add up all partial sums of one platform."""
    return [sum(col) for col in zip(*values)]

def run_mapreduce(path, chunk_rows=500_000):
    data = pd.read_parquet(path)
    groups = defaultdict(list)                                       # SHUFFLE: group partial sums by platform
    for start in range(0, len(data), chunk_rows):
        for key, partial in mapper(data.iloc[start:start + chunk_rows]).items():
            groups[key].append(partial)
    return pd.DataFrame({k: reducer(v) for k, v in groups.items()}, index=METRICS).T

# ---- 3) Spark ----
def run_spark(path):
    rows = (spark.read.parquet(path).groupBy("platform")
                 .agg(*[F.sum(c).alias(c) for c in METRICS]).collect())
    return pd.DataFrame([r.asDict() for r in rows]).set_index("platform")[METRICS]

ENGINES = {"Pandas": run_pandas, "MapReduce": run_mapreduce}
if spark:
    ENGINES["Spark"] = run_spark

SIZES = [10_000, 100_000, 1_000_000, 10_000_000]       # delete 10_000_000 if the computer is slow or runs out of memory

def timed(fn, *args):
    t = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t

def same(a, b):                                          # do two engines give the same totals?
    return np.allclose(a.loc[PLATFORMS, METRICS].to_numpy(float), b.loc[PLATFORMS, METRICS].to_numpy(float))

rows = []
for n in SIZES:
    path = os.path.join(tempfile.gettempdir(), f"ads_{n}.parquet")
    make_data(n)[["platform"] + METRICS].to_parquet(path)       # the same file for every engine
    if spark and n == SIZES[0]:
        run_spark(path)                                          # warm-up: the first Spark job pays one-time start-up costs
    results, row = {}, {"records": n}
    for name, run in ENGINES.items():
        results[name], row[name] = timed(run, path)
    assert all(same(results["Pandas"], r) for r in results.values()), "the engines disagree!"
    rows.append(row)
    os.remove(path)
    print(f"{n:>12,} records done")

bench = pd.DataFrame(rows).set_index("records")
print("\nSeconds to load + aggregate (all engines gave identical totals):")
print(bench.to_string(float_format="{:.3f}".format))
bench.to_csv("results/benchmark_seconds.csv")

bench.plot(marker="o", logx=True, logy=True, figsize=(7, 4), title="Seconds to load + aggregate (log scale)")
plt.ylabel("seconds"); save_fig("04_benchmark")

# How to read the chart: Pandas is usually the fastest on one computer. Plain-Python MapReduce slows down roughly in line with the number of rows.
# Spark has a fixed start-up cost (so it looks slow on small data), but it is the one that keeps scaling when the data no longer fits on a single computer.
