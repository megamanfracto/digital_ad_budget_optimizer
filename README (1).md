# Digital Advertising Budget Optimizer

**Question:** where should a company spend its next Rs. 1,000,000 across Google, Instagram, Facebook, YouTube and TikTok?

This project simulates ad-campaign data, calculates the KPIs, builds a simple **Marketing Budget Recommendation Engine**, and compares **Pandas vs MapReduce vs Spark** on 10K → 100K → 1M → 10M records.

## What it does

1. Simulates ad data with 12 columns: date, campaign, platform, impressions, clicks, spend, conversions, revenue, age, gender, city, device
2. KPIs per platform: CTR, CPC, CPA, conversion rate, ROAS, revenue
3. Campaign performance (plus a campaign × platform ROAS grid)
4. Audience view: age, gender, city, device
5. Budget Recommendation Engine: each platform gets a share of the budget proportional to its historical ROAS
6. Big Data test: Pandas vs MapReduce (plain Python: map → shuffle → reduce) vs Spark

## Files

| File | What it is |
|---|---|
| `ad_budget_optimizer.ipynb` | The notebook (Colab or Jupyter) |
| `ad_budget_optimizer.py` | The same code as a plain script |
| `requirements.txt` | Python libraries |

## Run it

**Google Colab (easiest):** go to colab.research.google.com → File → Upload notebook → pick `ad_budget_optimizer.ipynb` → Runtime → Run all (about 5 minutes). The first cell installs Java and PySpark for you. Once this repo is on GitHub you can also use File → Open notebook → GitHub and paste the repo URL.

**Your own computer:**

```bash
pip install -r requirements.txt
python ad_budget_optimizer.py        # tables in the terminal, charts + CSVs in ./results/
```

- Spark needs **Java 17 or newer** (check with `java -version`; install from adoptium.net, or `sudo apt install openjdk-17-jre-headless` on Ubuntu). If Java or PySpark is missing, the script says so, skips Spark and still runs Pandas and MapReduce.
- The 10M-record step needs roughly 3–4 GB of free RAM. For a lighter run, remove `10_000_000` from `SIZES`.
- To use the notebook locally: `pip install jupyterlab`, then `jupyter lab`.
- On Windows, use WSL or Colab (Spark is most reliable there).

## Example result

Simulated data, 1M records (your numbers may differ slightly with other library versions):

| Platform | ROAS | Today's share of spend | Recommended share | Of the next Rs. 1M |
|---|---|---|---|---|
| Google | 4.78 | 22.1% | 28.9% | Rs. 288,512 |
| Instagram | 4.08 | 18.2% | 24.6% | Rs. 246,070 |
| Facebook | 3.46 | 17.7% | 20.9% | Rs. 208,716 |
| YouTube | 2.55 | 14.7% | 15.4% | Rs. 153,927 |
| TikTok | 1.70 | 27.4% | 10.3% | Rs. 102,776 |

Expected revenue: Rs. 3,248,103 with today's split vs Rs. 3,670,288 with the recommended split (+13.0%).

## Notes

- The data is **simulated** from assumptions at the top of the script (CTR, CPC, conversion rate per platform, and so on), so results show the method, not real-world truth. To use real data, replace `make_data(...)` with `pd.read_csv("your_file.csv")` using the same 12 columns; Steps 2 to 5 then work unchanged.
- **MapReduce** here is the programming model written in plain Python, not Hadoop.
- "Expected revenue" assumes each platform keeps its average ROAS when it gets more money. In real life returns shrink as spend grows, so re-run the engine after every campaign cycle.
