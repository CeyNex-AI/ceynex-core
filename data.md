# Ceynex — Data Sources Reference

A curated, link-rich catalogue of data sources for **Ceynex: A Multi-Agent Decision Intelligence Platform for Sri Lanka's National Export Economy** (CS3501 – Group 07). Sources are grouped by the agent / pipeline component they primarily feed, with access method, cost, and coverage notes.

*Compiled July 2026. Access terms and availability change — verify before relying on any single source.*

---

## Quick-reference table

| Source | Feeds | Access | Cost |
|---|---|---|---|
| UN Comtrade | Export Analytics, Forecast | REST API + bulk | Free (registration) |
| WITS (World Bank) | Trade Economics, Export Analytics | REST API + bulk | Free (registration) |
| FAOSTAT | Agriculture & Commodity | Bulk ZIP + API | Free |
| Central Bank of Sri Lanka | Trade Economics | Web / Data Library | Free |
| JAAF bulletins | Apparel & Manufacturing | Web / press (monthly) | Free |
| Sri Lanka EDB — EPI | Export Analytics (both sectors) | Free PDF / paid platform | Free PDF; Rs. 2,500–5,000 optional |
| Cinnamon price dataset (2016–2024) | Agriculture, Forecast benchmark | Via published paper | Free (paper) |
| Forbes & Walker / Tea Board | Agriculture & Commodity | Web (weekly) | Free |
| Colombo Tea Price (World Bank) | Agriculture & Commodity | Pink Sheet CSV | Free |

---

## 1. Core bilateral trade flows

### UN Comtrade
Primary source for bilateral trade flow data by HS code and partner country — Sri Lanka's tea (HS 0902), cinnamon (HS 0906), and apparel (HS 61 knit / HS 62 woven) exports by destination and year.

- Main database: https://comtradeplus.un.org/
- Developer portal (API keys): https://comtradedeveloper.un.org/
- Account creation guide: https://uncomtrade.org/docs/how-to-create-an-account/
- API base endpoint: `https://comtradeapi.un.org/data/v1/get/{typeCode}/{freqCode}/{clCode}`
- Official Python package (`comtradeapicall`): https://github.com/uncomtrade/comtradeapicall
- R package (`comtradr`) docs: https://docs.ropensci.org/comtradr/

Access notes:
- Free registration → **500 API calls/day**, up to **100,000 records per call**.
- Free public *preview* endpoint (no key) → capped at **500 records per request**.
- Uses **UN M49 numeric country codes** (Sri Lanka = 144), not ISO alpha-3.
- Monthly queries limited to a single year per call; annual queries up to 12 years.
- Country groups reference: `https://comtradeapi.un.org/files/v1/app/reference/country_groups.json`

### World Integrated Trade Solution (WITS) — World Bank
World Bank/UNCTAD-hosted trade, tariff, and non-tariff data. Validates Comtrade and enables cross-country benchmarking (Bangladesh, Vietnam, Cambodia for apparel; Kenya, India, Indonesia for tea).

- Portal: https://wits.worldbank.org/
- Sri Lanka country snapshot: https://wits.worldbank.org/CountrySnapshot/en/LKA
- API introduction & user guide: https://wits.worldbank.org/witsapiintro.aspx?lang=en
- Bulk data download: https://wits.worldbank.org/datadownload.aspx?lang=en
- About / data sources: https://wits.worldbank.org/about_wits.html
- Python package (`world_trade_data`): https://github.com/mwouts/world_trade_data
- Data360 (beta) dataset page: https://data360.worldbank.org/en/dataset/WB_WITS

Access notes:
- Registration + login required for custom queries; TradeStats summaries browsable without login.
- Preferential and MFN tariff rates harmonized at HS 6-digit level (TRAINS/UNCTAD + WTO IDB/CTS).
- Includes the **Global Preferential Trade Agreement Database (GPTAD)** — relevant for the knowledge graph's GSP+ / FTA edges.
- Sri Lanka applied weighted-mean tariff indicator: https://data.worldbank.org/indicator/TM.TAX.MRCH.WM.AR.ZS?locations=LK

---

## 2. Agriculture & Commodity Agent

### FAOSTAT (FAO)
Free food & agriculture statistics for 245+ countries, 1961–present — production volumes, trade, and producer prices for tea, coconut, rubber, and cinnamon/spices.

- Portal: https://www.fao.org/faostat/
- Data explorer: https://www.fao.org/faostat/en/#data
- Bulk download base URL: `https://fenixservices.fao.org/faostat/static/bulkdownloads`
  - Crop production: `crop_production_E_All_Data_(Normalized).zip` (dataset code **QCL**)
  - Forestry (natural rubber): `Forestry_E_All_Data_(Normalized).zip`
- R package (`FAOSTAT`) on CRAN: https://cran.r-project.org/web/packages/FAOSTAT/FAOSTAT.pdf
- OWID FAOSTAT documentation (domain codes / structure): https://docs.owid.io/projects/etl/data/faostat/

Access notes:
- Fully free; no key required for bulk ZIPs. Long ("normalized") format is best for analysis.
- Key domains: **QCL** (production), **TCL/TM** (trade), **PP** (producer prices), **EF** (fertilizers).
- Export values reported FOB in thousand USD; imports CIF.

### Cinnamon purchasing-price dataset (2016–2024)
Historical domestic cinnamon purchase-price records for southern Sri Lanka, compiled and used in a published hybrid-forecasting study ("Stacked Boost Forest"). Provides both raw data and a **directly comparable forecasting benchmark** (reported 96% accuracy highest-price, 98% average-price).

- Related paper — "Factors Affecting Sri Lankan Cinnamon Export Income": https://www.researchgate.net/publication/389078215_Factors_Affecting_Sri_Lankan_Cinnamon_Export_Income
- Data spans 2016–2024; contact authors or extract from publication for the raw series.

### Sri Lanka Tea Board / Colombo Tea Auction
Weekly auction prices and quantities by elevation (High/Mid/Low grown, BOP/BOPF grades) — the highest-frequency domestic price signal for tea.

- Forbes & Walker — market reports: https://web.forbestea.com/market-reports
- Forbes & Walker — Sri Lankan statistics: https://web.forbestea.com/statistics/sri-lankan-statistics/65-sri-lanka-tea-production
- Weekly auction quantities & averages (2022 example): https://web.forbestea.com/statistics/sri-lankan-statistics/90-weekly-tea-auction-quantities-averages
- Tea Exporters Association Sri Lanka — market reports: https://teasrilanka.org/market-reports

### Colombo Tea Price — World Bank "Pink Sheet"
Clean, monthly, machine-readable benchmark tea price (Colombo auction), part of World Bank Commodity Price Data. Good for a validated, gap-free monthly series.

- YCharts mirror (quick view): https://ycharts.com/indicators/colombo_tea_price
- Underlying source: World Bank Commodity Markets "Pink Sheet" — https://www.worldbank.org/en/research/commodity-markets

---

## 3. Apparel & Manufacturing Agent

### Joint Apparel Association Forum (JAAF)
Sri Lanka's apparel industry body; publishes monthly apparel & textile export values by destination (US, EU, UK, "other"). **Primary open source** for the Apparel Agent.

- JAAF / Sri Lanka Apparel site: https://www.srilankaapparel.com/
- JAAF LinkedIn (monthly figures often posted first): https://lk.linkedin.com/company/jaaf-srilanka

Where the monthly numbers are reported (JAAF data, republished):
- Daily FT (Sri Lanka): https://www.ft.lk/ — search "apparel exports JAAF"
- Just-Style: https://www.just-style.com/news/jaaf-sri-lanka-apparel-export/
- Knitting Industry: https://knittingindustry.com/

Coverage note: JAAF releases are monthly, value-based (USD), split by three core markets plus "other." Full-year 2025 apparel & textile exports ≈ USD 5.02 bn. Also relevant to the Trade Economics Agent (the ongoing US Section 301 / tariff situation directly affects US-bound flows).

---

## 4. Trade Economics Agent

### Central Bank of Sri Lanka (CBSL) — exchange rates & Economic Data Library
Freely accessible historical USD/LKR daily and monthly series, plus the searchable Economic Data Library.

- Economic Data Library (main): https://www.cbsl.gov.lk/en/statistics/data/economic-data-library
- Economic Data Library (eResearch portal): https://www.cbsl.lk/eresearch/
- Exchange rates hub: https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates
- Daily indicative USD/LKR spot rate: https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates/daily-indicative-usd-spot-exchange-rates
- Daily indicative rates (all currencies): https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates/daily-indicative-exchange-rates
- USD/LKR indicative rate chart: https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates/usd-lkr-Indicative-rate-chart
- Statistics landing page: https://www.cbsl.gov.lk/en/statistics
- Economic & statistical charts: https://www.cbsl.gov.lk/en/economic-and-statistical-charts

Note: The middle-rate USD/LKR series was discontinued from 07.03.2023; use the **indicative spot rate** for continuity thereafter.

### Tariffs, trade agreements & GSP+
- WITS tariff data & GPTAD — see Section 1 (preferential + MFN rates at HS 6-digit; PTA/FTA browser).
- Feeds the Trade Economics Agent's simulation of how exchange-rate movements, tariffs, and agreements (GSP+, FTAs) affect net export revenue per sector.

---

## 5. National export performance (both sectors)

### Sri Lanka Export Development Board (EDB) — Export Performance Indicators (EPI)
Free annual EPI publications: 5–10 year historical series by product, sector, and region across agriculture and apparel. Compiled with Sri Lanka Customs, Department of Census & Statistics, and CBSL.

- Trade statistics landing: https://www.srilankabusiness.com/edb/trade-statistics.html
- Export statistics (exporters): https://www.srilankabusiness.com/exporters/export-statistics.html
- Export performance report: https://www.srilankabusiness.com/exporters/export-performance-report.html
- EPI 2024 (free PDF, latest — 38th volume): https://www.srilankabusiness.com/ebooks/export-performance-indicators-of-sri-lanka-2024.pdf
- EPI 2022 preview PDF: https://www.srilankabusiness.com/ebooks/preview-export-performance-indicators-of-sri-lanka-2022.pdf
- EDB e-service books (full versions): https://exporter.edb.gov.lk/eservice/books/19
- Historical EPI 2004–2014 archive: https://stat.edb.gov.lk/epi2015/

Access notes:
- The flagship **EPI annual PDF is free** (free/preview version; some editions have a paid full 308-page version at Rs. 5,000).
- The EDB's **live online trade-statistics platform is subscription-based (Rs. 2,500/year)** — *optional*, per the proposal.
- Values in USD, arranged by region and by product/product group.

### Department of Census & Statistics (upstream provider)
Underlies EDB and CBSL trade figures; worth citing as the ultimate national source. Referenced throughout EDB EPI publications.

---

## 6. Forecasting benchmarks (published, for the Forecast Agent evaluation)

Sri Lanka-specific published studies for benchmarking forecast accuracy (MAPE/RMSE); each also identifies usable data.

### Cinnamon — Liyanage & Silva (2025), "Stacked Boost Forest"
Hybrid ensemble (Random Forest + Gradient Boosting + stacked Linear Regression meta-model) on 2016–2024 domestic cinnamon purchasing prices, southern Sri Lanka.
- ResearchGate record: https://www.researchgate.net/publication/389078215_Factors_Affecting_Sri_Lankan_Cinnamon_Export_Income

### Tea — Mampitiya et al. (2025), Explainable AI for Ceylon Tea yield
First AI-based Ceylon Tea yield model; CatBoost + SHAP on meteorological, soil, and fertilizer parameters (Dampahala Tea Company data). Open access, Elsevier *Smart Agricultural Technology*.
- ScienceDirect (open access): https://www.sciencedirect.com/science/article/pii/S2772375525002321
- ResearchGate mirror: https://www.researchgate.net/publication/391670481_Explainable_artificial_intelligence_to_estimate_the_Sri_Lankan_Ceylon_Tea_crop_yield

---

## Data-integration notes

- **Join keys:** Standardize on **HS codes** (6-digit) for products and **UN M49 / ISO-3** for countries. Comtrade uses M49; WITS and most others accept ISO-3 — build a crosswalk early.
- **Currency:** Most trade values are USD; CBSL is the authority for USD/LKR conversion.
- **Frequency mismatch:** JAAF and CBSL are monthly/daily; Comtrade, FAOSTAT, and EDB EPI are largely annual (Comtrade also monthly). Resample/align for the unified time-indexed dataset.
- **Validation strategy:** Cross-check Comtrade against WITS and EDB EPI for the same product-year before back-testing (reported vs. mirror vs. gap-filled figures diverge).
- **Free-tier sufficiency:** Every source has a free access path adequate for a course project. The only paid items (EDB live platform Rs. 2,500/yr; EPI full PDF Rs. 5,000) are optional.
