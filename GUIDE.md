# ChurnGuard — Tester's Guide

Last updated 26 Nov 2026

This guide takes you from a downloaded zip file to a full review of the ChurnGuard project in about 45 minutes. You need no machine-learning knowledge: every term is explained, every step shows the exact command and what you should see.

## 1. What this project is

**Churn** means customers leaving. When a bank customer closes their account and moves to another bank, they have "churned". Banks lose money every time this happens, and winning a new customer usually costs more than keeping an existing one.

**ChurnGuard predicts which customers are likely to leave**, so a bank's retention team knows whom to call first, for example to offer a better rate or fix a complaint before the customer goes.

How it works, in three steps:

1. **It learns from history.** The project uses a public dataset of 10,000 past bank customers from France, Germany and Spain. For each one we know their age, balance, number of products, whether they were active, and, most importantly, whether they left (20.4% did).
2. **It finds patterns.** A machine-learning model studies 8,000 of those customers and learns which combinations of details tend to come before leaving. For example, customers aged 50–59 left 56% of the time, and customers with 3–4 products almost always left.
3. **It scores new customers.** Given a customer's details, it gives a probability of leaving (for example 95.8%) and a risk level: Low, Medium or High. It was tested on the other 2,000 customers, which it never saw while learning.

**What you will review:**

- A **data pipeline** (`main.py`) that cleans the data, draws charts, trains and compares 9 model versions, and saves the best one.
- An **interactive web app** (ChurnGuard) with 6 pages, where you can type in a customer and get a prediction.
- **47 automated tests**, a Jupyter **notebook** and **documentation**.

**Important:** this is a university project (CUSIT, Programming for AI, Fall 2026) by Talha Nadeem (16131), Muhammad Moaz (15966) and Syed Abdullah Bin Masood (16048), trained on a public dataset. It is not a real banking system and must never be used to make real decisions about people.

## 2. Words you will see

Keep this table open while testing; the app and the outputs use these terms.

| Term | Plain meaning | Example in this project |
| --- | --- | --- |
| Churn / churner | A customer who leaves; the person who left | 2,037 of 10,000 customers churned |
| Dataset | The table of past customers the model learns from | `Churn_Modelling.csv`, 10,000 rows |
| Feature | One piece of information about a customer | Age, Balance, Country, Number of products |
| Target | The thing being predicted | `Exited`: 1 = left, 0 = stayed |
| Model | A program that learned patterns from the data | A Random Forest (many decision trees voting) |
| Training set / test set | Customers used to learn / customers held back to check the model honestly | 8,000 / 2,000 customers |
| Probability | How likely the model thinks leaving is, 0–100% | 95.8% |
| Threshold | The cut-off above which a customer is flagged as "likely to churn" | 0.45 (45%) |
| Risk level | Low (below 30%), Medium (30% up to the threshold), High (at or above it) | High = flagged |
| Accuracy | Share of all predictions that were right | 83.2% |
| Recall | Share of real churners the model caught; the most important number here | 68.1% caught |
| Precision | Of the customers flagged, the share who really left | 57.2% |
| F1 score | One number balancing recall and precision (0–1, higher is better) | 0.622 |
| ROC-AUC | How well the model ranks churners above non-churners (0.5 = guessing, 1 = perfect) | 0.859 |
| False positive / false alarm | Flagged, but actually stayed | 207 customers |
| False negative / missed churner | Not flagged, but actually left | 130 customers |
| Confusion matrix | A 2×2 box counting correct and wrong predictions of each kind | Model performance page |
| Baseline | The simple first model everything else is compared with | Logistic Regression |
| Experiment | One documented attempt to improve the model | E1 to E5, 9 runs |
| Fairness | Whether the model makes more mistakes for some groups (e.g. a country or gender) | Fairness page |

Why recall matters most: missing a customer who leaves costs the bank that customer, while a false alarm only costs one phone call. Accuracy alone misleads, because predicting "stays" for everyone is already 79.6% accurate.

## 3. What you need before starting

| Item | Requirement | How to check |
| --- | --- | --- |
| Computer | Windows 10/11, macOS or Linux; 4 GB RAM; 1 GB free disk space | — |
| Python | Version 3.9 or newer (tested on 3.9 and 3.13) | Run `python --version` (macOS/Linux: `python3 --version`) |
| Internet | Needed once, to install packages, and for the app's fonts | — |
| Web browser | Chrome, Edge, Firefox or Safari | — |
| Time | About 45 minutes for a full review | — |
| The project | `churn-prediction.zip` | — |

No Python? Download it from [python.org/downloads](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"** on the first install screen.

**Opening a terminal** (the window where you type commands):

- **Windows:** press the Start key, type `cmd`, press Enter. (PowerShell also works.)
- **macOS:** press Cmd + Space, type `Terminal`, press Enter.
- **Linux:** press Ctrl + Alt + T.

In this guide, a grey `code` line is a command: type it in the terminal exactly as shown and press Enter. Where Windows and macOS/Linux differ, both are given.

## 4. Set up and run the project

### Step 1 — Unzip and open the folder (2 minutes)

1. Right-click `churn-prediction.zip` → **Extract All** (Windows) or double-click it (macOS).
2. In the terminal, move into the folder. Replace the path with where you extracted it:
   - Windows: `cd C:\Users\YourName\Downloads\churn-prediction`
   - macOS/Linux: `cd ~/Downloads/churn-prediction`
3. Check you are in the right place: `dir` (Windows) or `ls` (macOS/Linux) should list `main.py`, `app.py`, `README.md` and the folders `src`, `data`, `tests`.

### Step 2 — Install the required packages (3–5 minutes)

A *virtual environment* is a private box for this project's packages, so nothing else on your computer is affected.

1. Create it:
   - Windows: `python -m venv .venv`
   - macOS/Linux: `python3 -m venv .venv`
2. Switch it on (do this again every time you open a new terminal):
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
   - You should now see `(.venv)` at the start of the line.
3. Install the packages: `pip install -r requirements.txt`

**Expected:** several screens of downloads ending with `Successfully installed …`. Warnings in yellow are fine; red `ERROR` lines are not (see Troubleshooting).

### Step 3 — Train the model with the pipeline (4–5 minutes)

Run: `python main.py`

This single command does the whole machine-learning job: loads and checks the data, cleans it, draws 7 analysis charts, splits customers 80/20 into training and test sets, runs 9 model versions across 5 experiments, picks the best, tests it on the 2,000 held-back customers, checks fairness and saves everything.

**Expected output:** progress lines from `STEP 1/8` to `STEP 8/8`, including lines like:

```
E1   Logistic Regression          CV F1=0.315  test F1=0.284  recall=0.187
E3a  Random Forest                CV F1=0.604  test F1=0.596  recall=0.582
...
Done in 228 s. See outputs/results/run_summary.md
```

Numbers can differ slightly on your computer if your package versions differ from ours (Python 3.13, scikit-learn 1.9); that is normal. Want a faster check? `python main.py --quick` takes about 1 minute with a smaller search.

**What it created (open these in any file browser):**

| Folder or file | What it holds |
| --- | --- |
| `outputs/results/run_summary.md` | One-page summary of every result — the best single file to read |
| `outputs/figures/` | 15 charts (PNG) |
| `outputs/results/` | Data audit, chart explanations, experiment table, fairness, success criteria |
| `models/final_model.joblib` | The trained model |
| `data/processed/churn_cleaned.csv` | Cleaned data with names and IDs removed |

### Step 3b — Run the automated tests (1 minute)

Run: `python -m pytest -v`

**Expected:** a list of test names each marked `PASSED`, ending with `47 passed`. These tests deliberately feed the code bad input (a missing file, an empty file, an age of 150, an unknown country, text where a number belongs, and more) and check that each is caught with a clear message. A plain-language list of every test is in `docs/testing_log.md`.

## 5. Test the ChurnGuard app, page by page

Run: `streamlit run app.py`

Your browser opens at **http://localhost:8501**. If it does not, copy that address into the browser. Leave the terminal open: closing it stops the app (Ctrl + C also stops it).

The dark **sidebar** on the left is the menu. Its top card shows the model status ("Model ready", recall ring), then six pages in three groups (Analyse, Predict, Govern), the overall churn rate, the risk-level colours and the dataset name. Open each page in order and tick each check as you go.

If you see **"Set up ChurnGuard"** instead of the dashboard, the model has not been trained on this computer yet: click **Train the model now** (about 2–3 minutes) or go back to Step 3.

### Page 1 — Overview (Analyse)

The one-screen summary for a bank manager.

- [ ] Four number cards show 10,000 customers, 20.4% churn rate, recall about 68% and ROC-AUC about 0.86, the last two marked **"met"** in green.
- [ ] The bar chart shows customers with 2 products churn least (7.6%), and 3–4 products churn most (82.7% and 100%).
- [ ] "Test-set outcomes" adds up to 2,000 customers.
- [ ] "Needs attention" warns about a fairness gap for German customers.

### Page 2 — Data insights (Analyse)

What the data looks like and how it was checked.

- [ ] "Dataset audit" reports 0 missing values and 0 duplicates.
- [ ] "What drives the model" ranks Number of products and Age at the top.
- [ ] Open **"All exploratory charts"** at the bottom: 7 charts appear.

### Page 3 — Predict a customer (Predict) — the main feature

Change the form on the left, click **Predict churn risk**, and read the result on the right: a gauge with the probability, a risk label, a suggested action, and a **what-if** table showing how the risk would change if one thing about the customer changed.

Try these customers. Leave every field not listed at its default (credit score 650, tenure 5, balance 100,000, salary 100,000, has credit card ticked).

| # | Enter | Expected result |
| --- | --- | --- |
| A | Leave the defaults: Germany, Female, age 55, 3 products, Active member unticked | **95.8% · High risk** · flagged |
| B | As A, but 2 products and tick Active member | **57.5% · High** (risk falls by about 38 points) |
| C | France, Male, age 30, 2 products, Active member ticked | **8.2% · Low** |
| D | Spain, Male, age 34, 1 product, balance 0, Active member unticked | **37.7% · Medium** |
| E | France, Female, age 45, 1 product, Active member unticked | **57.5% · High** |
| F | As E, but tick Active member | **38.1% · Medium** |

Percentages may differ by a few points if your package versions differ; the risk levels should match.

- [ ] Each result matches the table (within a few points).
- [ ] The what-if table changes when you change the customer.
- [ ] The risk label colour matches the sidebar legend (green Low, amber Medium, red High).

### Page 4 — Batch scoring (Predict)

Scores many customers at once from a spreadsheet file (CSV).

- [ ] Click **Use 10 sample test customers**: 10 rows appear, ranked from highest risk; 1 is High (France, Male, 49, 62.3%).
- [ ] Click **Download ranked list (CSV)**: a file downloads and opens in Excel.
- [ ] Click **Upload** and choose `data/sample/invalid_customers.csv` (a file we broke on purpose): a red panel says **"File failed validation, nothing was scored"** and lists 3 problems: Italy is not a known country, age 150 is above 100, 0 products is below 1.
- [ ] Open **"Error-handling demo"** at the bottom and click the button: the app rejects an invalid customer and explains why.

### Page 5 — Model performance (Govern)

How well the model works, compared with the simple first version.

- [ ] Two confusion matrices: the baseline caught 76 churners and missed 331; the final model caught 277 and missed 130.
- [ ] "Baseline → final" shows recall rising from 0.187 to 0.681.
- [ ] The experiment log lists 9 runs (E1 to E5), with the final run E5 highlighted.

### Page 6 — Fairness & use (Govern)

Whether the model treats groups differently, and how it should and should not be used.

- [ ] The table shows Germany's false positive rate in red (about 25%, versus about 9–10% for France and Spain).
- [ ] Four cards list "Use it for", "Never use it for", "Limitations" and "Before real use".

**Also check:** shrink the browser window to phone width; cards should stack instead of overlapping.

## 6. Other features to try

### Predict from the command line (no browser)

Run: `python -m src.churn.predict --input data/sample/new_customers.csv`

**Expected:** a table of 10 customers with a churn probability, a 0/1 flag and a risk level for each. Then try the broken file: `python -m src.churn.predict --input data/sample/invalid_customers.csv` should print `ERROR:` and list the problems instead of crashing.

### The notebook (step-by-step walkthrough)

1. Run `jupyter notebook notebooks/churn_prediction.ipynb`; a browser tab opens.
2. Scroll through: every step has an explanation, the code and its saved output.
3. Optional: menu **Kernel → Restart & Run All** (takes about 5 minutes); it should finish with no red error boxes.

### Deploy the app online (optional)

The **Deploy** button at the top right of the app publishes it free on Streamlit Community Cloud, so anyone can open it from a web link. It needs a GitHub account and the project uploaded to a GitHub repository.

1. Upload the project folder to a GitHub repository. The raw dataset and the trained model are deliberately left out by the `.gitignore` file (the raw file contains customer surnames).
2. In the running app, click **Deploy** → **Deploy now** under Streamlit Community Cloud, and sign in with GitHub.
3. Choose the repository, branch `main` and main file `app.py`, then click **Deploy**. The first build takes a few minutes.
4. The online app opens straight on the dashboard, because the trained model is included in the repository. If it ever shows **"Set up ChurnGuard"** instead, click **Train the model now**: it retrains on the anonymised file `data/processed/churn_cleaned.csv` in about 2–3 minutes, then the dashboard appears.

- [ ] (If tested) The deployed app opens from its link and the Predict page works.

## 7. Troubleshooting

| What you see | Why | Fix |
| --- | --- | --- |
| `python` is not recognised / command not found | Python missing or not on PATH | Install from python.org (tick "Add to PATH"); on macOS/Linux use `python3` |
| `No module named streamlit` (or pandas, sklearn) | Packages not installed, or the virtual environment is off | Switch the environment on (Step 2.2), then `pip install -r requirements.txt` |
| `Dataset not found at …` | Not in the project folder, or the data file was moved | `cd` into the `churn-prediction` folder; check `data/raw/Churn_Modelling.csv` exists |
| App shows "Set up ChurnGuard" | Model not trained on this computer | Click **Train the model now**, or run `python main.py` |
| "Trained with scikit-learn X, but this computer has Y" | The saved model came from different package versions | Run `python main.py` to retrain it on your computer |
| `Port 8501 is already in use` | The app is already running in another terminal | Close the other terminal, or run `streamlit run app.py --server.port 8502` |
| Browser does not open | Normal on some systems | Open http://localhost:8501 yourself |
| Fonts look plain | No internet; fonts come from Google Fonts | Nothing to fix: the app uses your system font |
| Your numbers differ slightly from this guide | Different package versions | Normal; compare risk levels, and use your own `outputs/results/run_summary.md` |
| A test fails | Something broke | Note the test name and the last lines of the message in your review |

## 8. Your review

Score each area from 1 (poor) to 5 (excellent) and add a short note: what worked, what confused you, anything that broke.

| Area | What to judge | Where to look | Score (1–5) | Notes |
| --- | --- | --- | --- | --- |
| Setup | Could you install and start it by following this guide alone? | Steps 1–2 | | |
| Pipeline | Did `python main.py` finish, and are the outputs understandable? | Step 3, `run_summary.md` | | |
| Reliability | Did all tests pass? Did bad input give clear messages, never a crash? | Step 3b, Batch page, command line | | |
| Predictions | Did results match section 5's table, and do they make sense? | Predict page | | |
| Usability | Is the app easy to navigate and readable for a non-expert? | All pages | | |
| Visual design | Is it clear, consistent and professional? | All pages | | |
| Explanation | Do the pages explain what the numbers mean? | Overview, Model performance | | |
| Responsible use | Are limits, fairness and privacy handled honestly? | Fairness page, README | | |
| Documentation | Are README, notebook and code comments clear? | `README.md`, notebook, `src/` | | |

**Overall impression** (2–3 sentences):

**Biggest strength:**

**Most important thing to improve:**

**Problems encountered** (what you did, what you expected, what happened, any error text):

Reviewer name, date and computer (operating system and Python version):
