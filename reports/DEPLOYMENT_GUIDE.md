# 🚀 Deployment Guide — GitHub + Streamlit Community Cloud (free)

Beginner-friendly, zero-cost, no local server required. Total time: ~15 minutes.
Verified against the official Streamlit Community Cloud flow (2026):
<https://docs.streamlit.io/deploy/streamlit-community-cloud>

---

## Part A — Put the project on GitHub

### A1. Sign in / create an account
1. Go to <https://github.com> and click **Sign up** (or sign in as **ajeetk095**).
2. Verify your email if asked.

### A2. Create the repository
1. Click the **+** icon (top-right) → **New repository**.
2. Repository name: `GWP3-Quant-Portfolio-Intelligence`.
3. Visibility: **Public** (required for the free Community Cloud tier).
4. Do **not** tick "Add a README" (our project already has one).
5. Click **Create repository**.

### A3. Upload the project
**Easiest way (web upload):**
1. On the new empty repo page, click **uploading an existing file**.
2. Drag the **contents** of this folder (all files and folders — keep the
   folder structure: `src/`, `pages/`, `.streamlit/`, `tests/`, `reports/`).
   Tip: file managers show hidden folders like `.streamlit` after
   Ctrl+H (Windows/Linux) or Cmd+Shift+. (macOS).
3. Commit message: `Initial GWP3 deployment` → **Commit changes**.

**Better way (Git command line, if installed):**
```bash
cd GWP3-Quant-Portfolio-Intelligence
git init
git add .
git commit -m "Initial GWP3 deployment"
git branch -M main
git remote add origin https://github.com/ajeetk095/GWP3-Quant-Portfolio-Intelligence.git
git push -u origin main
```

### A4. Files that MUST be included
`app.py`, `requirements.txt`, the entire `src/` folder, the entire `pages/`
folder, `.streamlit/config.toml`, `tests/`, `reports/`, `README.md`.

### A5. Files that must NEVER be uploaded
- any real `.env` or `.streamlit/secrets.toml` (secrets!)
- `__pycache__/` folders, `.venv/`, `*.csv` caches (already blocked by `.gitignore`)

### A6. Verify
On the repo page confirm you can open `src/config.py`, `pages/8_About_Developer.py`
and `.streamlit/config.toml`. Update anytime with **Add file → Upload files** or
`git add . && git commit -m "update" && git push`.

---

## Part B — Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and click **Sign up** → choose
   **Continue with GitHub** (authorize the connection — this step must be
   completed by you, the account owner).
2. Click **Create app** (or **New app**).
3. Fill in exactly:
   - **Repository:** `ajeetk095/GWP3-Quant-Portfolio-Intelligence`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick a short name, e.g. `quant-portfolio-intelligence`
4. Click **Deploy**. First build takes ~3–5 minutes (installs the pinned
   dependencies, then downloads market data on first visit).

### Secrets (not needed for this app)
The app requires no credentials. If you ever add a private data provider:
App menu (⋮) → **Settings → Secrets**, then paste TOML — never commit them.

### Check the logs
While building, click **Manage app** → the terminal pane shows live logs.
"Deployed!" + your app URL = success.

### Fix common dependency errors
| Log message | Fix |
|---|---|
| `ModuleNotFoundError: pypfopt` | Confirm `requirements.txt` is in the repo root and lists `PyPortfolioOpt==1.5.5`. |
| `Could not install packages` / version conflict | Confirm Python version in Settings → Python version → 3.11/3.12. |
| `numpy` build error | Keep `numpy==1.26.4` exactly as pinned. |
| App boots then shows a data error | Yahoo Finance hiccup — press 🔄 Refresh; the app deliberately shows an error instead of fake data. |

### Update the live app
Any change pushed to `main` (web upload → Commit, or `git push`) redeploys
automatically in ~1 minute.

### Monitor after deployment
- **Manage app** → logs, memory use, reboot.
- Analytics: viewer counts in the dashboard.
- Free-tier notes: apps sleep after inactivity (first visit wakes them) and
  have ~1 GB RAM — this app uses far less thanks to caching.

---

## Part C — Pre-deployment checklist

- [ ] All folders visible on GitHub (`src/`, `pages/`, `.streamlit/`)
- [ ] `requirements.txt` in the repo root
- [ ] No `.env` / secrets committed
- [ ] `python -m pytest tests/ -v` passed locally
- [ ] Security review read (`SECURITY.md`)
- [ ] Deployed URL loads and Home page shows KPI cards + recommendations
- [ ] Refresh button re-computes the model successfully
