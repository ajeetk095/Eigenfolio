# Security Policy

## Supported scope

This repository contains the **Quant Portfolio Intelligence** Streamlit
application (GWP3). Security review date: February 2026, re-run before each
public deployment.

## Reporting a vulnerability

Contact the developer directly: ajeetk095@gmail.com (or via GitHub issues for
non-sensitive items). Please do not open public issues for exploitable
findings.

## Security audit summary (pre-deployment review)

### Secrets & credentials
- No hardcoded API keys, passwords or tokens anywhere in the codebase.
- The app runs with **zero credentials**: Yahoo Finance public endpoints need no key.
- `.env.example` documents optional future keys; `.gitignore` blocks real `.env`
  files and `.streamlit/secrets.toml`.
- Error messages are generic; internal paths, stack traces and provider URLs
  are never surfaced to users (`src/security.py: safe_error_message`).

### Application security
- No user file uploads exist — nothing to sanitize on that surface.
- No `eval`/`exec`, no shell execution, no pickle deserialization, no
  user-supplied code paths.
- All user-controllable inputs (tickers, dates) pass strict validation
  (`src/security.py`); UI selects are bounded to the 12-asset universe.
- `maxUploadSize = 5` MB and XSRF protection enabled in `.streamlit/config.toml`.
- No authentication system is implemented — and deliberately **no fake login**.
  The app exposes only public market data and the user's own research results.
  If authentication is ever needed, use a real IdP (e.g. Streamlit's
  auth via OIDC, or Cloudflare Access in front of the deployment).

### Public deployment risks & mitigations
| Risk | Mitigation / residual risk |
|---|---|
| Resource exhaustion (public app) | Data cached with TTL (10 min quotes / 1 h history); refresh button is manual. Streamlit Cloud enforces its own resource limits. Residual: many concurrent users can still trigger API throttling. |
| External data provider failure | Clean error states; no stale or invented data is ever shown. Residual: temporary unavailability of live pages. |
| Error information leakage | Generic user-facing messages; details only in server logs. |
| API usage limits (Yahoo) | Caching + bounded asset list (12 tickers). Residual: yfinance is unofficial and can be rate-limited. |
| Personal data collection | None. Browser usage stats disabled in config. |

## Dependency & code checks (commands)

```bash
pip install pip-audit bandit ruff pytest
pip-audit -r requirements.txt
bandit -r src/ app.py pages/
ruff check src/ app.py pages/
python -m pytest tests/ -v
```

## Remaining limitations (honest disclosure)

- This application can **never be guaranteed 100% secure or hack-proof**.
- yfinance is an unofficial API and may change without notice.
- Streamlit Community Cloud runs the app unauthenticated by design; treat all
  deployed content as public.
- No DDoS protection exists at the application layer (handled — partially —
  by the hosting platform).
