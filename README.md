# Roamly recommendation backend v1

A Flask API for destination recommendations. It combines a PyTorch neural collaborative filter, TF-IDF content similarity, and destination popularity.

This repository is the original recommendation backend. The later Roamly application added a frontend and FastAPI-based LLM agent orchestration; those components are not included in this repository.

## Architecture

1. The collaborative filter learns user and country embeddings and predicts a destination rating. Concatenated embeddings pass through three ReLU hidden layers and a scalar output. Training uses MSE loss, Adam, and a stepped learning rate.
2. The content model vectorizes destination keywords and climate with TF-IDF and uses cosine similarity. The hybrid path seeds content matching from the collaborative model's five highest-rated countries.
3. The API combines popularity, predicted rating, and content similarity using weights of 0.15, 0.70, and 0.15. These are configured weights, not an experimentally optimized policy.
4. Flask returns destination rankings and city coordinates to a client application.

The model and encoders are loaded from `recommender/model/` and `recommender/*.pkl`. Run commands from the repository root because the original code uses relative data paths.

## Included data

| File | Scope |
| --- | --- |
| `ratings.csv` | 120,849 ratings, 1,486 distinct user IDs, 222 destination IDs; ratings range from 1 to 5 |
| `world-countries.csv` | 222 destinations with keywords, climate, and daily cost fields |
| `world-popularity.csv` | 222 popularity entries |
| `world-cities.csv` | 26,146 city records with country and coordinates |

These are dataset counts, not evidence of active users or client adoption. Data collection provenance and reuse licensing are not established by this repository.

## Setup

Use Python 3.12. The pinned versions preserve the legacy pandas behavior and match the saved scikit-learn encoders. From the repository root, create an isolated environment. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
.\.venv\Scripts\python.exe serverAPI.py
```

On Linux/macOS, use `python3.12 -m venv .venv` and substitute `.venv/bin/python` for the Windows interpreter path. The CPU wheel index is suitable for the Windows/Linux CPU setup; use the appropriate PyTorch installation for other platforms.

The development server listens on port 5000 and binds to all interfaces. It is a demonstration API, not a production deployment configuration. The checked-in model and encoders are required for personalized recommendations; load serialized artifacts only from a trusted source.

## API examples

```text
GET http://127.0.0.1:5000/login?userId=0
GET http://127.0.0.1:5000/recommend?userId=0
GET http://127.0.0.1:5000/recommend?popular=1
GET http://127.0.0.1:5000/recommend/cities?country=Pakistan
```

- `/login` checks whether an integer ID exists in the ratings data. It does not authenticate an identity.
- `/recommend?userId=...` returns the hybrid destination ranking for a known ID.
- `/recommend?popular=1` returns the 20 highest popularity entries.
- `/recommend/cities?country=...` returns city names and coordinates. For larger sets, it selects the first three and up to five random remaining cities; this selection is not a learned city ranking.
- Missing, malformed, or unknown user IDs return HTTP 400. Missing country returns HTTP 400; a country with no matching cities returns an empty list.

## Fresh model training and evaluation

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

This trains the existing neural architecture from scratch for 20 epochs on CPU with seed 42. It uses a stratified 80/20 rating split, reports RMSE on held-out ratings, and compares against global-mean and user-mean baselines calculated only from training ratings. Results, counts, package versions, and file hashes are written to `benchmark-results.json`.

The evaluation does not overwrite the original model or encoders. The historical checkpoint's training provenance is unknown, so this is a fresh reproducibility benchmark, not a recovered historical score. The legacy `train_NN` function also trains the architecture, prompts before saving, and replaces the encoders; use a separate checkout if experimenting with that path.

### Recorded offline benchmark

Verified on Windows with Python 3.12 and the pinned CPU setup. Lower RMSE is better; the rating scale is 1 to 5.

| Method | Validation rating RMSE |
| --- | ---: |
| Neural collaborative filter | 0.8957 |
| Training-user mean baseline | 1.3890 |
| Global training-mean baseline | 1.5800 |

Training used 96,679 ratings; validation used 24,170. These results come from fresh training of the existing architecture, not from evaluating the old saved checkpoint. The fixed settings follow the original training defaults, with seed 42 and no search over configurations in this benchmark. Hardware or dependency changes may affect exact results.


RMSE evaluates rating prediction on a random split. It does not establish ranking relevance, cold-user generalization, temporal performance, travel booking conversion, or production adoption. The included rating data's external representativeness is unverified.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_reproducibility.py
```

The checks cover user-ID validation, popularity and personalized responses, city results, and the model save location without overwriting the checked-in checkpoint. The full evaluation also checks data completeness, duplicate user/destination pairs, split separation, and finite predictions.

## Current limits

There is no authentication, request throttling, persistent serving cache, or live-client evaluation. Budget fields exist in the content recommender but do not currently enforce filtering. The later FastAPI/agent workflow is outside this codebase. This backend is useful evidence of recommendation implementation, not evidence that the later agent system was production deployed.
