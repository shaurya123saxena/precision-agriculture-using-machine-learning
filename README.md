# CultivAI: Precision Agriculture with Machine Learning

A Flask web app that helps farmers decide what to grow and how to care for it:

- **Crop recommendation:** enter soil nutrients (N, P, K), pH, rainfall and your city. The app fetches live temperature and humidity and a RandomForest model recommends the best crop, with its top 3 alternatives and confidence.
- **Fertilizer suggestion:** compares your soil's N-P-K with what the chosen crop needs and suggests how to correct it.
- **Plant disease detection:** upload a leaf photo and a ResNet9 CNN identifies the crop and disease (38 classes), with the cause and treatment advice.

Users sign up and log in (passwords hashed with bcrypt) before using the tools.

![Home page](Project-docs/App-snaps/Home.png)

## What this version changes

This project builds on [precision-agriculture-using-machine-learning](https://github.com/atharval1/precision-agriculture-using-machine-learning) by Atharva Labhasetwar, Venkata Narayana Bommanaboina and Kundan Patil. Changes in this version:

- **Retrained crop model** ([train_model.py](train_model.py)): a RandomForest (300 trees, max depth 12) trained on an 80/20 split of the 2,200-row dataset. It scores 100% on the 440 held-out rows; this dataset is known to be easy to separate.
- **Better crop results page:** confidence score and top 3 recommendations, plus range inputs (e.g. `40-60` is read as 50) and validation of every field.
- **Disease detection fixed:** inference now uses the same preprocessing the ResNet9 was trained with, converts uploads to RGB, and shows the model's confidence. An unreadable file shows a friendly error instead of crashing.
- **Weather made transparent:** if live weather is unavailable (no API key, unknown city or network error), the app uses typical values and says so on the results page.
- **No secrets in the code:** the API key, Flask secret key and database URL come from environment variables. The database file is no longer tracked; the app creates it on first run.
- **Per-user prediction history** (`/history`): every crop, fertilizer and disease result is saved to a `prediction` table with a `user_id` foreign key, and each user sees only their own rows, newest first, 10 per page.
- **Access control fixed:** every prediction route requires login; the admin dashboard now requires an admin account (a normal user gets 403); only an existing admin can create another admin (previously anyone could, via `/reg`); and sessions record the role, so an admin session can never be loaded as the regular user with the same id.
- **Tests** ([tests/test_access.py](tests/test_access.py)): 7 tests covering history isolation between users, login requirements and admin separation. Run `python -m unittest discover -s tests -v`.
- A broken `/display` route was removed.

## Run it

```bash
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

export WEATHER_API_KEY=your_key       # free key from https://openweathermap.org/api
export SECRET_KEY=any-long-random-string
python app.py                         # http://127.0.0.1:8000
```

Without `WEATHER_API_KEY` the app still works, using typical weather values (25°C, 60% humidity) and saying so on the results page.

To retrain the crop model: `python train_model.py` (writes `models/new_model.pkl` and prints the test accuracy).

## How it works

| Feature | Model / logic | Input | Data |
|---|---|---|---|
| Crop recommendation | RandomForest classifier, 22 crops | N, P, K, pH, rainfall + live temperature and humidity | [Crop recommendation dataset](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset) |
| Fertilizer suggestion | Rule-based: largest N/P/K gap against the crop's ideal | Crop name, soil N, P, K | [fertilizer.csv](Data/fertilizer.csv) |
| Disease detection | ResNet9 CNN, 38 classes, 256×256 input | Leaf image | [New Plant Diseases dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset) |

| Path | Purpose |
|---|---|
| `app.py` | Flask routes, authentication, model loading and inference |
| `config.py` | Settings from environment variables |
| `train_model.py` | Trains and saves the crop recommendation model |
| `utils/model.py` | ResNet9 architecture |
| `utils/disease.py`, `utils/fertilizer.py` | Disease descriptions and fertilizer advice text |
| `models/` | Trained models |
| `templates/`, `static/` | Pages, styles and images |

## Known limitations

- Admin accounts can be registered by anyone through `/reg`, and any logged-in user can open the admin dashboard. Proper role checks are the next fix.
- The disease model only knows the 14 crops in its training set and expects a single, clear leaf photo.
- Fertilizer advice only looks at the single largest nutrient gap.

## Credits

- Original project: [Atharva Labhasetwar](https://www.linkedin.com/in/atharva-labhasetwar), [Venkata Narayana Bommanaboina](https://www.linkedin.com/in/bvnarayana515739/), [Kundan Patil](https://www.linkedin.com/in/kundan-patil-638979199)
- Datasets: Crop recommendation (Atharva Ingle, Kaggle), fertilizer data ([Harvestify](https://github.com/Gladiator07/Harvestify)), New Plant Diseases (Kaggle)
- Weather data: [OpenWeatherMap](https://openweathermap.org/)
