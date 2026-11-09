# Setup guide

## 1. Install Python
Python 3.10, 3.11 or 3.12 from python.org. On Windows, tick **"Add Python to PATH"** during install.

## 2. Create a virtual environment
Open a terminal in the project folder:
```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
```

## 3. Install libraries
On Linux, install the CPU version of PyTorch first to avoid a 2 GB GPU download:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then (all systems):
```
pip install -r requirements.txt
```

## 4. Download NLP models (needs internet, once)
```
python scripts/download_models.py
```
This gets spaCy `en_core_web_sm` (~12 MB), the MiniLM embedding model (~90 MB) and the cross-encoder (~90 MB).

## 5. Run
```
python app.py
```
Open **http://localhost:5000**, create an account and upload a resume.
Next time, just double-click `run.bat` (Windows) or run `./run.sh`.

The terminal shows which NLP components loaded. "fallback in use" means that
component is missing; the site still works with lighter methods.

## 6. Optional free keys (add them on the Settings page)
| Key | Where |
|---|---|
| JSearch (live Google for Jobs results) | openwebninja.com or rapidapi.com, search "JSearch" |
| Adzuna app ID + key | developer.adzuna.com |
| Gemini | aistudio.google.com, "Get API key" |

Results are cached for 12 hours to save the free quota. Use "Refresh results" to force a new search.

## 7. Train the fake job model (optional)
1. Download "Real / Fake Job Posting Prediction" from Kaggle (file `fake_job_postings.csv`).
2. Put it in `data/`.
3. Run `python scripts/train_fake_job.py` and restart the website.

## Troubleshooting
| Problem | Fix |
|---|---|
| `pip` not found | Use `python -m pip install ...` |
| spaCy model error | `python -m spacy download en_core_web_sm` |
| Models fail to download | Check internet; the site still runs in fallback mode |
| "Very little text could be read" | The PDF is scanned. Export a text PDF or DOCX |
| Gemini model error | Google may have renamed models; enter a current model name in Settings |
| Port 5000 busy | Change `port=5000` at the bottom of `app.py` |
| Start fresh | Delete `instance/career_compass.db` |
