# 🥗 NutriScan AI

A Streamlit app that analyses what you eat. Type what you ate or upload a photo of
your breakfast, lunch, dinner or snacks. Google Gemini estimates calories, macros and key
vitamins and minerals, compares them with your personal daily targets, and writes a
full report with suggestions on what to eat next.

## Features
- Text and/or photo input for Breakfast, Lunch, Dinner and Snacks
- Per-food nutrient breakdown (calories, protein, carbs, fat, fiber, sugar, sodium)
- Key vitamins and minerals with % Daily Value
- Personal daily targets from age, sex, weight, height, activity and goal
- AI report: score, strengths, gaps, specific food suggestions, foods to limit
- Charts and a downloadable Markdown report
- Respects dietary preferences and allergies you enter

## Tech stack
Streamlit, Google Gemini (`google-genai` SDK) with structured JSON output, Pillow, pandas.

## Run locally
1. Install Python 3.10 or newer.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free Gemini API key at https://aistudio.google.com/apikey
4. Start the app:
   ```bash
   streamlit run app.py
   ```
5. Paste your API key in the sidebar, or set it beforehand:
   - macOS/Linux: `export GEMINI_API_KEY="your-key"`
   - Windows (PowerShell): `$env:GEMINI_API_KEY="your-key"`

## Configuration
| Setting | Where | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | env var or Streamlit secret | Your Gemini API key |
| `GEMINI_MODEL` | env var or Streamlit secret (optional) | Model name (default `gemini-flash-latest`) |

Never commit your API key to GitHub.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and click **Create app**.
3. Select the repo, branch `main` and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Disclaimer
Nutrient values are AI estimates and can be inaccurate, especially from photos.
This app gives general information and is not medical advice.
