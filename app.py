"""
NutriScan AI - a Streamlit app that analyses what you ate (text and/or photos)
with a Gemini Flash model, estimates nutrients, and builds a personalised report
with suggestions for better nutrition.

Run locally:  streamlit run app.py
"""

import io
import json
import os
import time
from datetime import date

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🥗",
    layout="wide",
)

DEFAULT_MODEL = "gemini-flash-latest"  # alias that always points at the newest Flash
# Presets shown in the sidebar dropdown. Model names change over time, so the last
# option ("Custom...") lets you type any model ID that your API key can use.
MODEL_OPTIONS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
CUSTOM_MODEL_LABEL = "Custom..."
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snacks"]
MEAL_ICONS = {"Breakfast": "🍳", "Lunch": "🥗", "Dinner": "🍝", "Snacks": "🍎"}
MAX_IMAGE_SIDE = 1280  # px - keeps uploads small and fast


# --------------------------------------------------------------------------- #
# Structured-output schemas (Gemini returns JSON that matches these)
# --------------------------------------------------------------------------- #
class FoodItem(BaseModel):
    name: str
    portion: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float


class Micronutrient(BaseModel):
    name: str
    amount: str
    daily_value_percent: float


class MealAnalysis(BaseModel):
    is_food: bool
    summary: str
    items: list[FoodItem]
    micronutrients: list[Micronutrient]
    health_score: int
    confidence: str
    notes: str


class Suggestion(BaseModel):
    meal: str
    suggestion: str
    why: str


class DailyReport(BaseModel):
    overall_score: int
    headline: str
    summary: str
    strengths: list[str]
    gaps: list[str]
    suggestions: list[Suggestion]
    foods_to_limit: list[str]
    hydration_and_lifestyle: str


# --------------------------------------------------------------------------- #
# Helpers: configuration, targets, images
# --------------------------------------------------------------------------- #
def get_default_api_key() -> str:
    """Look for the key in Streamlit secrets first, then environment variables."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass  # no secrets.toml - perfectly fine locally
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def get_default_model() -> str:
    try:
        if "GEMINI_MODEL" in st.secrets:
            return str(st.secrets["GEMINI_MODEL"])
    except Exception:
        pass
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def compute_targets(
    sex: str, age: int, weight_kg: float, height_cm: float, activity: str, goal: str
) -> dict:
    """Estimate daily targets with the Mifflin-St Jeor equation."""
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "Male" else -161)
    factors = {
        "Sedentary (little exercise)": 1.2,
        "Lightly active (1-3 days/week)": 1.375,
        "Moderately active (3-5 days/week)": 1.55,
        "Very active (6-7 days/week)": 1.725,
    }
    tdee = bmr * factors.get(activity, 1.2)

    if goal == "Lose weight":
        calories = tdee - 500
        protein_per_kg = 1.6
    elif goal == "Gain muscle / weight":
        calories = tdee + 300
        protein_per_kg = 1.8
    else:
        calories = tdee
        protein_per_kg = 1.2

    calories = max(calories, 1500 if sex == "Male" else 1200)  # safety floor
    protein = protein_per_kg * weight_kg
    fat = calories * 0.28 / 9
    carbs = max((calories - protein * 4 - fat * 9) / 4, 0)
    return {
        "calories": round(calories),
        "protein_g": round(protein),
        "carbs_g": round(carbs),
        "fat_g": round(fat),
        "fiber_g": round(14 * calories / 1000),
        "sugar_g": round(calories * 0.10 / 4),
        "sodium_mg": 2300,
    }


def prepare_image(data: bytes) -> bytes:
    """Fix rotation, shrink and convert to JPEG bytes for Gemini."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def meal_totals(analysis: dict) -> dict:
    """Sum nutrients across food items (computed in Python so totals always add up)."""
    keys = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]
    totals = {k: 0.0 for k in keys}
    for item in analysis.get("items", []):
        for k in keys:
            totals[k] += float(item.get(k, 0) or 0)
    return totals


def day_totals(analyses: dict) -> dict:
    keys = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]
    totals = {k: 0.0 for k in keys}
    for a in analyses.values():
        t = meal_totals(a)
        for k in keys:
            totals[k] += t[k]
    return totals


def md_table(headers: list, rows: list) -> str:
    """Build a Markdown table (renders without any lazy-loaded Streamlit component)."""
    esc = lambda v: str(v).replace("|", "/").replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(esc(c) for c in row) + " |")
    return "\n".join(out)


def text_bar(value: float, total: float, width: int = 12) -> str:
    """Tiny text progress bar such as ████████░░░░."""
    if total <= 0:
        return ""
    filled = round(min(value / total, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


# --------------------------------------------------------------------------- #
# Gemini calls
# --------------------------------------------------------------------------- #
def _call_gemini(client, model: str, contents: list, schema, retries: int = 3):
    """Call Gemini with JSON-schema output and simple retry/backoff."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.2,
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
            text = (response.text or "").strip()
            if text.startswith("```"):  # strip accidental markdown fences
                text = text.strip("`")
                text = text[4:] if text.lower().startswith("json") else text
            return schema.model_validate_json(text)
        except Exception as exc:  # noqa: BLE001 - surface the final error to the UI
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gemini request failed: {last_error}")


def analyze_meal(
    client, model: str, meal: str, text: str, images: list[bytes], diet_notes: str
) -> MealAnalysis:
    prompt = f"""You are an expert registered dietitian and nutrition analyst.
Analyse the {meal.lower()} described below (text and/or photos).

Rules:
- Break the meal into individual food items and estimate a realistic portion for each
  (use the user's stated quantities when given; otherwise estimate from the photo/typical servings).
- For every item give calories (kcal), protein, carbs, fat, fiber, sugar (grams) and sodium (mg).
- List the most notable vitamins and minerals in the whole meal (e.g. Vitamin C, Iron, Calcium,
  Potassium, Vitamin B12) with an approximate amount and the % of adult Daily Value.
- health_score is an integer 1-10 for overall nutritional quality of this meal.
- confidence is one of: "low", "medium", "high" (how sure you are about the estimates).
- notes: mention important assumptions or anything unclear.
- If the input contains no food or drink, set is_food=false, items=[] and explain in summary.
- All numbers are estimates; never return null - use 0 when unknown.

User dietary notes (allergies / preferences): {diet_notes or "none"}

Meal description from the user:
\"\"\"{text.strip() or "(no text provided - rely on the photo(s))"}\"\"\""""
    contents: list = [prompt]
    for data in images:
        contents.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
    return _call_gemini(client, model, contents, MealAnalysis)


def generate_report(
    client, model: str, profile: dict, targets: dict, analyses: dict, missing_meals: list[str]
) -> DailyReport:
    meals_payload = {
        meal: {
            "summary": a["summary"],
            "health_score": a["health_score"],
            "totals": {k: round(v, 1) for k, v in meal_totals(a).items()},
            "foods": [f"{i['name']} ({i['portion']})" for i in a["items"]],
        }
        for meal, a in analyses.items()
    }
    totals = {k: round(v, 1) for k, v in day_totals(analyses).items()}
    prompt = f"""You are an expert registered dietitian writing a daily nutrition report.

USER PROFILE: {json.dumps(profile)}
DAILY TARGETS (estimated): {json.dumps(targets)}
LOGGED MEALS: {json.dumps(meals_payload)}
TOTALS SO FAR: {json.dumps(totals)}
MEALS NOT LOGGED YET: {json.dumps(missing_meals)}

Write the report:
- overall_score: integer 0-100 for how well the logged food supports the user's goal and targets.
- headline: one short, encouraging sentence.
- summary: 3-4 sentences comparing totals with targets (mention specific numbers).
- strengths: 2-4 short bullets of what was good.
- gaps: 2-5 short bullets of nutrients or food groups that are low or excessive.
- suggestions: 3-6 concrete, practical suggestions. Each has `meal` (which meal to improve or
  fill, e.g. "Dinner" or "Next snack"), `suggestion` (specific foods and portions) and `why`.
  Respect the user's dietary preferences and allergies strictly. Prefer affordable, widely
  available foods.
- foods_to_limit: 0-4 short bullets of foods from the log to reduce (empty list if none).
- hydration_and_lifestyle: 1-2 sentences of practical advice.
Be supportive, specific and non-judgemental. This is general guidance, not medical advice."""
    return _call_gemini(client, model, [prompt], DailyReport)


# --------------------------------------------------------------------------- #
# Report export
# --------------------------------------------------------------------------- #
def build_markdown_report(
    profile: dict, targets: dict, analyses: dict, report: dict | None
) -> str:
    totals = day_totals(analyses)
    lines = [f"# NutriScan AI - Nutrition Report ({date.today().isoformat()})", ""]
    lines += [
        "## Profile",
        f"- {profile['sex']}, {profile['age']} y, {profile['weight_kg']} kg, "
        f"{profile['height_cm']} cm",
        f"- Activity: {profile['activity']}",
        f"- Goal: {profile['goal']}",
        f"- Diet notes: {profile['diet_notes'] or 'none'}",
        "",
        "## Daily totals vs targets",
        "| Nutrient | Eaten | Target |",
        "|---|---|---|",
    ]
    labels = [
        ("calories", "Calories (kcal)"),
        ("protein_g", "Protein (g)"),
        ("carbs_g", "Carbs (g)"),
        ("fat_g", "Fat (g)"),
        ("fiber_g", "Fiber (g)"),
        ("sugar_g", "Sugar (g)"),
        ("sodium_mg", "Sodium (mg)"),
    ]
    for key, label in labels:
        lines.append(f"| {label} | {totals[key]:.0f} | {targets[key]} |")
    lines.append("")

    for meal, a in analyses.items():
        t = meal_totals(a)
        lines += [
            f"## {meal}",
            f"{a['summary']}",
            f"*Health score: {a['health_score']}/10 - confidence: {a['confidence']}*",
            "",
            "| Food | Portion | kcal | Protein | Carbs | Fat | Fiber |",
            "|---|---|---|---|---|---|---|",
        ]
        for i in a["items"]:
            lines.append(
                f"| {i['name']} | {i['portion']} | {i['calories']:.0f} | {i['protein_g']:.1f} | "
                f"{i['carbs_g']:.1f} | {i['fat_g']:.1f} | {i['fiber_g']:.1f} |"
            )
        lines.append(
            f"| **Total** | | **{t['calories']:.0f}** | **{t['protein_g']:.1f}** | "
            f"**{t['carbs_g']:.1f}** | **{t['fat_g']:.1f}** | **{t['fiber_g']:.1f}** |"
        )
        if a["micronutrients"]:
            lines += ["", "Key vitamins & minerals: " + ", ".join(
                f"{m['name']} {m['amount']} ({m['daily_value_percent']:.0f}% DV)"
                for m in a["micronutrients"]
            )]
        if a.get("notes"):
            lines += ["", f"_Notes: {a['notes']}_"]
        lines.append("")

    if report:
        lines += [
            "## AI Recommendations",
            f"**Overall score: {report['overall_score']}/100** - {report['headline']}",
            "",
            report["summary"],
            "",
            "### Strengths",
            *[f"- {s}" for s in report["strengths"]],
            "",
            "### Gaps",
            *[f"- {g}" for g in report["gaps"]],
            "",
            "### What to eat for better nutrition",
            *[
                f"- **{s['meal']}:** {s['suggestion']} - _{s['why']}_"
                for s in report["suggestions"]
            ],
        ]
        if report["foods_to_limit"]:
            lines += ["", "### Foods to limit", *[f"- {f}" for f in report["foods_to_limit"]]]
        lines += ["", "### Hydration & lifestyle", report["hydration_and_lifestyle"]]

    lines += [
        "",
        "---",
        "_Estimates generated by AI. This is general information, not medical advice. "
        "Consult a registered dietitian or doctor for personal medical needs._",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Sidebar: API settings + profile
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🥗 NutriScan AI")
    st.caption("Log your meals, get nutrients and smart suggestions.")

    st.subheader("🔑 Gemini settings")
    api_key = st.text_input(
        "Gemini API key",
        value=get_default_api_key(),
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    _default_model = get_default_model()
    _options = MODEL_OPTIONS + [CUSTOM_MODEL_LABEL]
    _choice = st.selectbox(
        "Gemini model",
        _options,
        index=_options.index(_default_model) if _default_model in MODEL_OPTIONS
        else len(_options) - 1,
        help="Pick a model, or choose 'Custom...' to type any model ID your key can use. "
        "Flash models are fast and cheap; Pro is slower but more careful.",
    )
    if _choice == CUSTOM_MODEL_LABEL:
        model_name = st.text_input(
            "Custom model ID",
            value="" if _default_model in MODEL_OPTIONS else _default_model,
            placeholder="e.g. gemini-3.8-flash",
        )
    else:
        model_name = _choice

    lite_mode = st.checkbox(
        "Lightweight display",
        value=True,
        help="Shows results as simple text, tables and bars. Turn ON if you see "
        "'Failed to fetch dynamically imported module' errors. Turn OFF for charts and cards.",
    )

    st.subheader("👤 Your profile")
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    col_a, col_b = st.columns(2)
    age = col_a.number_input("Age", 10, 100, 25)
    weight_kg = col_b.number_input("Weight (kg)", 30.0, 250.0, 70.0, step=0.5)
    height_cm = st.number_input("Height (cm)", 120.0, 230.0, 170.0, step=0.5)
    activity = st.selectbox(
        "Activity level",
        [
            "Sedentary (little exercise)",
            "Lightly active (1-3 days/week)",
            "Moderately active (3-5 days/week)",
            "Very active (6-7 days/week)",
        ],
    )
    goal = st.selectbox("Goal", ["Maintain weight", "Lose weight", "Gain muscle / weight"])
    diet_notes = st.text_area(
        "Diet preferences / allergies",
        placeholder="e.g. vegetarian, no peanuts, halal, lactose intolerant",
    )

profile = {
    "sex": sex,
    "age": int(age),
    "weight_kg": float(weight_kg),
    "height_cm": float(height_cm),
    "activity": activity,
    "goal": goal,
    "diet_notes": diet_notes.strip(),
}
targets = compute_targets(sex, int(age), float(weight_kg), float(height_cm), activity, goal)

# --------------------------------------------------------------------------- #
# Main page: meal inputs
# --------------------------------------------------------------------------- #
st.title("What did you eat today?")
st.write(
    "Describe each meal, upload a photo, or do both. "
    "Fill in as many meals as you like, then press **Analyze**."
)

meal_inputs: dict[str, dict] = {}
tabs = st.tabs([f"{MEAL_ICONS[m]} {m}" for m in MEAL_TYPES])
for tab, meal in zip(tabs, MEAL_TYPES):
    with tab:
        text = st.text_area(
            f"What did you eat for {meal.lower()}?",
            key=f"text_{meal}",
            placeholder="e.g. 2 parathas with a fried egg and a cup of chai with sugar",
            height=100,
        )
        files = st.file_uploader(
            f"Or upload photo(s) of your {meal.lower()}",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key=f"files_{meal}",
        )
        if files:
            cols = st.columns(min(len(files), 4))
            for col, f in zip(cols, files):
                col.image(f, width="stretch")
        meal_inputs[meal] = {"text": text, "files": files or []}

analyze_clicked = st.button("🔍 Analyze my meals", type="primary", width="stretch")

# --------------------------------------------------------------------------- #
# Run analysis
# --------------------------------------------------------------------------- #
if analyze_clicked:
    filled = {
        m: v for m, v in meal_inputs.items() if v["text"].strip() or len(v["files"]) > 0
    }
    if not api_key.strip():
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not filled:
        st.warning("Add a description or a photo for at least one meal first.")
    else:
        try:
            client = get_client(api_key.strip())
            new_analyses: dict[str, dict] = {}
            progress = st.empty()
            progress.info("⏳ Starting analysis...")
            for idx, (meal, data) in enumerate(filled.items()):
                progress.info(f"⏳ Analyzing {meal.lower()}...")
                images = [prepare_image(f.getvalue()) for f in data["files"]]
                result = analyze_meal(
                    client, model_name.strip() or DEFAULT_MODEL, meal,
                    data["text"], images, profile["diet_notes"],
                )
                if not result.is_food or not result.items:
                    st.warning(f"{meal}: no food detected. {result.summary}")
                    continue
                new_analyses[meal] = result.model_dump()

            if not new_analyses:
                progress.empty()
                st.error("No food could be identified. Try a clearer photo or description.")
            else:
                progress.info("⏳ Writing your personalised report...")
                missing = [m for m in MEAL_TYPES if m not in new_analyses]
                report = generate_report(
                    client, model_name.strip() or DEFAULT_MODEL,
                    profile, targets, new_analyses, missing,
                )
                progress.empty()
                st.session_state["analyses"] = new_analyses
                st.session_state["report"] = report.model_dump()
                st.session_state["snapshot"] = {"profile": profile, "targets": targets}
                st.session_state["model_used"] = model_name.strip() or DEFAULT_MODEL
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")
            st.info(
                "Check your API key and internet connection. If the error mentions "
                "'not found' or '404', pick a different model in the sidebar."
            )

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
analyses = st.session_state.get("analyses")
report = st.session_state.get("report")
snapshot = st.session_state.get("snapshot")

if analyses:
    st.divider()
    st.header("📊 Your nutrition report")
    if st.session_state.get("model_used"):
        st.caption(f"Analysed with: {st.session_state['model_used']}")
    used_profile = snapshot["profile"] if snapshot else profile
    used_targets = snapshot["targets"] if snapshot else targets
    totals = day_totals(analyses)

    nutrient_rows = [
        ("calories", "Calories", "kcal"),
        ("protein_g", "Protein", "g"),
        ("carbs_g", "Carbs", "g"),
        ("fat_g", "Fat", "g"),
        ("fiber_g", "Fiber", "g"),
        ("sugar_g", "Sugar (limit)", "g"),
        ("sodium_mg", "Sodium (limit)", "mg"),
    ]

    # --- Totals vs targets ---------------------------------------------------
    st.subheader("Today vs your targets")
    if lite_mode:
        rows = []
        for key, label, unit in nutrient_rows:
            eaten, target = totals[key], used_targets[key]
            pct = f"{eaten / target * 100:.0f}%" if target else ""
            rows.append(
                [label, f"{eaten:.0f} {unit}", f"{target} {unit}", pct, text_bar(eaten, target)]
            )
        st.markdown(md_table(["Nutrient", "Eaten", "Target", "% of target", ""], rows))
    else:
        metric_cols = st.columns(5)
        for col, (key, label, unit) in zip(metric_cols, nutrient_rows[:5]):
            target = used_targets[key]
            eaten = totals[key]
            col.metric(label, f"{eaten:.0f} {unit}", f"target {target} {unit}", delta_color="off")
            col.caption(f"{min(eaten / target * 100, 999):.0f}% of target" if target else "")
        extra_cols = st.columns(2)
        extra_cols[0].metric(
            "Sugar", f"{totals['sugar_g']:.0f} g",
            f"limit ~{used_targets['sugar_g']} g", delta_color="off",
        )
        extra_cols[1].metric(
            "Sodium", f"{totals['sodium_mg']:.0f} mg",
            f"limit {used_targets['sodium_mg']} mg", delta_color="off",
        )

    # --- Per-meal comparison -------------------------------------------------
    if lite_mode:
        st.markdown("**Calories and macros by meal**")
        max_cal = max((meal_totals(a)["calories"] for a in analyses.values()), default=0)
        meal_rows = []
        for m, a in analyses.items():
            mt = meal_totals(a)
            meal_rows.append([
                m, f"{mt['calories']:.0f}", f"{mt['protein_g']:.0f}", f"{mt['carbs_g']:.0f}",
                f"{mt['fat_g']:.0f}", text_bar(mt["calories"], max_cal),
            ])
        st.markdown(md_table(
            ["Meal", "kcal", "Protein (g)", "Carbs (g)", "Fat (g)", "Calories"], meal_rows
        ))
    else:
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("**Calories by meal**")
            cal_df = pd.DataFrame(
                {"kcal": [meal_totals(a)["calories"] for a in analyses.values()]},
                index=list(analyses.keys()),
            )
            st.bar_chart(cal_df)
        with chart_right:
            st.markdown("**Macros by meal (grams)**")
            macro_df = pd.DataFrame(
                {
                    "Protein": [meal_totals(a)["protein_g"] for a in analyses.values()],
                    "Carbs": [meal_totals(a)["carbs_g"] for a in analyses.values()],
                    "Fat": [meal_totals(a)["fat_g"] for a in analyses.values()],
                },
                index=list(analyses.keys()),
            )
            st.bar_chart(macro_df)

    # --- Per-meal breakdown --------------------------------------------------
    st.subheader("Meal breakdown")
    for meal, a in analyses.items():
        t = meal_totals(a)
        with st.expander(
            f"{MEAL_ICONS.get(meal, '🍽️')} {meal} - {t['calories']:.0f} kcal "
            f"- score {a['health_score']}/10",
            expanded=lite_mode,
        ):
            st.write(a["summary"])
            st.caption(f"Estimate confidence: {a['confidence']}")
            if lite_mode:
                item_rows = [
                    [i["name"], i["portion"], f"{i['calories']:.0f}", f"{i['protein_g']:.1f}",
                     f"{i['carbs_g']:.1f}", f"{i['fat_g']:.1f}", f"{i['fiber_g']:.1f}",
                     f"{i['sugar_g']:.1f}", f"{i['sodium_mg']:.0f}"]
                    for i in a["items"]
                ]
                item_rows.append([
                    "**Total**", "", f"**{t['calories']:.0f}**", f"**{t['protein_g']:.1f}**",
                    f"**{t['carbs_g']:.1f}**", f"**{t['fat_g']:.1f}**", f"**{t['fiber_g']:.1f}**",
                    f"**{t['sugar_g']:.1f}**", f"**{t['sodium_mg']:.0f}**",
                ])
                st.markdown(md_table(
                    ["Food", "Portion", "kcal", "Protein", "Carbs", "Fat", "Fiber",
                     "Sugar", "Sodium (mg)"], item_rows,
                ))
                if a["micronutrients"]:
                    st.markdown("**Key vitamins & minerals**")
                    st.markdown(md_table(
                        ["Nutrient", "Amount", "% Daily Value"],
                        [[m["name"], m["amount"], f"{m['daily_value_percent']:.0f}%"]
                         for m in a["micronutrients"]],
                    ))
            else:
                items_df = pd.DataFrame(a["items"]).rename(
                    columns={
                        "name": "Food", "portion": "Portion", "calories": "kcal",
                        "protein_g": "Protein (g)", "carbs_g": "Carbs (g)", "fat_g": "Fat (g)",
                        "fiber_g": "Fiber (g)", "sugar_g": "Sugar (g)",
                        "sodium_mg": "Sodium (mg)",
                    }
                )
                st.dataframe(items_df, width="stretch", hide_index=True)
                if a["micronutrients"]:
                    st.markdown("**Key vitamins & minerals**")
                    micro_df = pd.DataFrame(a["micronutrients"]).rename(
                        columns={
                            "name": "Nutrient", "amount": "Amount",
                            "daily_value_percent": "% Daily Value",
                        }
                    )
                    st.dataframe(micro_df, width="stretch", hide_index=True)
            if a.get("notes"):
                st.info(a["notes"])

    # --- AI recommendations --------------------------------------------------
    if report:
        st.subheader("🧠 Recommendations")
        if lite_mode:
            st.markdown(f"### Overall score: {report['overall_score']}/100")
            st.markdown(f"**{report['headline']}**")
            st.write(report["summary"])
        else:
            score_col, text_col = st.columns([1, 3])
            score_col.metric("Overall score", f"{report['overall_score']}/100")
            text_col.markdown(f"### {report['headline']}")
            text_col.write(report["summary"])

        good_col, gap_col = st.columns(2)
        with good_col:
            st.markdown("#### ✅ What went well")
            for s in report["strengths"]:
                st.markdown(f"- {s}")
        with gap_col:
            st.markdown("#### ⚠️ What's missing")
            for g in report["gaps"]:
                st.markdown(f"- {g}")

        st.markdown("#### 🍽️ What to eat for better nutrition")
        for s in report["suggestions"]:
            st.success(f"**{s['meal']}:** {s['suggestion']}\n\n_Why:_ {s['why']}")

        if report["foods_to_limit"]:
            st.markdown("#### 🚫 Foods to limit")
            for f in report["foods_to_limit"]:
                st.markdown(f"- {f}")

        st.markdown("#### 💧 Hydration & lifestyle")
        st.write(report["hydration_and_lifestyle"])

    # --- Download ------------------------------------------------------------
    st.download_button(
        "⬇️ Download full report (Markdown)",
        data=build_markdown_report(used_profile, used_targets, analyses, report),
        file_name=f"nutrition_report_{date.today().isoformat()}.md",
        mime="text/markdown",
        width="stretch",
    )

st.caption(
    "⚠️ Nutrient values are AI estimates and may be inaccurate, especially from photos. "
    "This app provides general information, not medical advice."
)
