"""
SmartDine: A Context-Aware Group Restaurant Recommendation System Using Machine Learning
Streamlit Web Application (app.py)
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Setup Path resolution
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

try:
    from group_preferences import MemberPreference, GroupPreferences, normalize_text
    from recommendation import SmartDineRecommender
except ImportError:
    from src.group_preferences import MemberPreference, GroupPreferences, normalize_text
    from src.recommendation import SmartDineRecommender

# Page Configuration
st.set_page_config(
    page_title="SmartDine — Context-Aware Group Dining AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    h1, h2, h3 { color: #f8fafc; font-family: 'Plus Jakarta Sans', sans-serif; }
    .rec-card {
        background: linear-gradient(135deg, rgba(22, 32, 54, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .rec-card:hover {
        border-color: rgba(245, 158, 11, 0.4);
        transform: translateY(-2px);
    }
    .score-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .exp-box {
        background: rgba(16, 185, 129, 0.06);
        border-left: 3px solid #10b981;
        padding: 0.6rem 0.9rem;
        border-radius: 0 6px 6px 0;
        margin-top: 0.7rem;
        font-size: 0.88rem;
    }
    .preset-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.06);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommender():
    data_candidate = os.path.join(ROOT_DIR, "data", "processed", "restaurant_features.csv")
    model_candidate = os.path.join(ROOT_DIR, "models", "smartdine_best_model.joblib")
    return SmartDineRecommender(data_path=data_candidate, model_path=model_candidate)


recommender = load_recommender()

# Common Options
ALL_LOCATIONS = sorted(list(set(loc for loc in recommender.locations_list if loc)))
COMMON_CUISINES = [
    "North Indian", "South Indian", "Chinese", "Fast Food", "Continental",
    "Italian", "Biryani", "Healthy Food", "Cafe", "Desserts", "Beverages",
    "Bakery", "Street Food", "Asian", "Burger", "Pizza", "Mughlai", "Seafood"
]

# Quick Presets Data
PRESETS = {
    "Custom Group": None,
    "Benchmark G001 (Healthy & Mixed Vegan/Veg)": [
        {"name": "Ananya", "cuisines": ["Healthy Food", "Fast Food"], "budget": 600, "diet": "vegan", "loc": "electronic city", "rating": 3.5},
        {"name": "Rahul", "cuisines": ["Burger", "Fast Food"], "budget": 800, "diet": "non-vegetarian", "loc": "electronic city", "rating": 3.5},
        {"name": "Sneha", "cuisines": ["Healthy Food", "Fast Food"], "budget": 700, "diet": "vegetarian", "loc": "malleshwaram", "rating": 3.5},
        {"name": "Vikram", "cuisines": ["Seafood", "Fast Food"], "budget": 900, "diet": "non-vegetarian", "loc": "bannerghatta road", "rating": 3.5},
        {"name": "Pooja", "cuisines": ["Healthy Food", "Desserts"], "budget": 650, "diet": "vegetarian", "loc": "electronic city", "rating": 3.5}
    ],
    "Benchmark G002 (Asian & Momos Foodies)": [
        {"name": "Karthik", "cuisines": ["Asian", "Chinese"], "budget": 850, "diet": "non-vegetarian", "loc": "whitefield", "rating": 3.8},
        {"name": "Meera", "cuisines": ["Seafood", "Asian"], "budget": 1000, "diet": "non-vegetarian", "loc": "whitefield", "rating": 3.5},
        {"name": "Arjun", "cuisines": ["Desserts", "Chinese"], "budget": 900, "diet": "vegetarian", "loc": "whitefield", "rating": 3.5},
        {"name": "Divya", "cuisines": ["Continental", "Asian"], "budget": 950, "diet": "vegetarian", "loc": "malleshwaram", "rating": 4.0}
    ],
    "Benchmark G005 (Strict Vegetarian Budget Hangout)": [
        {"name": "Ramesh", "cuisines": ["Fast Food", "South Indian"], "budget": 350, "diet": "vegetarian", "loc": "whitefield", "rating": 3.5},
        {"name": "Suresh", "cuisines": ["Fast Food", "Street Food"], "budget": 380, "diet": "vegetarian", "loc": "whitefield", "rating": 3.5},
        {"name": "Mahesh", "cuisines": ["Healthy Food", "Beverages"], "budget": 340, "diet": "vegetarian", "loc": "kalyan nagar", "rating": 3.5}
    ],
    "Tech Team Koramangala Hangout": [
        {"name": "Preetham", "cuisines": ["Biryani", "North Indian"], "budget": 1000, "diet": "non-vegetarian", "loc": "koramangala 5th block", "rating": 4.0},
        {"name": "Deepak", "cuisines": ["Continental", "Pizza"], "budget": 1200, "diet": "non-vegetarian", "loc": "koramangala 5th block", "rating": 3.8},
        {"name": "Tanvi", "cuisines": ["North Indian", "Chinese"], "budget": 900, "diet": "vegetarian", "loc": "koramangala 5th block", "rating": 3.8},
        {"name": "Siddharth", "cuisines": ["Biryani", "Fast Food"], "budget": 800, "diet": "non-vegetarian", "loc": "koramangala 5th block", "rating": 3.5}
    ]
}

# --- HEADER ---
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("<h1 style='font-size: 2.8rem;'>🍽️</h1>", unsafe_allow_html=True)
with col_title:
    st.title("SmartDine")
    st.caption("A Context-Aware Group Restaurant Recommendation System Using Machine Learning")

st.markdown("---")

# --- SIDEBAR: Context & Constraints ---
with st.sidebar:
    st.header("⚙️ Context & Settings")

    preset_choice = st.selectbox(
        "⚡ Quick Benchmark Preset",
        list(PRESETS.keys()),
        index=1
    )

    st.subheader("🕒 Context Attributes")
    time_of_day = st.selectbox("Time of Day", ["Lunch", "Dinner", "Late Night"], index=1)
    day_type = st.selectbox("Day Type", ["Weekend", "Weekday"], index=0)

    st.subheader("🔍 Search Filters")
    loc_filter = st.selectbox("Filter by Neighborhood", ["All Bangalore"] + [l.title() for l in ALL_LOCATIONS[:35]])
    loc_filter_clean = loc_filter if loc_filter != "All Bangalore" else None

    cui_filter = st.selectbox("Must-Have Cuisine", ["Any Cuisine"] + COMMON_CUISINES)
    cui_filter_clean = cui_filter if cui_filter != "Any Cuisine" else None

    min_rating_filter = st.slider("Filter: Minimum Rating", 0.0, 5.0, 0.0, 0.1)
    max_budget_filter = st.number_input("Filter: Maximum Budget for Two (₹)", min_value=0, max_value=8000, value=0, step=100)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        req_booking = st.checkbox("Book Table Only", False)
    with col_t2:
        req_online = st.checkbox("Online Order Only", False)

    top_n = st.slider("Number of Recommendations", 5, 20, 10)

# --- MAIN: Group Preference System ---
selected_preset_data = PRESETS.get(preset_choice)

if "group_size" not in st.session_state or selected_preset_data:
    if selected_preset_data:
        st.session_state.group_size = len(selected_preset_data)
    else:
        st.session_state.group_size = 4

st.subheader("👥 Group Preference Builder")
col_size, col_preset_info = st.columns([1, 2])
with col_size:
    group_size = st.number_input(
        "Number of Group Members",
        min_value=1,
        max_value=10,
        value=st.session_state.group_size,
        step=1
    )
    st.session_state.group_size = group_size

with col_preset_info:
    st.info(f"Context: **{time_of_day}** on a **{day_type}** | Scoring against **12,499** restaurants.")

# Individual Preferences Input
member_objects = []
st.write("Customize individual tastes, dietary restrictions, and budgets:")

cols = st.columns(min(group_size, 3))
for i in range(group_size):
    col_idx = i % min(group_size, 3)
    preset_m = selected_preset_data[i] if selected_preset_data and i < len(selected_preset_data) else None

    with cols[col_idx]:
        with st.expander(f"👤 Member {i+1}: {preset_m['name'] if preset_m else f'Friend {i+1}'}", expanded=True):
            name = st.text_input("Name", value=preset_m["name"] if preset_m else f"Friend {i+1}", key=f"name_{i}")

            diet = st.selectbox(
                "Dietary Restriction",
                ["vegetarian", "vegan", "non-vegetarian", "no preference"],
                index=["vegetarian", "vegan", "non-vegetarian", "no preference"].index(preset_m["diet"]) if preset_m else 3,
                key=f"diet_{i}"
            )

            budget = st.slider(
                "Budget for Two (₹)",
                min_value=200,
                max_value=3000,
                value=int(preset_m["budget"]) if preset_m else 800,
                step=50,
                key=f"budget_{i}"
            )

            cuisines = st.multiselect(
                "Preferred Cuisines",
                COMMON_CUISINES,
                default=preset_m["cuisines"] if preset_m else ["North Indian", "Fast Food"],
                key=f"cuisines_{i}"
            )

            pref_loc = st.selectbox(
                "Preferred Location",
                ["Flexible / Anywhere"] + [l.title() for l in ALL_LOCATIONS[:30]],
                index=(1 + [l.lower() for l in ALL_LOCATIONS[:30]].index(preset_m["loc"].lower())) if preset_m and preset_m["loc"].lower() in [l.lower() for l in ALL_LOCATIONS[:30]] else 0,
                key=f"loc_{i}"
            )
            clean_pref_loc = pref_loc if pref_loc != "Flexible / Anywhere" else ""

            member_objects.append(MemberPreference(
                name=name,
                preferred_cuisines=cuisines,
                budget=float(budget),
                dietary_preference=diet,
                preferred_location=clean_pref_loc,
                min_rating=preset_m.get("rating", 3.5) if preset_m else 3.5
            ))

# Compute Group Profile
group_prefs = GroupPreferences(members=member_objects, time_of_day=time_of_day.lower(), day_type=day_type.lower())
profile = group_prefs.compute_group_profile()

# Display Group Consensus & Harmony Summary
st.markdown("### 🤝 Aggregated Group Profile & Consensus")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Weighted Group Budget", f"₹{profile['weighted_budget']:.0f}")
with c2:
    st.metric("Dietary Inclusivity", ", ".join(profile["dietary_mix"]).title())
with c3:
    st.metric("Consensus Harmony", f"{profile['group_agreement'] * 100:.1f}%")
with c4:
    st.metric("Context", f"{time_of_day} • {day_type}")

# Recommend Button
if st.button("🚀 Recommend Restaurants", type="primary", use_container_width=True):
    with st.spinner("Scoring candidate restaurants using trained Linear Regression model..."):
        results = recommender.recommend(
            group_prefs=group_prefs,
            top_n=top_n,
            filter_location=loc_filter_clean,
            filter_cuisine=cui_filter_clean,
            max_budget=max_budget_filter if max_budget_filter > 0 else None,
            min_rating=min_rating_filter if min_rating_filter > 0 else None,
            require_booking=req_booking if req_booking else None,
            require_online=req_online if req_online else None
        )

    st.success(f"Evaluated {results['total_candidates_evaluated']} candidate restaurants. Top {len(results['recommendations'])} recommendations generated:")

    for r in results["recommendations"]:
        with st.container():
            st.markdown(f"""
            <div class="rec-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 700; color: #f59e0b; text-transform: uppercase;">Rank #{r['rank']}</span>
                        <h3 style="margin: 0.2rem 0; font-size: 1.3rem;">{r['name']}</h3>
                        <div style="color: #94a3b8; font-size: 0.85rem;">
                            📍 {r['location']} &nbsp;•&nbsp; 🍽️ {r['cuisines']} &nbsp;•&nbsp; 🏷️ {r['rest_type']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span class="score-badge">{r['match_percentage']}% Match</span>
                    </div>
                </div>
                <div style="display: flex; gap: 1.5rem; margin-top: 0.8rem; font-size: 0.85rem; color: #cbd5e1;">
                    <div><strong>Cost:</strong> ₹{r['cost_for_two']} for two</div>
                    <div><strong>Rating:</strong> ★ {r['rating'] if r['rating'] else 'New'} ({r['votes']} votes)</div>
                    <div>{'✅ Table Booking' if r['book_table'] else '❌ No Booking'}</div>
                    <div>{'✅ Online Ordering' if r['online_order'] else '❌ Dine-in Only'}</div>
                </div>
                <div class="exp-box">
                    <strong>💡 SmartDine AI Rationale:</strong> {r['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Compatibility breakdown metrics
            b_cols = st.columns(4)
            with b_cols[0]:
                st.caption(f"Cuisine Match: {r['breakdown']['cuisine_match']}%")
                st.progress(r['breakdown']['cuisine_match'] / 100)
            with b_cols[1]:
                st.caption(f"Budget Fit: {r['breakdown']['budget_match']}%")
                st.progress(r['breakdown']['budget_match'] / 100)
            with b_cols[2]:
                st.caption(f"Dietary Safety: {r['breakdown']['dietary_compatibility']}%")
                st.progress(r['breakdown']['dietary_compatibility'] / 100)
            with b_cols[3]:
                st.caption(f"Location Match: {r['breakdown']['location_match']}%")
                st.progress(r['breakdown']['location_match'] / 100)
