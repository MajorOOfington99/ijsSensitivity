import os
import re
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(layout="wide", page_title="ISU Score Simulation Dashboard")
st.title("ISU Score Simulation Dashboard")

st.markdown(
    "This dashboard lets you test **figure skating scoring rule changes** on historical competition data. "
    "Adjust the parameters in the sidebar (GOE scaling, base value scale, fall deduction, and SOV levers) "
    "to simulate how new rules would affect scores and placements. Select a competition segment, and view the "
    "recalculated rankings and each skater’s element-by-element score breakdown under the new rules."
)

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    base_df = None
    sim_df = None

    if os.path.exists("base_value.csv"):
        base_df = pd.read_csv("base_value.csv")
    if os.path.exists("isu_simulation.csv"):
        sim_df = pd.read_csv("isu_simulation.csv")

    return base_df, sim_df


base_values, df_sim = load_data()

if df_sim is None:
    st.error("Could not find `isu_simulation.csv` in the same folder as this app file.")
    st.stop()
if base_values is None:
    st.warning("Could not find `base_value.csv` in the same folder as this app file. (App can still run without it.)")

df_sim.columns = [c.strip() for c in df_sim.columns]

# -----------------------------
# Validate required columns
# -----------------------------
required_cols = [
    "Year", "Discipline", "Segment",
    "Skater_Name", "Rank",
    "Element_Name",
    "Base_Value",
    "GOE_Mid7_Avg",
    "GOE",
    "Element_Score",
    "Total_Element_Score", "Total_Component_Score", "Total_Deductions", "Total_Segment_Score",
]
missing = [c for c in required_cols if c not in df_sim.columns]
if missing:
    st.error(f"`isu_simulation.csv` is missing required columns: {missing}")
    st.stop()

has_goe_base = "GOE_Base" in df_sim.columns

# -----------------------------
# Sidebar - Event selection
# -----------------------------
years = sorted(df_sim["Year"].dropna().unique().tolist())
disciplines = {"Men": "M", "Women": "W", "Pairs": "P"}
segments = {"Short Program": "Short", "Free Skate": "Free"}

st.sidebar.header("Select Competition Segment")
selected_year = st.sidebar.selectbox("Year", years, index=max(0, len(years) - 1))
selected_disc = st.sidebar.selectbox("Discipline", list(disciplines.keys()), index=0)
selected_seg = st.sidebar.selectbox("Segment", list(segments.keys()), index=1)

disc_code = disciplines[selected_disc]
seg_code = segments[selected_seg]

event_mask = (
    (df_sim["Year"] == selected_year)
    & (df_sim["Discipline"] == disc_code)
    & (df_sim["Segment"] == seg_code)
)
event_df_raw = df_sim[event_mask].copy()

if event_df_raw.empty:
    st.error("No data available for the selected event.")
    st.stop()

# -----------------------------
# Sidebar - Simulation parameters
# -----------------------------
st.sidebar.header("Simulation Parameters")

GOE_pct = st.sidebar.slider(
    "GOE value per ±1 GOE (percent of base)",
    min_value=5, max_value=20, value=10, step=1,
    help="Percentage of GOE reference base for each +1 or -1 GOE. (10% is the common current rule)"
)

base_scale = st.sidebar.slider(
    "Base Value Scale Factor (global)",
    min_value=0.80, max_value=1.50, value=1.00, step=0.01,
    help="Global multiplier on all element base values. (1.00 = no change)"
)

fall_value = st.sidebar.slider(
    "Fall Deduction (points per fall)",
    min_value=0, max_value=3, value=1, step=1,
    help="Points deducted for each fall. (1.0 is the current rule)"
)

# -----------------------------
# NEW: SOV adjustments by element type (±20%)
# -----------------------------
SOV_TYPES = ["Jump", "Spin", "Step", "Choreo", "Lift", "Twist", "Death Spiral"]

with st.sidebar.expander("Scale of Values (SOV) by element type (±20%)", expanded=False):
    st.caption(
        "Apply a separate ±% multiplier to base values (and proportional GOE) for each element type. "
        "Defaults are 0% (no change)."
    )
    type_adj_pct = {}
    for t in SOV_TYPES:
        type_adj_pct[t] = st.slider(
            f"{t} adjustment (%)",
            min_value=-20, max_value=20, value=0, step=1
        )

type_scales = {t: 1.0 + (type_adj_pct[t] / 100.0) for t in SOV_TYPES}

# -----------------------------
# NEW: Jump rotation levers (±20%)
# -----------------------------
ROT_BUCKETS = [
    (1, "Single (1)"),
    (2, "Double (2)"),
    (3, "Triple (3)"),
    (4, "Quad (4)"),
    (5, "Quint (5)"),
]

with st.sidebar.expander("Jump rotations (±20%)", expanded=False):
    st.caption(
        "Apply a separate ±% multiplier to base values (and proportional GOE) for JUMPS only, "
        "bucketed by the leading rotation digit in the jump code (e.g., 3Lz, 4T, 5Lz)."
    )
    rot_adj_pct = {}
    for r, label in ROT_BUCKETS:
        rot_adj_pct[r] = st.slider(
            f"{label} adjustment (%)",
            min_value=-20, max_value=20, value=0, step=1
        )

rot_scales = {r: 1.0 + (rot_adj_pct[r] / 100.0) for r, _ in ROT_BUCKETS}

# -----------------------------
# Core math helpers
# -----------------------------
CHSQ_STEP_POINTS = 0.5

def _ensure_bool_second_half(df: pd.DataFrame) -> pd.DataFrame:
    if "Second_Half" not in df.columns:
        df["Second_Half"] = False
        return df
    if df["Second_Half"].dtype == bool:
        return df
    df["Second_Half"] = (
        df["Second_Half"]
        .astype(str).str.strip().str.lower()
        .isin(["true", "1", "t", "yes", "y"])
    )
    return df

def classify_element(element_name: str) -> str:
    name = (element_name or "").replace(" ", "").lower()

    # Pairs heuristics (your original intent)
    is_pairs = any(tag in name for tag in ["th", "li", "ds", "tw", "pcossp"])
    if is_pairs:
        if "li" in name:
            return "Lift"
        if "ds" in name:
            return "Death Spiral"
        if "tw" in name:
            return "Twist"
        if "sp" in name or "pcossp" in name:
            return "Spin"
        if "chsq" in name:
            return "Choreo"
        if "sq" in name:
            return "Step"
        return "Jump"

    if "chsq" in name:
        return "Choreo"
    if "sq" in name:
        return "Step"
    if "sp" in name:
        return "Spin"
    return "Jump"

def parse_jump_rotation_bucket(element_name: str) -> int | None:
    """
    For a *jump element row*, infer rotation bucket (1..5) by scanning jump tokens.
    We take the MAX rotation found in the element string (e.g., 4T+3T -> 4).
    """
    if element_name is None or (isinstance(element_name, float) and np.isnan(element_name)):
        return None

    s = str(element_name).strip()
    if not s:
        return None

    # normalize & remove common symbols/annotations that can confuse tokenization
    s = s.replace(" ", "")
    s = re.sub(r"\+REP", "", s, flags=re.IGNORECASE)

    # split combos/sequences into parts (3Lz+3T+2T etc.)
    parts = re.split(r"\+", s)

    rots: list[int] = []
    for p in parts:
        # strip non-alnum to make matching safer (removes <, <<, !, q, etc.)
        p2 = re.sub(r"[^0-9A-Za-z]", "", p)
        m = re.match(r"^([1-5])", p2)
        if m:
            rots.append(int(m.group(1)))

    if not rots:
        return None
    return max(rots)

def compute_segment_summaries(
    df_in: pd.DataFrame,
    goe_pct: float,
    bv_scale: float,
    fall_pts: float,
    *,
    type_scales: dict | None = None,
    rot_scales: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_elements: element-level with Base_Value_new / GOE_points_new / Element_Score_new (+ audit columns)
      df_seg:      skater+segment summary with Total_Segment_Score_new plus reconciled deduction pieces
    """
    if type_scales is None:
        type_scales = {}
    if rot_scales is None:
        rot_scales = {}

    df = df_in.copy()
    df = _ensure_bool_second_half(df)

    # numeric
    for col in [
        "Base_Value", "GOE", "GOE_Mid7_Avg", "Element_Score",
        "Total_Element_Score", "Total_Component_Score", "Total_Deductions", "Total_Segment_Score",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "GOE_Base" in df.columns:
        df["GOE_Base"] = pd.to_numeric(df["GOE_Base"], errors="coerce")

    # classify element type
    df["Element_Type_sim"] = df["Element_Name"].astype(str).apply(classify_element)
    df["Type_Scale"] = df["Element_Type_sim"].map(type_scales).fillna(1.0).astype(float)

    # rotation bucket ONLY for jumps
    jump_mask = df["Element_Type_sim"].eq("Jump")
    df["Jump_Rotation"] = np.nan
    if jump_mask.any():
        df.loc[jump_mask, "Jump_Rotation"] = df.loc[jump_mask, "Element_Name"].apply(parse_jump_rotation_bucket)

    df["Rotation_Scale"] = df["Jump_Rotation"].map(rot_scales).fillna(1.0).astype(float)

    # detect REP
    rep_mask = df["Element_Name"].astype(str).str.contains(r"\+REP", case=False, na=False)

    # avoid double-REP if CSV already applied it
    rep_already_applied = True  # safe default
    if rep_mask.any() and ("GOE_Base" in df.columns):
        name_no_rep = df["Element_Name"].astype(str).str.replace(r"\+REP", "", regex=True, case=False)
        clean_calls = ~name_no_rep.str.contains(r"<<|<|q|e|!", case=False, na=False)
        clean_rep = rep_mask & clean_calls & (~df["Second_Half"].fillna(False))

        if clean_rep.any():
            expected = df.loc[clean_rep, "GOE_Base"] * 0.70
            actual = df.loc[clean_rep, "Base_Value"]
            rep_already_applied = ((actual - expected).abs() <= 0.03).any()

    rep_mult = 0.70 if (rep_mask.any() and (not rep_already_applied)) else 1.00

    # combined base scale used per-row
    df["BV_Scale_Used"] = (float(bv_scale) * df["Type_Scale"] * df["Rotation_Scale"]).astype(float)

    # base value scaling (+ optional REP application)
    df["Base_Value_new"] = (df["Base_Value"] * df["BV_Scale_Used"]).astype(float)
    if rep_mult != 1.00:
        df.loc[rep_mask, "Base_Value_new"] = (df.loc[rep_mask, "Base_Value_new"] * rep_mult).astype(float)

    # GOE scaling
    is_chsq = df["Element_Name"].astype(str).str.contains("chsq", case=False, na=False)

    # For most elements: scale protocol GOE proportionally with BV changes
    df["GOE_points_new"] = (
        df["GOE"].fillna(0.0) * (float(goe_pct) / 10.0) * df["BV_Scale_Used"]
    ).astype(float)

    # For ChSq: keep fixed-step approach, and only apply the CHOREO type lever (no rotation lever)
    df.loc[is_chsq, "GOE_points_new"] = (
        df.loc[is_chsq, "GOE_Mid7_Avg"].fillna(0.0) * CHSQ_STEP_POINTS * df.loc[is_chsq, "Type_Scale"]
    ).astype(float)

    df["GOE_points_new"] = df["GOE_points_new"].round(2)
    df["Element_Score_new"] = (df["Base_Value_new"] + df["GOE_points_new"]).round(2)

    # segment summaries
    seg = (
        df.groupby(["Skater_Name", "Segment"], as_index=False)
        .agg(
            Total_Element_Score_new=("Element_Score_new", "sum"),
            Total_Element_Score=("Total_Element_Score", "first"),
            Total_Component_Score=("Total_Component_Score", "first"),
            Total_Deductions=("Total_Deductions", "first"),
            Total_Segment_Score=("Total_Segment_Score", "first"),
            Original_Rank=("Rank", "first"),
        )
    )

    # reconcile deductions so default matches, and only "fall lever" changes the fall portion
    seg["Deductions_Effective"] = (
        seg["Total_Segment_Score"] - seg["Total_Element_Score"] - seg["Total_Component_Score"]
    ).round(2)

    seg["Falls"] = (
        pd.to_numeric(seg["Deductions_Effective"], errors="coerce")
        .fillna(0.0)
        .abs()
        .round()
        .astype(int)
    )

    # Non-fall deductions inferred so that default mode always matches:
    # D_eff = NonFall + (-Falls*1.0)  =>  NonFall = D_eff + Falls
    seg["NonFall_Deductions"] = (seg["Deductions_Effective"] + seg["Falls"] * 1.0).round(2)

    # New deductions: keep non-fall same, change only the fall part
    seg["Total_Deductions_new"] = (seg["NonFall_Deductions"] - seg["Falls"] * float(fall_pts)).round(2)

    seg["Total_Segment_Score_new"] = (
        seg["Total_Element_Score_new"] + seg["Total_Component_Score"] + seg["Total_Deductions_new"]
    ).round(2)

    return df, seg


# -----------------------------
# Compute selected program + overall (SP+FS) under the same levers
# -----------------------------
event_df, seg_df = compute_segment_summaries(
    event_df_raw, GOE_pct, base_scale, fall_value, type_scales=type_scales, rot_scales=rot_scales
)

summary_df = seg_df.copy()
summary_df["New_Rank"] = summary_df["Total_Segment_Score_new"].rank(method="dense", ascending=False).astype(int)
summary_df["Rank_Change"] = summary_df["New_Rank"] - summary_df["Original_Rank"]
summary_df.sort_values("Original_Rank", inplace=True)

# Overall standings within year+discipline (Short + Free)
year_disc_df_raw = df_sim[(df_sim["Year"] == selected_year) & (df_sim["Discipline"] == disc_code)].copy()
_, year_seg_df = compute_segment_summaries(
    year_disc_df_raw, GOE_pct, base_scale, fall_value, type_scales=type_scales, rot_scales=rot_scales
)

overall = (
    year_seg_df.groupby("Skater_Name", as_index=False)
    .agg(
        Original_Total_Score=("Total_Segment_Score", "sum"),
        New_Total_Score=("Total_Segment_Score_new", "sum"),
        Segments_Count=("Segment", "nunique"),
    )
)

overall_complete = overall[overall["Segments_Count"] >= 2].copy()
overall_complete["Original_Overall_Ranking"] = overall_complete["Original_Total_Score"].rank(
    method="dense", ascending=False
).astype(int)
overall_complete["New_Overall_Ranking"] = overall_complete["New_Total_Score"].rank(
    method="dense", ascending=False
).astype(int)

overall_complete["Total_RankingChange"] = (
    overall_complete["New_Overall_Ranking"] - overall_complete["Original_Overall_Ranking"]
).astype(int)

overall_complete = overall_complete[
    ["Skater_Name", "Original_Total_Score", "New_Total_Score",
     "Original_Overall_Ranking", "New_Overall_Ranking", "Total_RankingChange"]
].copy()

summary_df = summary_df.merge(overall_complete, on="Skater_Name", how="left")

# -----------------------------
# Overview section
# -----------------------------
st.subheader(f"Results – {selected_year} {selected_disc} {selected_seg} (Simulated)")

total_skaters = len(summary_df)
changed_count = int((summary_df["Rank_Change"] != 0).sum())

if changed_count == 0:
    st.markdown("Under the new scoring parameters, **placements remain unchanged** for this segment.")
else:
    up_move = int(abs(summary_df["Rank_Change"].min())) if summary_df["Rank_Change"].min() < 0 else 0
    down_move = int(summary_df["Rank_Change"].max()) if summary_df["Rank_Change"].max() > 0 else 0
    st.markdown(
        f"Under these rules, **{changed_count} of {total_skaters} skaters** would change placement. "
        f"The largest gain is **↑{up_move}** place(s), and the largest drop is **↓{down_move}** place(s)."
    )

summary_display = summary_df.copy()

if "Nation" in event_df.columns:
    nations = event_df.groupby("Skater_Name")["Nation"].first().reset_index()
    summary_display = summary_display.merge(nations, on="Skater_Name", how="left")
    summary_display["Name"] = summary_display["Skater_Name"] + " (" + summary_display["Nation"].astype(str) + ")"
else:
    summary_display["Name"] = summary_display["Skater_Name"]

summary_display = summary_display[
    [
        "Name",
        "Total_Segment_Score",
        "Total_Segment_Score_new",
        "Original_Rank",
        "New_Rank",
        "Rank_Change",
        "Original_Total_Score",
        "New_Total_Score",
        "Original_Overall_Ranking",
        "New_Overall_Ranking",
        "Total_RankingChange",
    ]
].copy()

summary_display.rename(
    columns={
        "Original_Rank": "OriginalRank",
        "New_Rank": "NewRank",
        "Total_Segment_Score": "OrigScore",
        "Total_Segment_Score_new": "New Score",
        "Rank_Change": "ProgramRankChange",
        "Original_Total_Score": "OrigTotalScore",
        "New_Total_Score": "NewTotalScore",
        "Original_Overall_Ranking": "OrigOverallRank",
        "New_Overall_Ranking": "NewOverallRank",
        "Total_RankingChange": "TotalRankChange",
    },
    inplace=True,
)

for c in ["OrigScore", "New Score", "OrigTotalScore", "NewTotalScore"]:
    summary_display[c] = pd.to_numeric(summary_display[c], errors="coerce").round(2)

for c in ["OrigOverallRank", "NewOverallRank", "TotalRankChange"]:
    summary_display[c] = pd.to_numeric(summary_display[c], errors="coerce")

def _color_rank_change(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: red; font-weight: bold;"
    if val < 0:
        return "color: green; font-weight: bold;"
    return ""

def _format_rank_change(val):
    if pd.isna(val):
        return "–"
    val = int(val)
    return f"↑{abs(val)}" if val < 0 else (f"↓{val}" if val > 0 else "–")

def _fmt_rank_or_dash(v):
    if pd.isna(v):
        return "–"
    return f"{int(v)}"

styled_summary = (
    summary_display.style
    .applymap(_color_rank_change, subset=["ProgramRankChange", "TotalRankChange"])
    .format(
        {
            "OrigScore": "{:.2f}",
            "New Score": "{:.2f}",
            "OrigTotalScore": lambda v: "–" if pd.isna(v) else f"{float(v):.2f}",
            "NewTotalScore": lambda v: "–" if pd.isna(v) else f"{float(v):.2f}",
            "ProgramRankChange": _format_rank_change,
            "OrigOverallRank": _fmt_rank_or_dash,
            "NewOverallRank": _fmt_rank_or_dash,
            "TotalRankChange": _format_rank_change,
        }
    )
)

st.dataframe(styled_summary, use_container_width=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Overall Results", "Skater Breakdown"])

with tab1:
    skater_order = summary_df.sort_values("Original_Rank")["Skater_Name"].tolist()
    filtered_events = event_df[event_df["Skater_Name"].isin(skater_order)].copy()

    elem_mix = (
        filtered_events.groupby(["Skater_Name", "Element_Type_sim"])["Element_Score_new"]
        .sum()
        .reset_index()
        .rename(columns={"Element_Type_sim": "Element_Type"})
    )

    comp_mix = (
        summary_df[summary_df["Skater_Name"].isin(skater_order)][["Skater_Name", "Total_Component_Score"]]
        .rename(columns={"Total_Component_Score": "Element_Score_new"})
        .assign(Element_Type="Components")
    )

    ded_mix = (
        summary_df[summary_df["Skater_Name"].isin(skater_order)][["Skater_Name", "Total_Deductions_new"]]
        .rename(columns={"Total_Deductions_new": "Element_Score_new"})
        .assign(Element_Type="Deductions")
    )
    ded_mix["Element_Score_new"] = ded_mix["Element_Score_new"].abs()

    elem_mix = pd.concat([elem_mix, comp_mix, ded_mix], ignore_index=True)

    elem_mix["Signed_Value"] = np.where(
        elem_mix["Element_Type"] == "Deductions",
        -elem_mix["Element_Score_new"],
        elem_mix["Element_Score_new"],
    )

    total_mix = (
        elem_mix.groupby("Skater_Name")["Element_Score_new"]
        .sum()
        .reset_index()
        .rename(columns={"Element_Score_new": "Total_With_Components"})
    )
    elem_mix = elem_mix.merge(total_mix, on="Skater_Name", how="left")
    elem_mix["Percent"] = elem_mix["Element_Score_new"] / elem_mix["Total_With_Components"]

    type_domain = sorted(elem_mix["Element_Type"].unique().tolist())
    chart_height = max(600, 24 * len(skater_order))

    mix_chart = (
        alt.Chart(elem_mix)
        .mark_bar()
        .encode(
            y=alt.Y("Skater_Name:N", sort=skater_order, axis=alt.Axis(title="Skater")),
            x=alt.X("Element_Score_new:Q", stack="normalize", axis=alt.Axis(format="%", title="Share of total score")),
            color=alt.Color("Element_Type:N", title="Element type", scale=alt.Scale(domain=type_domain)),
            tooltip=[
                alt.Tooltip("Skater_Name:N", title="Skater"),
                alt.Tooltip("Element_Type:N", title="Element type"),
                alt.Tooltip("Signed_Value:Q", title="Points", format=".2f"),
                alt.Tooltip("Percent:Q", title="Share (of magnitudes)", format=".1%"),
            ],
        )
        .properties(height=chart_height, width=900)
    )

    components.html(mix_chart.to_html(), height=min(chart_height + 120, 1200), scrolling=True)
    st.caption("Element + component composition of each skater's new total (100% stacked). Scroll for full list.")

with tab2:
    skaters = summary_df.sort_values("Original_Rank")["Skater_Name"].tolist()
    selected_skater = st.selectbox("Select a Skater", skaters)

    skater_sum = summary_df[summary_df["Skater_Name"] == selected_skater].iloc[0]

    orig_score = float(skater_sum["Total_Segment_Score"])
    new_score = float(skater_sum["Total_Segment_Score_new"])
    orig_rank = int(skater_sum["Original_Rank"])
    new_rank = int(skater_sum["New_Rank"])
    rank_change = new_rank - orig_rank

    col1, col2 = st.columns(2)
    col1.metric("Program Score (New)", f"{new_score:.2f}", delta=f"{(new_score - orig_score):+.2f}")

    if rank_change < 0:
        rank_note = f"↑{orig_rank - new_rank} place(s)"
    elif rank_change > 0:
        rank_note = f"↓{new_rank - orig_rank} place(s)"
    else:
        rank_note = "no change"
    col2.markdown(f"**Program Rank:** {orig_rank} → {new_rank} ({rank_note})")

    skater_elems = event_df[event_df["Skater_Name"] == selected_skater].copy()

    if "Element_Number" in skater_elems.columns:
        skater_elems["Element_Number"] = pd.to_numeric(skater_elems["Element_Number"], errors="coerce")
        skater_elems.sort_values("Element_Number", inplace=True)

    for col in ["Element_Score", "Base_Value", "GOE_Mid7_Avg", "Element_Score_new", "Jump_Rotation"]:
        if col in skater_elems.columns:
            skater_elems[col] = pd.to_numeric(skater_elems[col], errors="coerce")

    breakdown_cols = [
        "Element_Name", "Element_Type_sim", "Jump_Rotation",
        "Base_Value", "GOE_Mid7_Avg", "Element_Score", "Element_Score_new"
    ]
    breakdown_df = skater_elems[breakdown_cols].copy()

    breakdown_df.rename(
        columns={
            "Element_Name": "Element",
            "Element_Type_sim": "Type",
            "Jump_Rotation": "Rot",
            "Base_Value": "Base Value",
            "GOE_Mid7_Avg": "GOE avg",
            "Element_Score": "Original Score",
            "Element_Score_new": "New Score",
        },
        inplace=True,
    )

    breakdown_df["Base Value"] = breakdown_df["Base Value"].round(2)
    breakdown_df["GOE avg"] = breakdown_df["GOE avg"].round(3)
    breakdown_df["Original Score"] = breakdown_df["Original Score"].round(2)
    breakdown_df["New Score"] = breakdown_df["New Score"].round(2)
    breakdown_df["Change"] = (breakdown_df["New Score"] - breakdown_df["Original Score"]).round(2)

    # show rotations as integers, dash otherwise
    breakdown_df["Rot"] = breakdown_df["Rot"].apply(lambda v: "–" if pd.isna(v) else str(int(v)))

    def _color_elem_change(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: green;"
        if val < 0:
            return "color: red;"
        return ""

    breakdown_style = (
        breakdown_df.style
        .applymap(_color_elem_change, subset=["Change"])
        .format({"Change": lambda v: f"{v:+.2f}" if pd.notna(v) else "–"})
    )

    st.dataframe(breakdown_style, use_container_width=True, hide_index=True)

    falls = int(skater_sum["Falls"])
    if falls > 0 and float(fall_value) != 1.0:
        original_fall_part = -falls * 1.0
        new_fall_part = -falls * float(fall_value)
        st.write(
            f"*This skater had {falls} fall(s) (inferred from reconciled deductions). "
            f"Fall portion was {original_fall_part:.1f}, now {new_fall_part:.1f} under the new fall rule.*"
        )

    diff_chart = (
        alt.Chart(breakdown_df)
        .mark_bar()
        .encode(
            y=alt.Y("Element:N", sort=None, title="Element"),
            x=alt.X("Change:Q", title="Score Change"),
            color=alt.condition(alt.datum.Change > 0, alt.value("#36a64f"), alt.value("#d94c3d")),
            tooltip=["Element", "Type", "Rot", "Original Score", "New Score", "Change"],
        )
        .properties(height=320)
    )
    st.altair_chart(diff_chart, use_container_width=True)
