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
    "Adjust the parameters in the sidebar (GOE scaling, base value scale, fall logic, and SOV levers) "
    "to simulate how new rules would affect scores and placements. Select a competition segment, and view the "
    "recalculated rankings and each skater’s element-by-element score breakdown under the new rules."
)

# -----------------------------
# Data loading
# -----------------------------
def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

@st.cache_data
def load_data():
    base_df = None
    sim_df = None

    base_path = _first_existing("base_value.csv", "/mnt/data/base_value.csv")
    sim_path  = _first_existing("isu_simulation.csv", "/mnt/data/isu_simulation.csv")

    if base_path:
        base_df = pd.read_csv(base_path)
    if sim_path:
        sim_df = pd.read_csv(sim_path)

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

has_info_col = "Info_Column" in df_sim.columns

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
    "GOE value per ±1 GOE (percent of GOE reference base)",
    min_value=5, max_value=20, value=10, step=1,
    help="Your dataset’s GOE points are consistent with 10% per GOE step. This slider changes that %."
)

base_scale = st.sidebar.slider(
    "Base Value Scale Factor (global)",
    min_value=0.80, max_value=1.50, value=1.00, step=0.01,
    help="Global multiplier on all element base values. (1.00 = no change)"
)

fall_value = st.sidebar.slider(
    "Fall Deduction (points per fall)",
    min_value=0, max_value=3, value=1, step=1,
    help="Points deducted for each fall. (1.0 matches how this dataset is constructed.)"
)

# -----------------------------
# NEW: Fall rule toggle (BV -50% and NO -1 fall deduction)
# -----------------------------
with st.sidebar.expander("Fall rule experiment", expanded=False):
    fall_bv_penalty_mode = st.checkbox(
        "If an element has a fall: cut that element’s base value by 50% AND remove the -1 fall deduction",
        value=False
    )
    st.caption(
        "When enabled: any element whose Info column contains `F` gets a 0.50 multiplier applied to base value "
        "(and proportional GOE). The fall deduction is set to 0. Other (non-fall) deductions are left unchanged."
        + ("" if has_info_col else " (Note: your file has no Info_Column; fall detection will fall back to parsing Element_Name.)")
    )

# -----------------------------
# NEW: Under-rotation call levers (±20% on BV for q / < / <<)
# -----------------------------
with st.sidebar.expander("Jump call scaling: q / < / << (±20% BV sensitivity)", expanded=False):
    st.caption(
        "Apply a separate ±% multiplier to base values (and proportional GOE) for elements with rotation calls. "
        "Detection uses the protocol Info column when available."
    )
    q_adj_pct = st.slider("q adjustment (%)", min_value=-20, max_value=20, value=0, step=1)
    lt_adj_pct = st.slider("< adjustment (%)", min_value=-20, max_value=20, value=0, step=1)
    ltlt_adj_pct = st.slider("<< adjustment (%)", min_value=-20, max_value=20, value=0, step=1)

ur_scales = {
    "q": 1.0 + (q_adj_pct / 100.0),
    "<": 1.0 + (lt_adj_pct / 100.0),
    "<<": 1.0 + (ltlt_adj_pct / 100.0),
}

# -----------------------------
# SOV adjustments by element type (±20%)
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
# Jump rotation levers (±20%)
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

def parse_jump_rotation_bucket(element_name: str):
    """
    For a *jump element row*, infer rotation bucket (1..5) by scanning jump tokens.
    We take the MAX rotation found in the element string (e.g., 4T+3T -> 4).
    """
    if element_name is None or (isinstance(element_name, float) and np.isnan(element_name)):
        return np.nan

    s = str(element_name).strip()
    if not s:
        return np.nan

    s = s.replace(" ", "")
    s = re.sub(r"\+REP", "", s, flags=re.IGNORECASE)

    parts = re.split(r"\+", s)
    rots = []
    for p in parts:
        # remove call annotations (<, <<, !, q, e, etc.) safely
        p2 = re.sub(r"[^0-9A-Za-z]", "", p)
        m = re.match(r"^([1-5])", p2)
        if m:
            rots.append(int(m.group(1)))

    return float(max(rots)) if rots else np.nan

def _type_from_dataset(df: pd.DataFrame) -> pd.Series:
    """
    Prefer the dataset's Element_Type if present; otherwise fall back to string heuristics.
    Returns categories in: Jump / Spin / Step / Choreo / (other Title Cased).
    """
    if "Element_Type" in df.columns:
        s = df["Element_Type"].astype(str).str.strip().str.lower()
        s = s.replace({"jump": "Jump", "spin": "Spin", "step": "Step", "choreo": "Choreo"})
        s = s.where(s.isin(["Jump", "Spin", "Step", "Choreo"]), s.str.title())
        return s

    # fallback heuristic on Element_Name
    name = df["Element_Name"].astype(str).str.replace(" ", "", regex=False).str.lower()
    out = pd.Series(np.where(name.str.contains("chsq"), "Choreo",
                    np.where(name.str.contains("sq"), "Step",
                    np.where(name.str.contains("sp"), "Spin", "Jump"))),
                    index=df.index)
    return out

def _extract_call_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses Info_Column when available; otherwise falls back to Element_Name.
    Outputs boolean columns:
      Call_F, Call_q, Call_<, Call_<<
    """
    if "Info_Column" in df.columns:
        info = df["Info_Column"].fillna("").astype(str)
    else:
        info = df["Element_Name"].fillna("").astype(str)

    # Token boundary = non-alphanumeric (handles separators like space, '|', etc.)
    pat_F = r"(?:^|[^A-Za-z0-9])F(?:[^A-Za-z0-9]|$)"
    pat_q = r"(?:^|[^A-Za-z0-9])q(?:[^A-Za-z0-9]|$)"
    pat_lt = r"(?:^|[^A-Za-z0-9])<(?:[^A-Za-z0-9]|$)"

    call_f = info.str.contains(pat_F, case=False, regex=True, na=False)
    call_q = info.str.contains(pat_q, case=False, regex=True, na=False)
    call_ltlt = info.str.contains("<<", regex=False, na=False)
    call_lt = info.str.contains(pat_lt, regex=True, na=False) & (~call_ltlt)

    return pd.DataFrame(
        {
            "Call_F": call_f,
            "Call_q": call_q,
            "Call_<": call_lt,
            "Call_<<": call_ltlt,
        },
        index=df.index,
    )

def compute_segment_summaries(
    df_in: pd.DataFrame,
    goe_pct: float,
    bv_scale: float,
    fall_pts: float,
    *,
    type_scales=None,
    rot_scales=None,
    ur_scales=None,
    fall_bv_penalty_mode: bool = False,
    fall_bv_multiplier: float = 0.50
):
    """
    Returns:
      df_elements: element-level with Base_Value_new / GOE_points_new / Element_Score_new (+ audit columns)
      df_seg:      skater+segment summary with Total_Segment_Score_new plus reconciled deduction pieces
    """
    if type_scales is None:
        type_scales = {}
    if rot_scales is None:
        rot_scales = {}
    if ur_scales is None:
        ur_scales = {"q": 1.0, "<": 1.0, "<<": 1.0}

    df = df_in.copy()
    df = _ensure_bool_second_half(df)

    # numeric
    for col in [
        "Base_Value", "GOE", "GOE_Mid7_Avg", "Element_Score",
        "Total_Element_Score", "Total_Component_Score", "Total_Deductions", "Total_Segment_Score",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # classify element type
    df["Element_Type_sim"] = _type_from_dataset(df)
    df["Type_Scale"] = df["Element_Type_sim"].map(type_scales).fillna(1.0).astype(float)

    # rotation bucket ONLY for jumps
    jump_mask = df["Element_Type_sim"].eq("Jump")
    df["Jump_Rotation"] = np.nan
    if jump_mask.any():
        df.loc[jump_mask, "Jump_Rotation"] = df.loc[jump_mask, "Element_Name"].apply(parse_jump_rotation_bucket)
    df["Rotation_Scale"] = df["Jump_Rotation"].map(rot_scales).fillna(1.0).astype(float)

    # call flags: F / q / < / <<
    call_flags = _extract_call_flags(df)
    df = pd.concat([df, call_flags], axis=1)

    # Under-rotation call scaling (multiplicative)
    ur = np.ones(len(df), dtype=float)
    ur *= np.where(df["Call_q"], float(ur_scales.get("q", 1.0)), 1.0)
    ur *= np.where(df["Call_<<"], float(ur_scales.get("<<", 1.0)),
                   np.where(df["Call_<"], float(ur_scales.get("<", 1.0)), 1.0))
    df["UR_Scale"] = ur

    # Fall BV penalty (when enabled): element gets 0.50 multiplier (and proportional GOE), and fall deductions removed
    if fall_bv_penalty_mode:
        df["Fall_BV_Scale"] = np.where(df["Call_F"], float(fall_bv_multiplier), 1.0).astype(float)
        fall_pts_effective = 0.0
    else:
        df["Fall_BV_Scale"] = 1.0
        fall_pts_effective = float(fall_pts)

    # combined base scale used per-row
    df["BV_Scale_Used"] = (
        float(bv_scale) * df["Type_Scale"] * df["Rotation_Scale"] * df["UR_Scale"] * df["Fall_BV_Scale"]
    ).astype(float)

    # base value scaling
    df["Base_Value_new"] = (df["Base_Value"] * df["BV_Scale_Used"]).astype(float)

    # GOE scaling
    is_chsq = df["Element_Name"].astype(str).str.contains("chsq", case=False, na=False)

    # For most elements: scale protocol GOE proportionally with BV changes.
    # Your dataset's GOE column is already "points under 10% per GOE step", so this rescales cleanly.
    df["GOE_points_new"] = (
        df["GOE"].fillna(0.0) * (float(goe_pct) / 10.0) * df["BV_Scale_Used"]
    ).astype(float)

    # For ChSq: fixed-step approach; apply only the Choreo type lever + fall BV penalty (if enabled)
    df.loc[is_chsq, "GOE_points_new"] = (
        df.loc[is_chsq, "GOE_Mid7_Avg"].fillna(0.0) * CHSQ_STEP_POINTS
        * df.loc[is_chsq, "Type_Scale"]
        * df.loc[is_chsq, "Fall_BV_Scale"]
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
            Falls=("Call_F", "sum"),
        )
    )
    seg["Falls"] = seg["Falls"].fillna(0).astype(int)

    # audit: reconciliation (should be ~ equal to Total_Deductions)
    seg["Deductions_Effective"] = (
        seg["Total_Segment_Score"] - seg["Total_Element_Score"] - seg["Total_Component_Score"]
    ).round(2)

    # Keep non-fall deductions fixed using element-level fall detection:
    # Total_Deductions = NonFall + (-Falls*1.0)  => NonFall = Total_Deductions + Falls
    seg["NonFall_Deductions"] = (seg["Total_Deductions"] + seg["Falls"] * 1.0).round(2)

    # New deductions: keep non-fall same, change only the fall part
    seg["Total_Deductions_new"] = (seg["NonFall_Deductions"] - seg["Falls"] * float(fall_pts_effective)).round(2)

    seg["Total_Segment_Score_new"] = (
        seg["Total_Element_Score_new"] + seg["Total_Component_Score"] + seg["Total_Deductions_new"]
    ).round(2)

    return df, seg


# -----------------------------
# Compute selected program + overall (SP+FS) under the same levers
# -----------------------------
event_df, seg_df = compute_segment_summaries(
    event_df_raw,
    GOE_pct,
    base_scale,
    fall_value,
    type_scales=type_scales,
    rot_scales=rot_scales,
    ur_scales=ur_scales,
    fall_bv_penalty_mode=fall_bv_penalty_mode,
    fall_bv_multiplier=0.50,
)

summary_df = seg_df.copy()
summary_df["New_Rank"] = summary_df["Total_Segment_Score_new"].rank(method="dense", ascending=False).astype(int)
summary_df["Rank_Change"] = summary_df["New_Rank"] - summary_df["Original_Rank"]
summary_df.sort_values("Original_Rank", inplace=True)

# Overall standings within year+discipline (Short + Free)
year_disc_df_raw = df_sim[(df_sim["Year"] == selected_year) & (df_sim["Discipline"] == disc_code)].copy()
_, year_seg_df = compute_segment_summaries(
    year_disc_df_raw,
    GOE_pct,
    base_scale,
    fall_value,
    type_scales=type_scales,
    rot_scales=rot_scales,
    ur_scales=ur_scales,
    fall_bv_penalty_mode=fall_bv_penalty_mode,
    fall_bv_multiplier=0.50,
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

    for col in [
        "Element_Score", "Base_Value", "GOE", "GOE_Mid7_Avg",
        "Base_Value_new", "GOE_points_new", "Element_Score_new", "Jump_Rotation", "BV_Scale_Used"
    ]:
        if col in skater_elems.columns:
            skater_elems[col] = pd.to_numeric(skater_elems[col], errors="coerce")

    breakdown_cols = [
        "Element_Name", "Element_Type_sim", "Jump_Rotation",
        "Info_Column" if "Info_Column" in skater_elems.columns else None,
        "Base_Value", "GOE", "Element_Score",
        "Base_Value_new", "GOE_points_new", "Element_Score_new",
        "BV_Scale_Used",
    ]
    breakdown_cols = [c for c in breakdown_cols if c is not None and c in skater_elems.columns]
    breakdown_df = skater_elems[breakdown_cols].copy()

    rename_map = {
        "Element_Name": "Element",
        "Element_Type_sim": "Type",
        "Jump_Rotation": "Rot",
        "Info_Column": "Info",
        "Base_Value": "Base Value (orig)",
        "GOE": "GOE pts (orig)",
        "Element_Score": "Original Score",
        "Base_Value_new": "Base Value (new)",
        "GOE_points_new": "GOE pts (new)",
        "Element_Score_new": "New Score",
        "BV_Scale_Used": "BV Mult",
    }
    breakdown_df.rename(columns=rename_map, inplace=True)

    if "Rot" in breakdown_df.columns:
        breakdown_df["Rot"] = breakdown_df["Rot"].apply(lambda v: "–" if pd.isna(v) else str(int(v)))

    for c in ["Base Value (orig)", "GOE pts (orig)", "Original Score", "Base Value (new)", "GOE pts (new)", "New Score", "BV Mult"]:
        if c in breakdown_df.columns:
            breakdown_df[c] = pd.to_numeric(breakdown_df[c], errors="coerce")

    if "BV Mult" in breakdown_df.columns:
        breakdown_df["BV Mult"] = breakdown_df["BV Mult"].round(3)

    for c in ["Base Value (orig)", "GOE pts (orig)", "Original Score", "Base Value (new)", "GOE pts (new)", "New Score"]:
        if c in breakdown_df.columns:
            breakdown_df[c] = breakdown_df[c].round(2)

    breakdown_df["Change"] = (breakdown_df["New Score"] - breakdown_df["Original Score"]).round(2)

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

    falls = int(skater_sum.get("Falls", 0))
    if fall_bv_penalty_mode:
        if falls > 0:
            st.write(
                f"*This skater has **{falls}** element-level fall(s) (from the protocol Info column). "
                f"Under the alternative rule: fall deductions are removed, and each fallen element’s BV (and proportional GOE) is multiplied by 0.50.*"
            )
    else:
        if falls > 0 and float(fall_value) != 1.0:
            original_fall_part = -falls * 1.0
            new_fall_part = -falls * float(fall_value)
            st.write(
                f"*This skater has **{falls}** element-level fall(s) (from the protocol Info column). "
                f"Fall portion was {original_fall_part:.1f}, now {new_fall_part:.1f} under the new fall-deduction setting. "
                f"Other deductions are held constant.*"
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
