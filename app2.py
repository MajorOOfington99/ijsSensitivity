import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt

# Set page layout and title
st.set_page_config(layout="wide", page_title="ISU Score Simulation Dashboard")
st.title("ISU Score Simulation Dashboard")

# Introductory text
st.markdown("This dashboard lets you test **figure skating scoring rule changes** on historical competition data. "
            "Adjust the parameters in the sidebar (GOE scaling, base value scale, fall deduction) to simulate how new rules would affect scores and placements. "
            "Select a competition segment, and view the recalculated rankings and each skater’s element-by-element score breakdown under the new rules.")

# Load data (cached to avoid re-reading on every interaction)
@st.cache_data
def load_data():
    base_df = pd.read_csv("base_value.csv")
    sim_df = pd.read_csv("isu_simulation.csv")
    return base_df, sim_df

base_values, df_sim = load_data()

# Sidebar - Event selection
years = sorted(df_sim["Year"].unique().tolist())
disciplines = {"Men": "M", "Women": "W", "Pairs": "P"}
segments = {"Short Program": "Short", "Free Skate": "Free"}

st.sidebar.header("Select Competition Segment")
selected_year = st.sidebar.selectbox("Year", years, index=len(years)-1)  # default to latest year
selected_disc = st.sidebar.selectbox("Discipline", list(disciplines.keys()), index=0)
selected_seg = st.sidebar.selectbox("Segment", list(segments.keys()), index=1)

# Filter the data for the chosen event (year, discipline, segment)
event_code = (df_sim["Year"] == selected_year) & (df_sim["Discipline"] == disciplines[selected_disc]) & (df_sim["Segment"] == segments[selected_seg])
event_df = df_sim[event_code].copy()
if event_df.empty:
    st.error("No data available for the selected event.")
    st.stop()

# Sidebar - Simulation parameters
st.sidebar.header("Simulation Parameters")
GOE_pct = st.sidebar.slider("GOE value per ±1 GOE (percent of base)", min_value=5, max_value=20, value=10, step=1,
                             help="Percentage of base value for each +1 or -1 GOE. (10% is the current rule):contentReference[oaicite:1]{index=1}")
base_scale = st.sidebar.slider("Base Value Scale Factor", min_value=0.8, max_value=1.5, value=1.0, step=0.01,
                                help="Global multiplier on all element base values. (1.00 = no change)")
fall_value = st.sidebar.slider("Fall Deduction (points per fall)", min_value=0, max_value=3, value=1, step=1,
                                help="Points deducted for each fall. (1.0 is the current rule):contentReference[oaicite:2]{index=2}")

# Recalculate scores under new parameters
# Scale base values
event_df["Base_Value_new"] = event_df["Base_Value"] * base_scale
event_df["GOE_Base_new"] = event_df["GOE_Base"] * base_scale
# (Note: Base_Value in the data already includes second-half 1.1 bonuses where applicable:contentReference[oaicite:3]{index=3}, 
# and GOE_Base is the base of the most difficult jump in a combo:contentReference[oaicite:4]{index=4}.)

# Compute new GOE points for each element
event_df["GOE_points_new"] = event_df["GOE_Mid7_Avg"] * (GOE_pct / 100.0) * event_df["GOE_Base_new"]
# Round GOE points to two decimals (as in official scoring):contentReference[oaicite:5]{index=5}
event_df["GOE_points_new"] = event_df["GOE_points_new"].round(2)

# New element score = new base value + new GOE points
event_df["Element_Score_new"] = (event_df["Base_Value_new"] + event_df["GOE_points_new"]).round(2)

# Aggregate per skater to get total scores
skater_group = event_df.groupby("Skater_Name", as_index=False)
summary_df = skater_group.agg(
    Total_Element_Score_new=("Element_Score_new", "sum"),
    Total_Element_Score=("Total_Element_Score", "first"),
    Total_Component_Score=("Total_Component_Score", "first"),
    Total_Deductions=("Total_Deductions", "first"),
    Total_Segment_Score=("Total_Segment_Score", "first"),
    Original_Rank=("Rank", "first")
)
# Compute new deductions based on fall count
summary_df["Falls"] = summary_df["Total_Deductions"].abs().round().astype(int)  # assume 1 point per fall originally
summary_df["Total_Deductions_new"] = - summary_df["Falls"] * fall_value
# New total segment score = new element score + original components + new deductions
summary_df["Total_Segment_Score_new"] = summary_df["Total_Element_Score_new"] + summary_df["Total_Component_Score"] + summary_df["Total_Deductions_new"]
# Determine new placement ranks
summary_df["New_Rank"] = summary_df["Total_Segment_Score_new"].rank(method="dense", ascending=False).astype(int)
# Rank change (positive = dropped in placement, negative = moved up)
summary_df["Rank_Change"] = summary_df["New_Rank"] - summary_df["Original_Rank"]

# Sort by original rank for display
summary_df.sort_values("Original_Rank", inplace=True)

# Main results overview
st.subheader(f"Results – {selected_year} {selected_disc} {selected_seg} (Simulated)")
# Summary text about ranking changes
total_skaters = len(summary_df)
changed_count = (summary_df["Rank_Change"] != 0).sum()
if changed_count == 0:
    st.markdown("Under the new scoring parameters, **placements remain unchanged** for this segment.")
else:
    # Determine biggest moves
    up_move = int(abs(summary_df["Rank_Change"].min())) if summary_df["Rank_Change"].min() < 0 else 0  # max places gained
    down_move = int(summary_df["Rank_Change"].max()) if summary_df["Rank_Change"].max() > 0 else 0     # max places dropped
    st.markdown(f"Under these rules, **{changed_count} of {total_skaters} skaters** would change placement. "
                f"The largest gain is **↑{up_move}** place(s), and the largest drop is **↓{down_move}** place(s).")

# Prepare summary table with original vs new scores and ranks
summary_display = summary_df.copy()
# Combine name and nation for clarity
if "Nation" in event_df.columns:
    # Use the first nation value for each skater (all rows for a skater have same Nation)
    nations = event_df.groupby("Skater_Name")["Nation"].first().reset_index()
    summary_display = summary_display.merge(nations, on="Skater_Name", how="left")
    summary_display["Name"] = summary_display["Skater_Name"] + " (" + summary_display["Nation"] + ")"
else:
    summary_display["Name"] = summary_display["Skater_Name"]
# Select and rename columns for display
summary_display = summary_display[["Name", "Original_Rank", "New_Rank", "Total_Segment_Score", "Total_Segment_Score_new", "Rank_Change"]]
summary_display.rename(columns={
    "Original_Rank": "Original Rank",
    "New_Rank": "New Rank",
    "Total_Segment_Score": "Original Score",
    "Total_Segment_Score_new": "New Score",
    "Rank_Change": "Rank Change"
}, inplace=True)
# Round scores to two decimals for display
summary_display["Original Score"] = summary_display["Original Score"].round(2)
summary_display["New Score"] = summary_display["New Score"].round(2)

def classify_element(element_name: str) -> str:
    """Group elements, including pairs-specific types, for mix visualization."""
    name = (element_name or "").replace(" ", "").lower()

    # Detect pairs-specific patterns
    is_pairs = any(tag in name for tag in ["th", "li", "ds", "tw", "pcossp"])
    if is_pairs:
        if "li" in name:
            return "Lift"
        if "ds" in name:
            return "Death Spiral"
        if "tw" in name:
            return "Twist"
        if "sp" in name or "pcossp" in name:
            return "Spin"  # Pair spins
        if name.startswith("chsq") or "chsq" in name:
            return "Choreo"
        if "sq" in name:
            return "Step"
        return "Jump"  # Throw jumps or other pairs jumps

    # Singles classification
    if name.startswith("chsq") or "chsq" in name:
        return "Choreo"
    if "sq" in name:
        return "Step"
    if "sp" in name:
        return "Spin"
    return "Jump"

# Tag each element with a type for mix visualization
event_df["Element_Type"] = event_df["Element_Name"].apply(classify_element)

# Highlight rank changes: up (green), down (red)
def _color_rank_change(val):
    color = ""
    if val > 0:   # positive means rank number increased (placement worse)
        color = "red"
    elif val < 0: # negative means rank number decreased (placement better)
        color = "green"
    return f"color: {color}; font-weight: bold;" if color else ""

def _format_rank_change(val):
    # Format with arrow and number (e.g. ↑2, ↓1, or – for no change)
    return f"↑{abs(int(val))}" if val < 0 else (f"↓{int(val)}" if val > 0 else "–")

# Apply styling
styled_summary = summary_display.style.applymap(_color_rank_change, subset=["Rank Change"])\
                                     .format({"Original Score": "{:.2f}", "New Score": "{:.2f}", "Rank Change": _format_rank_change})
st.dataframe(styled_summary, use_container_width=True)

# Tabs for overall results vs individual breakdown
tab1, tab2 = st.tabs(["Overall Results", "Skater Breakdown"])

with tab1:
    # 100% stacked bar: share of element types plus components for all skaters (labeled)
    skater_order = summary_df.sort_values("Original_Rank")["Skater_Name"].tolist()
    filtered_events = event_df[event_df["Skater_Name"].isin(skater_order)]

    # Element contributions
    elem_mix = filtered_events.groupby(["Skater_Name", "Element_Type"])["Element_Score_new"].sum().reset_index()

    # Components contribution (unchanged by simulation)
    comp_mix = (
        summary_df[summary_df["Skater_Name"].isin(skater_order)][["Skater_Name", "Total_Component_Score"]]
        .rename(columns={"Total_Component_Score": "Element_Score_new"})
        .assign(Element_Type="Components")
    )

    # Deductions (negative) shown as penalty slice; use magnitude for stacking
    ded_mix = (
        summary_df[summary_df["Skater_Name"].isin(skater_order)][["Skater_Name", "Total_Deductions_new"]]
        .rename(columns={"Total_Deductions_new": "Element_Score_new"})
        .assign(
            Element_Type="Deductions",
        )
    )
    ded_mix["Element_Score_new"] = ded_mix["Element_Score_new"].abs()

    elem_mix = pd.concat([elem_mix, comp_mix, ded_mix], ignore_index=True)
    elem_mix["Signed_Value"] = elem_mix.apply(
        lambda r: -r["Element_Score_new"] if r["Element_Type"] == "Deductions" else r["Element_Score_new"], axis=1
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
    chart_height = max(600, 24 * len(skater_order))  # ensure room per athlete
    mix_chart = alt.Chart(elem_mix).mark_bar().encode(
        y=alt.Y("Skater_Name:N", sort=skater_order, axis=alt.Axis(title="Skater")),
        x=alt.X("Element_Score_new:Q", stack="normalize", axis=alt.Axis(format="%", title="Share of total score")),
        color=alt.Color(
            "Element_Type:N",
            title="Element type",
            scale=alt.Scale(domain=type_domain, scheme="category20")
        ),
        tooltip=[
            alt.Tooltip("Skater_Name:N", title="Skater"),
            alt.Tooltip("Element_Type:N", title="Element type"),
            alt.Tooltip("Signed_Value:Q", title="Points", format=".2f"),
            alt.Tooltip("Percent:Q", title="Share (of magnitudes)", format=".1%")
        ]
    ).properties(height=chart_height, width=800)

    # Render with scrolling so labels stay aligned even with many athletes
    components.html(mix_chart.to_html(), height=min(chart_height + 120, 1200), scrolling=True)
    st.caption("Element + component composition of each skater's new total (all skaters; 100% stacked). Scroll for full list.")

with tab2:
    # Skater-specific detailed breakdown
    skaters = summary_df.sort_values("Original_Rank")["Skater_Name"].tolist()
    selected_skater = st.selectbox("Select a Skater", skaters)
    # Get that skater's summary info
    skater_sum = summary_df[summary_df["Skater_Name"] == selected_skater].iloc[0]
    orig_score = skater_sum["Total_Segment_Score"]
    new_score = skater_sum["Total_Segment_Score_new"]
    orig_rank = int(skater_sum["Original_Rank"])
    new_rank = int(skater_sum["New_Rank"])
    rank_change = new_rank - orig_rank
    # Display skater's total score and rank change
    col1, col2 = st.columns(2)
    col1.metric("Total Segment Score (New)", f"{new_score:.2f}", delta=f"{(new_score - orig_score):+.2f}")
    # Show rank change with arrow in text (using Markdown for custom formatting)
    if rank_change < 0:
        rank_note = f"↑{orig_rank - new_rank} place(s)"
    elif rank_change > 0:
        rank_note = f"↓{new_rank - orig_rank} place(s)"
    else:
        rank_note = "no change"
    col2.markdown(f"**Segment Rank:** {orig_rank} → {new_rank} ({rank_note})")

    # Retrieve element-by-element data for the skater
    skater_elems = event_df[event_df["Skater_Name"] == selected_skater].copy()
    skater_elems.sort_values("Element_Number", inplace=True)
    # Build a breakdown table
    breakdown_df = skater_elems[["Element_Name", "Base_Value", "GOE_Mid7_Avg", "Element_Score", "Element_Score_new"]].copy()
    breakdown_df.rename(columns={
        "Element_Name": "Element",
        "Base_Value": "Base Value",
        "GOE_Mid7_Avg": "GOE avg",
        "Element_Score": "Original Score",
        "Element_Score_new": "New Score"
    }, inplace=True)
    # Round values for neat display
    breakdown_df["Base Value"] = breakdown_df["Base Value"].round(2)
    breakdown_df["GOE avg"] = breakdown_df["GOE avg"].round(3)
    breakdown_df["Original Score"] = breakdown_df["Original Score"].round(2)
    breakdown_df["New Score"] = breakdown_df["New Score"].round(2)
    breakdown_df["Change"] = (breakdown_df["New Score"] - breakdown_df["Original Score"]).round(2)
    # Style the breakdown table: highlight positive/negative changes
    def _color_elem_change(val):
        if val > 0:
            return "color: green;"
        elif val < 0:
            return "color: red;"
        return ""
    def _fmt_elem_change(val):
        return f"{val:+.2f}"
    breakdown_style = breakdown_df.style.applymap(_color_elem_change, subset=["Change"])\
                                       .format({"Change": _fmt_elem_change})
    st.table(breakdown_style.hide(axis="index"))
    # If falls affected the score, note the deduction change
    falls = int(abs(skater_sum["Total_Deductions"]))
    if falls > 0 and fall_value != 1:
        original_ded = falls * 1.0
        new_ded = falls * fall_value
        st.write(f"*This skater had {falls} fall(s). Deduction was {original_ded:.1f} point(s), now {new_ded:.1f} under the new rule.*")

    # Bar chart: point difference per element
    diff_chart = alt.Chart(breakdown_df).mark_bar().encode(
        y=alt.Y("Element:N", sort=None, title="Element"),
        x=alt.X("Change:Q", title="Score Change"),
        color=alt.condition(alt.datum.Change > 0, alt.value("#36a64f"), alt.value("#d94c3d")),  # green for positive, red for negative
        tooltip=["Element", "Original Score", "New Score", "Change"]
    ).properties(height=300)
    st.altair_chart(diff_chart, use_container_width=True)
