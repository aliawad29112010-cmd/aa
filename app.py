"""
3-Digit PIN Frequency Dashboard
===============================
An interactive Streamlit + Matplotlib dashboard for 3-digit PIN codes.
It works only with PINs 000-999 and reads data in the "[PIN]: [N]" format,
one entry per line, for example:

    123: 1783558
    666: 204408
    007: 23667

HOW TO INSTALL AND RUN IN VISUAL STUDIO CODE
--------------------------------------------
1. Install Python 3.9 or newer from https://www.python.org/downloads/
   (on Windows, tick "Add python.exe to PATH" in the installer).
2. Open VS Code, choose File > Open Folder, and pick the folder that
   contains this file. Save this file as:  app.py
3. Open the VS Code terminal:  Terminal > New Terminal  (or Ctrl+` ).
4. (Optional but recommended) Create and activate a virtual environment:
       python -m venv .venv
       Windows (PowerShell):   .venv\\Scripts\\Activate.ps1
       macOS / Linux:          source .venv/bin/activate
5. Install the required libraries:
       pip install streamlit pandas matplotlib
6. Start the dashboard:
       streamlit run app.py
   If the "streamlit" command is not found, use:
       python -m streamlit run app.py
7. Your browser opens at http://localhost:8501 automatically.
   Press Ctrl+C in the terminal to stop the app.

DATA SOURCES (choose one in the sidebar)
----------------------------------------
- Load from file : reads a text file in "[PIN]: [N]" format. The default
                   file name is "HIBP 3 digit.txt", placed next to app.py.
- Upload a file  : upload a .txt or .csv file in the same format.
- Simulate data  : generates synthetic demo data (not real PIN statistics).
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend; Streamlit displays the figures.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DEFAULT_FILENAME = "HIBP 3 digit.txt"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = SCRIPT_DIR / DEFAULT_FILENAME

# Matches one "[PIN]: [N]" line: exactly three digits, a colon, then a count.
# Counts may contain thousands separators (for example 1,783,558).
LINE_PATTERN = re.compile(r"^(\d{3})\s*:\s*(\d[\d,]*)$")

BAR_COLOR = "#2a78d6"
HIGHLIGHT_COLOR = "#eb6834"
GRID_COLOR = "#e1e0d9"
TEXT_COLOR = "#33322f"

st.set_page_config(page_title="3-Digit PIN Dashboard", page_icon="🔢", layout="wide")


# ----------------------------------------------------------------------------
# Data loading and simulation
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_pin_text(text: str) -> tuple[pd.DataFrame, int]:
    """Parse '[PIN]: [N]' lines into a DataFrame.

    Returns (dataframe with columns PIN and Count, number of ignored lines).
    Lines that are blank are skipped silently; other lines that do not match
    the format (or whose PIN is not exactly 3 digits) are counted as ignored.
    Duplicate PINs are added together.
    """
    counts: dict[str, int] = {}
    ignored = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_PATTERN.match(line)
        if not match:
            ignored += 1
            continue
        pin = match.group(1)
        count = int(match.group(2).replace(",", ""))
        counts[pin] = counts.get(pin, 0) + count

    df = pd.DataFrame({"PIN": list(counts.keys()), "Count": list(counts.values())})
    return df, ignored


@st.cache_data(show_spinner=False)
def simulate_pins(total_records: int, seed: int) -> pd.DataFrame:
    """Create synthetic counts for all 1,000 three-digit PINs.

    People tend to pick patterns, so repeated digits (111), ascending runs
    (123, 456) and a few famous codes (007, 666) get higher weights. The
    records are then spread across PINs with a multinomial draw.
    """
    rng = np.random.default_rng(seed)
    weights = rng.lognormal(mean=0.0, sigma=0.35, size=1000)  # mild random noise

    for p in range(1000):
        a, b, c = p // 100, (p // 10) % 10, p % 10
        if a == b == c:
            weights[p] *= 15  # repeated digits: 000, 111 ... 999
        elif b == a + 1 and c == b + 1:
            weights[p] *= 12  # ascending runs: 012 ... 789
        elif b == a - 1 and c == b - 1:
            weights[p] *= 6  # descending runs: 210 ... 987
        elif a == c:
            weights[p] *= 1.3  # mirrored: 101, 121 ...

    boosts = {123: 25, 666: 4, 100: 8, 7: 8, 321: 3, 456: 3, 789: 2, 200: 4, 159: 3}
    for pin, factor in boosts.items():
        weights[pin] *= factor

    probabilities = weights / weights.sum()
    counts = rng.multinomial(total_records, probabilities)
    return pd.DataFrame({"PIN": [f"{p:03d}" for p in range(1000)], "Count": counts})


def resolve_path(text: str) -> Path:
    """Return the file path the user typed, trying the app folder first."""
    path = Path(text.strip().strip('"')).expanduser()
    if path.is_absolute():
        return path
    candidate = SCRIPT_DIR / path
    return candidate if candidate.exists() else Path.cwd() / path


# ----------------------------------------------------------------------------
# Calculations
# ----------------------------------------------------------------------------
def build_ranked_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sort PINs by frequency and add rank, share and cumulative share."""
    ranked = df[df["Count"] > 0].sort_values(["Count", "PIN"], ascending=[False, True])
    ranked = ranked.reset_index(drop=True)
    total = ranked["Count"].sum()
    ranked.insert(0, "Rank", ranked.index + 1)
    ranked["Share (%)"] = ranked["Count"] / total * 100
    ranked["Cumulative share (%)"] = ranked["Count"].cumsum() / total * 100
    return ranked


# ----------------------------------------------------------------------------
# Charts (Matplotlib)
# ----------------------------------------------------------------------------
def style_axes(ax) -> None:
    """Apply one clean, minimal look to every chart."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a988f")
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_axisbelow(True)


def draw_top_bar_chart(top: pd.DataFrame, horizontal: bool, log_scale: bool, show_labels: bool):
    """Bar chart of the most frequent PINs, with count and share labels."""
    n = len(top)
    max_count = top["Count"].max()
    colors = [HIGHLIGHT_COLOR if i == 0 else BAR_COLOR for i in range(n)]

    if horizontal:
        fig, ax = plt.subplots(figsize=(9, max(4.0, 0.34 * n + 1.6)))
        # Reverse the order so that rank 1 is drawn at the top.
        y_positions = np.arange(n)[::-1]
        ax.barh(y_positions, top["Count"], color=colors, height=0.7, zorder=3)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top["PIN"])
        for label in ax.get_yticklabels():
            label.set_family("monospace")
        ax.set_xlabel("Frequency" + (" (log scale)" if log_scale else ""))
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        if log_scale:
            ax.set_xscale("log")
            ax.set_xlim(left=max(1, top["Count"].min() * 0.5), right=max_count * 6)
        else:
            ax.set_xlim(0, max_count * 1.28)
        ax.tick_params(axis="y", length=0)
        if show_labels:
            for y, count, share in zip(y_positions, top["Count"], top["Share (%)"]):
                ax.annotate(
                    f"{count:,}  ({share:.2f}%)",
                    xy=(count, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT_COLOR,
                )
    else:
        fig, ax = plt.subplots(figsize=(max(9, 0.5 * n + 3), 5.2))
        x_positions = np.arange(n)
        ax.bar(x_positions, top["Count"], color=colors, width=0.7, zorder=3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(top["PIN"], rotation=90 if n > 15 else 0)
        for label in ax.get_xticklabels():
            label.set_family("monospace")
        ax.set_ylabel("Frequency" + (" (log scale)" if log_scale else ""))
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(1, top["Count"].min() * 0.5), top=max_count * 8)
        else:
            ax.set_ylim(0, max_count * 1.25)
        ax.tick_params(axis="x", length=0)
        if show_labels:
            for x, share in zip(x_positions, top["Share (%)"]):
                ax.annotate(
                    f"{share:.1f}%" if share >= 1 else f"{share:.2f}%",
                    xy=(x, top["Count"].iloc[x]), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR,
                    rotation=90 if n > 15 else 0,
                )

    ax.set_title(f"Top {n} most frequent 3-digit PINs", loc="left", fontsize=12,
                 fontweight="bold", color="#111111")
    style_axes(ax)
    fig.tight_layout()
    return fig


def draw_cumulative_chart(ranked: pd.DataFrame, top_n: int, log_x: bool):
    """Line chart: share of all records covered by the k most frequent PINs."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(ranked["Rank"], ranked["Cumulative share (%)"], color=BAR_COLOR, linewidth=2)
    ax.fill_between(ranked["Rank"], 0, ranked["Cumulative share (%)"], color=BAR_COLOR, alpha=0.10)
    n = min(top_n, len(ranked))
    covered = ranked["Cumulative share (%)"].iloc[n - 1]
    ax.axvline(n, color=HIGHLIGHT_COLOR, linewidth=1)
    ax.plot([n], [covered], "o", color=HIGHLIGHT_COLOR, markersize=7, markeredgecolor="white")
    ax.annotate(f"Top {n}: {covered:.1f}%", xy=(n, covered), xytext=(8, -16),
                textcoords="offset points", fontsize=9, color=TEXT_COLOR)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlim(1, len(ranked))
    ax.set_ylim(0, 100)
    ax.set_xlabel("PIN rank, most frequent first" + (" (log scale)" if log_x else ""))
    ax.set_ylabel("Cumulative share of all records (%)")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_title("How quickly the top PINs add up", loc="left", fontsize=12,
                 fontweight="bold", color="#111111")
    style_axes(ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Sidebar: data source and chart options
# ----------------------------------------------------------------------------
st.sidebar.header("Data")
source_options = ["Load from file", "Upload a file", "Simulate data"]
source = st.sidebar.radio(
    "Data source",
    source_options,
    index=0 if DEFAULT_PATH.exists() else 2,
    help='Files must contain one "[PIN]: [N]" entry per line, for example "123: 1783558".',
)

raw_df = None
ignored_lines = 0
is_simulated = False

if source == "Load from file":
    path_text = st.sidebar.text_input("File path", value=DEFAULT_FILENAME)
    file_path = resolve_path(path_text)
    if file_path.is_file():
        text = file_path.read_text(encoding="utf-8-sig", errors="replace")
        raw_df, ignored_lines = parse_pin_text(text)
    else:
        st.sidebar.error(f"File not found: {file_path}")
elif source == "Upload a file":
    uploaded = st.sidebar.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
    if uploaded is not None:
        text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
        raw_df, ignored_lines = parse_pin_text(text)
else:
    total_records = st.sidebar.number_input(
        "Total simulated records", min_value=1_000, max_value=100_000_000,
        value=1_000_000, step=100_000,
    )
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
    raw_df = simulate_pins(int(total_records), int(seed))
    is_simulated = True

if raw_df is None or raw_df.empty or raw_df["Count"].sum() == 0:
    st.title("3-Digit PIN Frequency Dashboard")
    st.info(
        "No data loaded yet. Choose a data source in the sidebar. "
        'Files need one "[PIN]: [N]" entry per line, for example "123: 1783558".'
    )
    if ignored_lines:
        st.warning(f"{ignored_lines} line(s) were ignored because they did not match the format.")
    st.stop()

if ignored_lines:
    st.sidebar.warning(
        f'{ignored_lines} line(s) ignored (not a 3-digit PIN in "[PIN]: [N]" format).'
    )

st.sidebar.header("Analysis")
all_ranked = build_ranked_table(raw_df)
excluded = st.sidebar.multiselect(
    "Exclude PINs (to look past an outlier)",
    options=all_ranked["PIN"].tolist(),
    default=[],
    help="Excluded PINs are removed before any total or share is calculated.",
)
ranked = build_ranked_table(raw_df[~raw_df["PIN"].isin(excluded)]) if excluded else all_ranked
if ranked.empty:
    st.warning("All PINs are excluded. Remove at least one exclusion in the sidebar.")
    st.stop()

if len(ranked) > 3:
    top_n = st.sidebar.slider("Number of top PINs", min_value=3, max_value=min(50, len(ranked)),
                              value=min(15, len(ranked)))
else:
    top_n = len(ranked)  # Too few PINs for a slider; show all of them.

st.sidebar.header("Chart options")
orientation = st.sidebar.radio("Bar direction", ["Horizontal", "Vertical"], horizontal=True)
log_scale = st.sidebar.checkbox("Logarithmic scale", value=False,
                                help="Helps when one PIN is far more frequent than the others.")
show_labels = st.sidebar.checkbox("Show value labels", value=True)

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("3-Digit PIN Frequency Dashboard")
st.caption('Frequencies of PINs 000 to 999 from "[PIN]: [N]" data. '
           "Shares are relative to the total of the PINs included in the analysis.")
if is_simulated:
    st.info("You are viewing simulated demo data. It is synthetic and does not describe real PINs.")
if excluded:
    st.caption("Excluded PINs: " + ", ".join(excluded))

top = ranked.head(top_n)
total_all = int(ranked["Count"].sum())
top_total = int(top["Count"].sum())
top_share = top_total / total_all * 100
best = ranked.iloc[0]

# Summary metrics, first row: the headline numbers.
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total frequency (all PINs)", f"{total_all:,}")
m2.metric(f"Top {top_n} frequency", f"{top_total:,}")
m3.metric(f"Top {top_n} share", f"{top_share:.1f}%")
m4.metric("Most frequent PIN", str(best["PIN"]), f"{best['Share (%)']:.1f}% of total", delta_color="off")

# Summary metrics, second row: context.
n_distinct = len(ranked)
top10_share = ranked["Share (%)"].head(10).sum()
top100_share = ranked["Share (%)"].head(100).sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Distinct PINs", f"{n_distinct:,}")
c2.metric("Median count per PIN", f"{ranked['Count'].median():,.0f}")
c3.metric("Top 10 share", f"{top10_share:.1f}%")
c4.metric("Top 100 share", f"{top100_share:.1f}%")

st.divider()

tab_bar, tab_cum, tab_table = st.tabs(["Top PINs (bar chart)", "Cumulative share", "Data table"])

with tab_bar:
    fig_bar = draw_top_bar_chart(top, orientation == "Horizontal", log_scale, show_labels)
    st.pyplot(fig_bar)
    plt.close(fig_bar)
    st.caption(
        f"The top {top_n} PINs account for {top_total:,} of {total_all:,} records "
        f"({top_share:.1f}%). The highlighted bar is the most frequent PIN."
    )

with tab_cum:
    log_x = st.checkbox("Logarithmic rank axis", value=True)
    fig_cum = draw_cumulative_chart(ranked, top_n, log_x)
    st.pyplot(fig_cum)
    plt.close(fig_cum)
    r50 = int((ranked["Cumulative share (%)"] < 50).sum()) + 1
    r80 = int((ranked["Cumulative share (%)"] < 80).sum()) + 1
    st.caption(f"{r50} PIN(s) reach 50% of all records and {r80} reach 80%.")

with tab_table:
    table = ranked.copy()
    table["Share (%)"] = table["Share (%)"].round(3)
    table["Cumulative share (%)"] = table["Cumulative share (%)"].round(3)
    st.dataframe(table, hide_index=True)
    st.download_button(
        "Download table as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="pin_frequencies.csv",
        mime="text/csv",
    )
