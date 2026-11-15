
from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ARTIFACT_DIR = APP_DIR / "model_artifacts"
DB_PATH = DATA_DIR / "clubconnect.db"
PALETTE = {
    "lavender": "#DCCFF5",
    "lavender_dark": "#8065B5",
    "pink": "#F3B2CF",
    "yellow": "#F6D978",
    "sky": "#A9DDF5",
    "mint": "#BCE8D5",
    "ink": "#302B43",
    "muted": "#827D91",
    "surface": "#FFFFFF",
    "bg": "#F8F6FC",
}
CHART_COLORS = [PALETTE["lavender_dark"], "#F09FC4", "#80C9EA", "#F0C94F",
                "#9DDCC5", "#B7A3E6", "#F5B38A", "#91B6E8"]

st.set_page_config(
    page_title="ClubConnect AI | Campus Community",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Demo data ----------
CLUBS = [
    ("TechNova", "Technology", "Build, code and explore emerging technologies.", ["Coding", "AI", "Technology"], "💻", "Sky Blue"),
    ("PixelPerfect", "Arts & Design", "A creative space for design, photography and visual storytelling.", ["Design", "Photography", "Creativity"], "🎨", "Pink"),
    ("Rhythm House", "Music & Dance", "Celebrate music, dance, rhythm and campus performances.", ["Music", "Dance", "Performance"], "🎵", "Lavender"),
    ("GreenSphere", "Environment", "Make campus greener through sustainability and volunteering.", ["Environment", "Volunteering", "Nature"], "🌱", "Mint"),
    ("SpeakUp Society", "Leadership", "Build confidence through debates, public speaking and leadership.", ["Leadership", "Debate", "Communication"], "🎤", "Yellow"),
    ("Frame by Frame", "Media", "Create short films, reels, podcasts and campus stories.", ["Video", "Photography", "Media"], "🎬", "Pink"),
    ("Athletica", "Sports", "Bring students together through fitness, games and sports.", ["Sports", "Fitness", "Teamwork"], "🏅", "Sky Blue"),
    ("Readers' Circle", "Literature", "Discover books, writing, poetry and thoughtful conversations.", ["Reading", "Writing", "Literature"], "📚", "Lavender"),
    ("Social Impact Crew", "Community Service", "Turn empathy into action through community projects.", ["Volunteering", "Social Work", "Leadership"], "💗", "Mint"),
    ("BizSpark", "Entrepreneurship", "Explore business ideas, innovation and startup culture.", ["Business", "Entrepreneurship", "Innovation"], "🚀", "Yellow"),
]

INTERESTS = ["Coding", "AI", "Technology", "Design", "Photography", "Creativity", "Music", "Dance",
             "Performance", "Environment", "Volunteering", "Nature", "Leadership", "Debate",
             "Communication", "Video", "Media", "Sports", "Fitness", "Teamwork", "Reading",
             "Writing", "Literature", "Social Work", "Business", "Entrepreneurship", "Innovation"]

@st.cache_data(show_spinner=False)
def build_demo_data(seed: int = 27):
    rng = np.random.default_rng(seed)
    n_students = 180
    names = ["Aarav", "Aanya", "Aditya", "Anaya", "Arjun", "Diya", "Ishaan", "Kavya",
             "Krish", "Meera", "Neha", "Rohan", "Saanvi", "Sahil", "Tanvi", "Vedant",
             "Zoya", "Ira", "Kabir", "Mira"]
    departments = ["BCA AIML", "BCA", "BBA", "B.Tech CSE", "B.Sc IT", "B.Com"]
    years = ["First Year", "Second Year", "Third Year"]
    students = []
    for i in range(n_students):
        name = f"{rng.choice(names)} {chr(65 + (i % 26))}."
        interests = rng.choice(INTERESTS, size=int(rng.integers(2, 6)), replace=False).tolist()
        students.append({
            "student_id": f"STU{i+1001}", "student_name": name,
            "department": rng.choice(departments, p=[.24,.18,.12,.22,.14,.10]),
            "year": rng.choice(years, p=[.34,.33,.33]),
            "email": f"student{i+1001}@aurelia.edu",
            "interests": ", ".join(interests),
            "joined_on": (date.today() - timedelta(days=int(rng.integers(30, 600)))).isoformat(),
        })
    students_df = pd.DataFrame(students)

    clubs = []
    for i, (name, category, desc, tags, emoji, accent) in enumerate(CLUBS):
        clubs.append({"club_id": f"CLB{i+1:02d}", "club_name": name, "category": category,
                      "description": desc, "tags": ", ".join(tags), "emoji": emoji,
                      "accent": accent, "is_active": True,
                      "members": int(rng.integers(18, 66))})
    clubs_df = pd.DataFrame(clubs)

    # Create registrations with realistic interest-to-club affinity.
    regs = []
    reg_id = 1
    for _, s in students_df.iterrows():
        student_tags = set(s["interests"].split(", "))
        weights = []
        for _, c in clubs_df.iterrows():
            club_tags = set(c["tags"].split(", "))
            overlap = len(student_tags & club_tags)
            weights.append(0.35 + overlap * 1.8)
        count = int(rng.choice([0, 1, 2, 3], p=[.12, .46, .32, .10]))
        if count:
            chosen = rng.choice(len(clubs_df), size=min(count, len(clubs_df)), replace=False,
                                p=np.array(weights)/sum(weights))
            for ci in chosen:
                club = clubs_df.iloc[int(ci)]
                joined = date.today() - timedelta(days=int(rng.integers(15, 450)))
                regs.append({"registration_id": f"REG{reg_id:04d}", "student_id": s["student_id"],
                             "club_id": club["club_id"], "registered_on": joined.isoformat(),
                             "status": rng.choice(["Active", "Active", "Active", "Inactive"], p=[.7,.1,.1,.1])})
                reg_id += 1
    registrations_df = pd.DataFrame(regs)

    event_types = ["Workshop", "Competition", "Meetup", "Volunteer Drive", "Performance", "Seminar"]
    event_names = ["AI Discovery Day", "Design Jam", "Campus Open Mic", "Green Campus Drive",
                   "Debate Arena", "Photography Walk", "Code Sprint", "Book & Brew",
                   "Startup Mixer", "Sports Fiesta", "Creative Showcase", "Community Connect"]
    events = []
    for i in range(32):
        club = clubs_df.iloc[int(rng.integers(0, len(clubs_df)))]
        event_date = date.today() + timedelta(days=int(rng.integers(-75, 55)))
        capacity = int(rng.integers(30, 150))
        registered = int(rng.integers(10, capacity + 1))
        attended = int(registered * rng.uniform(.55, .98))
        events.append({"event_id": f"EVT{i+1:03d}", "event_name": rng.choice(event_names),
                       "club_id": club["club_id"], "club_name": club["club_name"],
                       "event_type": rng.choice(event_types), "event_date": event_date.isoformat(),
                       "venue": rng.choice(["Innovation Lab", "Auditorium", "Seminar Hall 2", "Open Lawn", "Sports Arena"]),
                       "capacity": capacity, "registered": registered, "attended": attended,
                       "status": "Upcoming" if event_date >= date.today() else "Completed"})
    events_df = pd.DataFrame(events)

    # Event attendance records are synthetic and used for engagement analytics.
    attendance = []
    for _, ev in events_df.iterrows():
        pool = students_df["student_id"].to_numpy()
        k = min(int(ev["attended"]), len(pool))
        for sid in rng.choice(pool, size=k, replace=False):
            attendance.append({"event_id": ev["event_id"], "student_id": sid, "attended": True})
    attendance_df = pd.DataFrame(attendance)
    return students_df, clubs_df, registrations_df, events_df, attendance_df

def ensure_database():
    DATA_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        return
    students, clubs, regs, events, attendance = build_demo_data()
    with sqlite3.connect(DB_PATH) as conn:
        for name, frame in [("students", students), ("clubs", clubs), ("registrations", regs),
                            ("events", events), ("attendance", attendance)]:
            frame.to_sql(name, conn, index=False, if_exists="replace")

@st.cache_data(show_spinner=False)
def load_data():
    ensure_database()
    with sqlite3.connect(DB_PATH) as conn:
        tables = {name: pd.read_sql_query(f"SELECT * FROM {name}", conn)
                  for name in ["students", "clubs", "registrations", "events", "attendance"]}
    return tables

def save_attendance(event_id: str, attended: int):
    # Demo action: update event attendance count safely with a parameterized query.
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE events SET attended = ? WHERE event_id = ?", (int(attended), event_id))
    load_data.clear()

def train_models(students: pd.DataFrame, clubs: pd.DataFrame, regs: pd.DataFrame):
    # K-means segments students using club count and participation frequency.
    counts = regs.groupby("student_id").size().rename("club_count")
    active = regs[regs["status"] == "Active"].groupby("student_id").size().rename("active_clubs")
    features = students[["student_id"]].merge(counts, on="student_id", how="left").merge(
        active, on="student_id", how="left").fillna(0)
    features["engagement_score"] = (features["active_clubs"] * 25 + features["club_count"] * 10).clip(0, 100)
    X = features[["club_count", "active_clubs", "engagement_score"]].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_clusters = min(3, max(2, len(features) // 20))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0
    features["segment_id"] = labels
    centers = features.groupby("segment_id")["engagement_score"].mean().sort_values()
    mapping = {int(k): ["Needs Encouragement", "Growing Engagement", "Highly Engaged"][i]
               for i, k in enumerate(centers.index)}
    features["segment"] = features["segment_id"].map(mapping)
    metadata = {"model": "KMeans", "version": "1.0.0", "trained_at": datetime.now().isoformat(timespec="seconds"),
                "n_students": int(len(features)), "n_clusters": int(n_clusters),
                "silhouette_score": round(score, 4), "features": list(X.columns)}
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return features, metadata

def recommend_clubs(student_interests: List[str], clubs: pd.DataFrame, top_n: int = 4):
    if not student_interests:
        return pd.DataFrame()
    vocab = sorted(set(student_interests).union(*(set(c.split(", ")) for c in clubs["tags"])))
    student_vec = np.array([[1 if t in student_interests else 0 for t in vocab]])
    club_vecs = np.array([[1 if t in set(tags.split(", ")) else 0 for t in vocab] for tags in clubs["tags"]])
    scores = cosine_similarity(student_vec, club_vecs)[0]
    result = clubs.copy()
    result["match_score"] = scores
    return result.sort_values("match_score", ascending=False).head(top_n)

def metric_card(label: str, value, note: str, color: str):
    st.markdown(
        f"""<div class="metric-card"><div class="metric-top"><span>{label}</span>
        <span class="metric-dot" style="background:{color}"></span></div>
        <div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )

def section_heading(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-heading"><h2>{title}</h2><p>{subtitle}</p></div>',
                unsafe_allow_html=True)

def chart_layout(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito Sans, sans-serif", color=PALETTE["ink"], size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#EEEAF5")
    fig.update_yaxes(gridcolor="#F0EDF5", zeroline=False)
    return fig

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;500;600;700;800&family=Poppins:wght@500;600;700;800&display=swap');
:root { --ink:#302B43; --muted:#827D91; --lav:#8065B5; --bg:#F8F6FC; }
html, body, [class*="css"] { font-family:'Nunito Sans', sans-serif; color:var(--ink); }
.stApp { background:var(--bg); }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #EEEAF5; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
.block-container { padding-top:1.45rem; padding-bottom:2rem; max-width:1500px; }
h1,h2,h3 { font-family:'Poppins', sans-serif !important; color:var(--ink); }
h1 { font-size:2rem !important; font-weight:700 !important; }
h2 { font-size:1.28rem !important; font-weight:650 !important; }
p { color:var(--muted); }
.brand { display:flex; align-items:center; gap:11px; padding:5px 4px 22px 4px; }
.brand-mark { width:43px;height:43px;border-radius:14px;background:linear-gradient(135deg,#DCCFF5,#F3B2CF);
display:flex;align-items:center;justify-content:center;font-size:23px;box-shadow:0 5px 14px #DCCFF566; }
.brand-name { font-family:'Poppins';font-weight:800;font-size:19px;line-height:1.05;color:#7257A7; }
.brand-sub { font-size:9px;letter-spacing:1.6px;color:#A09AAE;font-weight:800;margin-top:4px; }
.topline { display:flex;align-items:center;justify-content:space-between;margin-bottom:18px; }
.eyebrow { color:#907DB5;font-size:11px;font-weight:800;letter-spacing:1.7px;text-transform:uppercase; }
.hero { background:linear-gradient(110deg,#EAE2FA 0%,#FCEAF2 56%,#E6F5FC 100%);
border:1px solid #EEE7F7;border-radius:22px;padding:25px 28px;margin:4px 0 22px 0;position:relative;overflow:hidden; }
.hero h1 { margin:3px 0 7px 0;font-size:27px !important; }
.hero p { margin:0;max-width:680px;font-size:14px; }
.hero-chip { display:inline-block;background:#FFFFFFB8;border:1px solid #FFFFFF;border-radius:30px;padding:6px 11px;font-size:11px;color:#7257A7;font-weight:800;margin-bottom:10px; }
.metric-card { background:#FFFFFF;border:1px solid #EEEAF5;border-radius:17px;padding:17px 18px;min-height:116px;box-shadow:0 5px 18px #4A3C6A08; }
.metric-top { display:flex;justify-content:space-between;align-items:center;color:#817B91;font-size:12px;font-weight:700; }
.metric-dot { width:9px;height:9px;border-radius:50%;display:inline-block; }
.metric-value { font-family:'Poppins';font-size:29px;font-weight:700;color:#332D48;margin-top:9px;line-height:1.1; }
.metric-note { color:#9A94A7;font-size:11px;margin-top:6px; }
.panel { background:#FFFFFF;border:1px solid #EEEAF5;border-radius:18px;padding:18px 19px;box-shadow:0 5px 18px #4A3C6A08; }
.section-heading { margin:20px 0 12px 0; }
.section-heading h2 { margin:0 0 3px 0; }
.section-heading p { margin:0;font-size:12px; }
.club-card { background:#FFFFFF;border:1px solid #EEEAF5;border-radius:17px;padding:17px;height:100%;box-shadow:0 5px 18px #4A3C6A08; }
.club-icon { width:43px;height:43px;border-radius:14px;background:#F1EBFB;display:flex;align-items:center;justify-content:center;font-size:22px;margin-bottom:12px; }
.club-title { font-family:'Poppins';font-size:15px;font-weight:700;color:#352F49;margin-bottom:4px; }
.club-desc { font-size:12px;color:#888197;min-height:48px; }
.tag { display:inline-block;border-radius:20px;background:#F4F0FB;color:#7963A4;padding:4px 9px;font-size:10px;font-weight:700;margin:3px 3px 0 0; }
.small-muted { color:#948EA2;font-size:11px; }
div[data-testid="stButton"] button { border-radius:11px;border:1px solid #E5DDF2;background:#FFFFFF;color:#7057A5;font-weight:800; }
div[data-testid="stButton"] button:hover { border-color:#A58DD1;background:#F5F0FC;color:#5F478E; }
.stButton>button[kind="primary"] { background:#8065B5;color:white;border:0; }
[data-testid="stDataFrame"] { border:1px solid #EEEAF5;border-radius:12px;overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; }
hr { border-color:#EEEAF5; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- App boot ----------
ensure_database()
data = load_data()
students, clubs, regs, events, attendance = (data[k].copy() for k in
    ["students", "clubs", "registrations", "events", "attendance"])
segments, model_meta = train_models(students, clubs, regs)

with st.sidebar:
    st.image(str(APP_DIR / "assets" / "aurelia_university_logo.png"), use_container_width=True)
    st.markdown('<div class="brand-name">ClubConnect AI</div><div class="brand-sub">CAMPUS COMMUNITY PLATFORM</div>', unsafe_allow_html=True)
    st.caption("CAMPUS COMMUNITY PLATFORM")
    page = st.radio("NAVIGATION", ["Overview", "Explore Clubs", "Students", "Events & Sessions",
                                   "Engagement Analytics", "ML Lab", "About & Help"],
                    label_visibility="visible")
    st.markdown("---")
    st.markdown("**Your campus, connected.**")
    st.caption("Demo environment • Synthetic sample data")
    st.markdown("---")
    st.markdown('<div class="small-muted">Made for student communities<br>ClubConnect AI · v1.0.0</div>',
                unsafe_allow_html=True)

# ---------- Pages ----------
if page == "Overview":
    st.markdown('<div class="topline"><div><div class="eyebrow">AURELIA UNIVERSITY · CAMPUS LIFE</div><h1>Community Overview</h1><p style="margin:0">A little more connected, a little more involved.</p></div><div class="hero-chip">✦ Welcome to campus life</div></div>', unsafe_allow_html=True)
    total_students = len(students)
    active_clubs = int(clubs["is_active"].sum())
    total_regs = len(regs[regs["status"] == "Active"])
    upcoming = int((pd.to_datetime(events["event_date"]) >= pd.Timestamp.today().normalize()).sum())
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Registered students", f"{total_students:,}", "Across all departments", PALETTE["lavender_dark"])
    with c2: metric_card("Active clubs", f"{active_clubs}", "Student-led communities", "#F09FC4")
    with c3: metric_card("Active memberships", f"{total_regs:,}", "Club registrations", "#80C9EA")
    with c4: metric_card("Upcoming events", f"{upcoming}", "On the campus calendar", "#E9C44C")

    st.markdown("""
    <div class="hero">
      <span class="hero-chip">✦ YOUR CAMPUS, YOUR COMMUNITY</span>
      <h1>Find your people. Follow your passion.</h1>
      <p>Discover student communities, explore campus events and turn your interests into meaningful experiences.</p>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        section_heading("Club community snapshot", "Active memberships by student club")
        club_counts = regs[regs["status"] == "Active"].merge(clubs[["club_id","club_name"]], on="club_id")
        counts = club_counts.groupby("club_name").size().reset_index(name="memberships").sort_values("memberships", ascending=True)
        fig = px.bar(counts, x="memberships", y="club_name", orientation="h", color="club_name",
                     color_discrete_sequence=CHART_COLORS, text="memberships")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    with right:
        section_heading("Club category mix", "How campus interests are represented")
        category_counts = clubs.groupby("category").size().reset_index(name="clubs")
        fig = px.pie(category_counts, values="clubs", names="category", hole=.68,
                     color_discrete_sequence=CHART_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="white", width=3)))
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        section_heading("Upcoming campus moments", "A preview of what's happening next")
        upcoming_events = events[pd.to_datetime(events["event_date"]) >= pd.Timestamp.today().normalize()].sort_values("event_date").head(5)
        if upcoming_events.empty:
            st.info("No upcoming events in the demo calendar.")
        else:
            for _, ev in upcoming_events.iterrows():
                d1,d2,d3 = st.columns([1.5,3.5,1.2])
                with d1:
                    st.markdown(f"**{pd.to_datetime(ev['event_date']).strftime('%d %b')}**")
                    st.caption(ev["event_type"])
                with d2:
                    st.markdown(f"**{ev['event_name']}**")
                    st.caption(f"{ev['club_name']} · {ev['venue']}")
                with d3:
                    st.caption(f"{int(ev['registered'])} joined")
                st.divider()
    with right:
        section_heading("Engagement pulse", "Student activity segments from the ML model")
        seg_counts = segments["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment","students"]
        fig = px.bar(seg_counts, x="segment", y="students", color="segment",
                     color_discrete_sequence=[PALETTE["pink"], PALETTE["sky"], PALETTE["lavender_dark"]])
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(chart_layout(fig, 270), use_container_width=True)
        st.caption(f"Model: K-Means · Silhouette score: {model_meta['silhouette_score']:.3f}")

elif page == "Explore Clubs":
    st.markdown('<div class="eyebrow">DISCOVER YOUR COMMUNITY</div><h1>Explore Clubs</h1><p>Find a space where your interests can grow.</p>', unsafe_allow_html=True)
    search = st.text_input("🔎 Search clubs or interests", placeholder="Try coding, music, design...")
    category_options = ["All categories"] + sorted(clubs["category"].unique().tolist())
    category = st.selectbox("Filter by category", category_options)
    filtered = clubs.copy()
    if search:
        mask = (filtered["club_name"].str.contains(search, case=False, na=False) |
                filtered["category"].str.contains(search, case=False, na=False) |
                filtered["tags"].str.contains(search, case=False, na=False))
        filtered = filtered[mask]
    if category != "All categories":
        filtered = filtered[filtered["category"] == category]
    st.caption(f"Showing {len(filtered)} communities")
    cols = st.columns(3)
    for i, (_, club) in enumerate(filtered.iterrows()):
        with cols[i % 3]:
            tags = "".join([f'<span class="tag">{t.strip()}</span>' for t in club["tags"].split(",")])
            st.markdown(f"""<div class="club-card">
              <div class="club-icon">{club['emoji']}</div>
              <div class="club-title">{club['club_name']}</div>
              <div class="small-muted">{club['category']} · {int(club['members'])} listed members</div>
              <div class="club-desc">{club['description']}</div>{tags}</div>""", unsafe_allow_html=True)
            if st.button("View club insights", key=f"club_{club['club_id']}", use_container_width=True):
                st.session_state["selected_club"] = club["club_id"]
                st.session_state["page_club_detail"] = True
    if st.session_state.get("page_club_detail", False):
        club = clubs[clubs["club_id"] == st.session_state.get("selected_club")].iloc[0]
        st.markdown("---")
        section_heading(f"{club['emoji']} {club['club_name']}", club["description"])
        cr = regs[(regs["club_id"] == club["club_id"]) & (regs["status"] == "Active")]
        a,b,c = st.columns(3)
        with a: metric_card("Active members", len(cr), "Synthetic demo membership", PALETTE["lavender_dark"])
        with b: metric_card("Category", club["category"], "Club community", "#F09FC4")
        with c: metric_card("Interest tags", len(club["tags"].split(", ")), "Matching dimensions", "#80C9EA")
        st.dataframe(cr.merge(students[["student_id","student_name","department","year"]], on="student_id")
                     [["student_name","department","year","registered_on"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    section_heading("Find clubs made for you", "The recommendation engine compares your interests with club tags.")
    selected_interests = st.multiselect("Choose your interests", INTERESTS, default=["AI","Coding","Technology"])
    recs = recommend_clubs(selected_interests, clubs)
    if not recs.empty:
        for _, rec in recs.iterrows():
            score = int(round(rec["match_score"] * 100))
            st.markdown(f"**{rec['emoji']} {rec['club_name']}** · {rec['category']} · {score}% interest match")
            st.progress(min(score,100))
            st.caption(rec["description"])
    else:
        st.info("Choose at least one interest to receive recommendations.")

elif page == "Students":
    st.markdown('<div class="eyebrow">STUDENT COMMUNITY DIRECTORY</div><h1>Students</h1><p>Explore a sample view of the campus community and engagement segments.</p>', unsafe_allow_html=True)
    dept = st.multiselect("Department", sorted(students["department"].unique()), default=[])
    year = st.multiselect("Study year", sorted(students["year"].unique()), default=[])
    view = students.copy()
    if dept: view = view[view["department"].isin(dept)]
    if year: view = view[view["year"].isin(year)]
    view = view.merge(segments[["student_id","segment","engagement_score"]], on="student_id", how="left")
    st.caption(f"{len(view)} students shown · Synthetic demo records")
    st.dataframe(view[["student_id","student_name","department","year","interests","segment","engagement_score"]],
                 use_container_width=True, hide_index=True)
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download filtered student data", csv, "clubconnect_students.csv", "text/csv")
    section_heading("Engagement segment distribution", "Segments are generated by the K-Means clustering model.")
    sc = view["segment"].value_counts().reset_index()
    sc.columns = ["segment","students"]
    fig = px.pie(sc, names="segment", values="students", hole=.6,
                 color_discrete_sequence=[PALETTE["pink"],PALETTE["sky"],PALETTE["lavender_dark"]])
    st.plotly_chart(chart_layout(fig, 320), use_container_width=True)

elif page == "Events & Sessions":
    st.markdown('<div class="eyebrow">WHAT IS HAPPENING ON CAMPUS</div><h1>Events & Sessions</h1><p>Explore events, participation and attendance records.</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Event calendar", "Attendance desk"])
    with tab1:
        ev_view = events.copy()
        ev_view["event_date"] = pd.to_datetime(ev_view["event_date"])
        ev_view = ev_view.sort_values("event_date")
        status_filter = st.selectbox("Event status", ["All","Upcoming","Completed"])
        if status_filter != "All": ev_view = ev_view[ev_view["status"] == status_filter]
        st.dataframe(ev_view[["event_name","club_name","event_type","event_date","venue","registered","attended","status"]],
                     use_container_width=True, hide_index=True)
        fig = px.scatter(ev_view, x="event_date", y="registered", size="capacity", color="event_type",
                         hover_name="event_name", color_discrete_sequence=CHART_COLORS)
        st.plotly_chart(chart_layout(fig, 320), use_container_width=True)
    with tab2:
        st.caption("Demo attendance desk: changing attendance updates the local SQLite demo database.")
        ev_options = {f"{r.event_name} · {r.event_date}": r.event_id for r in events.itertuples()}
        chosen_label = st.selectbox("Choose an event", list(ev_options.keys()))
        chosen_id = ev_options[chosen_label]
        selected_event = events[events["event_id"] == chosen_id].iloc[0]
        new_attended = st.number_input("Number of students attended", min_value=0,
                                       max_value=int(selected_event["registered"]),
                                       value=int(selected_event["attended"]), step=1)
        if st.button("Save attendance", type="primary"):
            save_attendance(chosen_id, int(new_attended))
            st.success("Attendance updated in the local demo database.")
            st.rerun()

elif page == "Engagement Analytics":
    st.markdown('<div class="eyebrow">INSIGHTS THAT BRING PEOPLE TOGETHER</div><h1>Engagement Analytics</h1><p>Understand how students participate across clubs and events.</p>', unsafe_allow_html=True)
    active_regs = regs[regs["status"] == "Active"]
    club_stats = active_regs.groupby("club_id").size().rename("active_members").reset_index().merge(
        clubs[["club_id","club_name","category"]], on="club_id")
    a,b,c = st.columns(3)
    with a: metric_card("Average clubs per student", f"{len(active_regs)/max(len(students),1):.1f}", "Active memberships / students", PALETTE["lavender_dark"])
    with b: metric_card("Students with a club", f"{active_regs['student_id'].nunique()}", "At least one active membership", "#F09FC4")
    with c: metric_card("Event attendance rate", f"{events['attended'].sum()/max(events['registered'].sum(),1)*100:.0f}%", "Attendance / registrations", "#80C9EA")
    left,right = st.columns(2)
    with left:
        section_heading("Membership by club", "Which communities attract the most active members?")
        fig = px.bar(club_stats.sort_values("active_members"), x="active_members", y="club_name",
                     orientation="h", color="category", color_discrete_sequence=CHART_COLORS)
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
    with right:
        section_heading("Event attendance", "Registered versus attended students")
        e = events.melt(id_vars=["event_name"], value_vars=["registered","attended"],
                        var_name="measure", value_name="students")
        fig = px.bar(e, x="event_name", y="students", color="measure", barmode="group",
                     color_discrete_sequence=[PALETTE["sky"],PALETTE["pink"]])
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
    section_heading("Engagement by department", "Active club memberships across academic departments")
    dept_regs = active_regs.merge(students[["student_id","department"]], on="student_id").groupby("department").size().reset_index(name="memberships")
    fig = px.bar(dept_regs, x="department", y="memberships", color="department", color_discrete_sequence=CHART_COLORS)
    fig.update_layout(showlegend=False)
    st.plotly_chart(chart_layout(fig, 320), use_container_width=True)

elif page == "ML Lab":
    st.markdown('<div class="eyebrow">MACHINE LEARNING OPERATIONS</div><h1>ML Lab & Model Monitoring</h1><p>Explore the recommendation engine and student engagement segmentation model.</p>')
    a,b,c = st.columns(3)
    with a: metric_card("Algorithm", "K-Means", "Unsupervised learning", PALETTE["lavender_dark"])
    with b: metric_card("Model version", model_meta["version"], "Demo model metadata", "#F09FC4")
    with c: metric_card("Silhouette score", f"{model_meta['silhouette_score']:.3f}", "Cluster separation metric", "#80C9EA")
    section_heading("Training metadata", "The model is trained from the seeded synthetic demo dataset when the app starts.")
    st.json(model_meta)
    st.markdown("**Model input features**")
    st.write("Club count · Active club count · Derived engagement score")
    st.markdown("**Model output**")
    st.write("Three interpretable engagement groups: Needs Encouragement, Growing Engagement and Highly Engaged.")
    st.markdown("---")
    section_heading("Student engagement segments", "Inspect the output of the clustering model.")
    show = segments.merge(students[["student_id","student_name","department","year"]], on="student_id")
    st.dataframe(show[["student_id","student_name","department","year","club_count","active_clubs","engagement_score","segment"]],
                 use_container_width=True, hide_index=True)
    st.download_button("⬇ Download model metadata", json.dumps(model_meta, indent=2),
                       "clubconnect_model_metadata.json", "application/json")
    st.info("MLOps extension: connect MLflow to log parameters, metrics and model artifacts. This starter records local model metadata but does not require an MLflow server to run.")

elif page == "About & Help":
    st.markdown('<div class="eyebrow">PROJECT GUIDE</div><h1>About ClubConnect AI</h1><p>A student engagement analytics and club discovery platform.</p>')
    st.markdown("""
    <div class="hero"><span class="hero-chip">PROJECT PURPOSE</span>
    <h1>Helping every student find their community.</h1>
    <p>ClubConnect AI uses club-interest matching and engagement analytics to help students discover campus communities and help administrators understand participation patterns.</p></div>
    """, unsafe_allow_html=True)
    st.subheader("Main modules")
    modules = pd.DataFrame([
        ["Overview","Campus KPIs, club activity and upcoming events"],
        ["Explore Clubs","Search clubs and receive interest-based recommendations"],
        ["Students","Filter the student directory and view engagement segments"],
        ["Events & Sessions","View events and update demo attendance"],
        ["Engagement Analytics","Compare membership, attendance and department trends"],
        ["ML Lab","Inspect clustering model metrics and metadata"],
    ], columns=["Module","What it does"])
    st.dataframe(modules, use_container_width=True, hide_index=True)
    st.subheader("Demo data notice")
    st.warning("All student names, emails, clubs, registrations and event records in this starter project are synthetic demo data. Do not use them as real student records.")
    st.subheader("Beginner troubleshooting")
    st.markdown("""
    - If packages are missing, run `pip install -r requirements.txt`.
    - If the app port is busy, run `streamlit run app.py --server.port 8502`.
    - To reset demo data, stop the app and delete `data/clubconnect.db`; it will be recreated automatically.
    - Keep real student data private. Add authentication, authorization, backups and institution approval before any real deployment.
    """)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#A29BAD;font-size:11px;padding:8px">✦ ClubConnect AI · AURELIA UNIVERSITY · Built for connected campus communities ✦</div>',
            unsafe_allow_html=True)
