# ClubConnect AI — Student Club Engagement Analytics

A beginner-friendly BCA AIML + MLOps dashboard starter built with Streamlit, Plotly,
Pandas, Scikit-learn and SQLite.

## What's included

- Pastel dashboard UI: lavender, pink, yellow, white and sky blue
- Custom Aurelia University demo identity (fictional brand)
- Preloaded synthetic demo students, clubs, registrations, events and attendance
- Overview KPIs and interactive Plotly charts
- Club search, category filtering and interest-based recommendations
- Student directory with filters and CSV export
- Event calendar and editable demo attendance
- Engagement analytics
- K-Means student engagement segmentation and model metadata
- Local SQLite database created automatically on first run

## 1. Install Python

Use Python 3.10 or 3.11. During installation on Windows, select "Add Python to PATH".

Check:
```bash
python --version
```

## 2. Extract the ZIP

Extract `ClubConnect_AI_Project.zip` to a folder, for example:
`C:\Users\YourName\Desktop\ClubConnect_AI`

Open that folder in VS Code.

## 3. Create a virtual environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the dashboard

```bash
streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, open:
`http://localhost:8501`

## 6. Demo data

The app automatically creates `data/clubconnect.db` the first time it runs.
The generated data is synthetic and reproducible. It includes 180 sample students,
10 clubs, club registrations, 32 events and attendance data.

To reset the demo database:
1. Stop Streamlit with Ctrl+C in the terminal.
2. Delete `data/clubconnect.db`.
3. Run `streamlit run app.py` again.

## 7. Dashboard navigation

- Overview: campus KPIs, club membership charts, upcoming events, engagement segments
- Explore Clubs: search and filter clubs, get interest-based club recommendations
- Students: filter student demo records and export CSV
- Events & Sessions: browse events and update demo attendance
- Engagement Analytics: membership, attendance and department charts
- ML Lab: inspect K-Means clustering and model metadata
- About & Help: project purpose and troubleshooting

## 8. Machine learning used

### Club recommendation
The recommendation engine creates multi-hot vectors from a student's selected interests
and each club's tags. It calculates cosine similarity and ranks clubs by match score.

### Engagement segmentation
K-Means groups students using:
- Number of registered clubs
- Number of active club memberships
- Derived engagement score

The app displays a silhouette score as a clustering evaluation metric. This is a demo
analytics model, not a validated measure of a student's real-world engagement.

## 9. MLOps roadmap for your college submission

This starter gives you a working local app and a basic model metadata artifact.
For a stronger MLOps demonstration, implement these stages one by one:

1. **Data versioning:** save a dated copy of the input CSV / SQLite export and track changes with Git or DVC.
2. **Experiment tracking:** install MLflow and log model name, parameters, silhouette score, dataset version and training timestamp.
3. **Model registry:** register approved model versions and document promotion from Staging to Production.
4. **Testing:** add pytest tests for data schema, recommendation output, missing values and model training.
5. **CI:** add GitHub Actions to install dependencies and run tests on every push.
6. **Monitoring:** log prediction requests, recommendation scores, errors and data drift indicators.
7. **Deployment:** deploy the Streamlit app only after removing demo credentials/data and adding appropriate access controls.

MLflow is optional in this starter. It is deliberately not required to launch the beginner demo.

## 10. Security and responsible use

- The bundled data is fictional. Do not interpret it as real student information.
- Do not upload real student records without institution approval and an appropriate privacy basis.
- Keep API keys and credentials out of source code; use environment variables or Streamlit secrets.
- Validate uploaded files and restrict file size/type if you add uploads.
- Use parameterized SQL queries; this app uses parameterized SQL for attendance updates.
- Do not load untrusted pickle/joblib model files.
- Add authentication and role-based authorization before real multi-user deployment.
- Use HTTPS, dependency updates, backups and security testing for production.
- No application can be guaranteed completely error-free or vulnerability-free; test in the target environment.

## Troubleshooting

**`streamlit` is not recognized**
```bash
python -m streamlit run app.py
```

**ModuleNotFoundError**
Activate `.venv`, then run:
```bash
pip install -r requirements.txt
```

**Port 8501 already in use**
```bash
streamlit run app.py --server.port 8502
```

**Dashboard data needs a reset**
Stop the app and delete `data/clubconnect.db`, then restart it.

## Suggested college project title

"ClubConnect AI: A Machine Learning-Based Student Club Recommendation and Campus Engagement Analytics Platform with MLOps Monitoring"
