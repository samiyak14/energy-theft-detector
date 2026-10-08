# ⚡ Energy Theft Detection & Analytics Dashboard

An AI/ML-based project for identifying potentially suspicious electricity consumption patterns using synthetic smart-meter data, anomaly detection, clustering, and interactive visualizations.

The project simulates household electricity usage and introduces different abnormal consumption patterns that may indicate energy theft or meter tampering. The resulting data is analysed using machine learning techniques and presented through an interactive dashboard.

> **Note:** This project uses synthetically generated electricity consumption data for experimentation and demonstration purposes. It does not use real customer or utility data.

---

## 📌 Project Overview

Energy theft can produce unusual changes in electricity consumption that are difficult to identify through simple manual inspection.

This project explores how machine learning and data analytics can be used to:

- Analyse household electricity consumption patterns
- Identify unusual or suspicious consumption behaviour
- Detect different types of consumption anomalies
- Group households based on behavioural patterns
- Visualize suspicious households and consumption trends
- Provide an interactive dashboard for analysis

The project combines **unsupervised machine learning, feature analysis, and data visualization** to explore potential indicators of energy theft.

---

## 🔍 Anomaly Types

The synthetic data generator simulates several types of abnormal behaviour:

- **Sudden Drop** — significant reduction in consumption over a short period
- **Flatline** — unusually constant or near-zero meter readings
- **Spikes** — sudden abnormal increases in consumption
- **Gradual Drop** — progressive reduction in the normal consumption baseline

These patterns are introduced into otherwise normal household consumption profiles to create a controlled environment for testing detection techniques.

---

## 🤖 Machine Learning Approach

The project uses unsupervised learning techniques to analyse consumption behaviour.

### Isolation Forest

Isolation Forest is used for **anomaly detection**, helping identify observations that differ significantly from normal consumption patterns.

It is particularly useful for this problem because suspicious electricity usage may not always have clearly labelled examples available.

### K-Means Clustering

K-Means clustering is used to group households according to similarities in their consumption behaviour.

This allows the analysis to identify groups of households with similar patterns and investigate clusters containing potentially suspicious behaviour.

---

## 📊 Dashboard

The project includes an interactive dashboard built with **Streamlit** and **Plotly**.

The dashboard is designed to provide visual insights into:

- Household consumption trends
- Suspicious households
- Anomaly patterns
- Cluster distributions
- High-risk or unusual consumption behaviour
- Geographic/zone-based patterns

Interactive visualizations make it easier to explore the generated data and understand how different households behave over time.

---

## 🗂️ Project Structure

```text
energy-theft-detector/
│
├── app/
│   └── # Streamlit dashboard and application files
│
├── src/
│   └── # Data processing and machine learning components
│
├── energy_theft_data_generator.py
│   └── Synthetic smart-meter data generation
│
├── requirements.txt
└── README.md
```

---

## 🧪 Synthetic Dataset

The project includes a custom data generator that creates smart-meter-like electricity consumption data.

The generator:

1. Creates hourly consumption data for multiple households
2. Simulates daily consumption patterns
3. Incorporates weekday/weekend behaviour
4. Adds household-specific variability and noise
5. Injects different anomaly types
6. Assigns households to zones for visualization
7. Generates metadata describing the simulated anomalies

Example:

```bash
python energy_theft_data_generator.py --n_houses 80 --days 90 --seed 42
```

The generated data includes household identifiers, timestamps, electricity consumption, zone information, anomaly counts, and simulated labels.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| Isolation Forest | Anomaly detection |
| K-Means | Customer/household clustering |
| Streamlit | Interactive dashboard |
| Plotly | Interactive visualizations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model persistence |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/samiyak14/energy-theft-detector.git
cd energy-theft-detector
```

### 2. Install dependencies

It is recommended to create a virtual environment first.

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python energy_theft_data_generator.py
```

Or specify the number of households and number of days:

```bash
python energy_theft_data_generator.py --n_houses 80 --days 90 --seed 42
```

### 4. Run the dashboard

```bash
streamlit run app/<dashboard-file>.py
```

Replace `<dashboard-file>` with the Streamlit application file in the `app` directory.

---

## 📈 Analysis Workflow

```text
Synthetic Data Generation
          ↓
Data Preprocessing
          ↓
Feature Engineering
          ↓
Exploratory Data Analysis
          ↓
Isolation Forest
          ↓
K-Means Clustering
          ↓
Suspicious Pattern Identification
          ↓
Interactive Dashboard
```

---

## 🎯 Key Learning Outcomes

This project provided practical experience with:

- Unsupervised machine learning
- Anomaly detection
- Clustering techniques
- Time-series consumption analysis
- Synthetic data generation
- Feature engineering
- Interactive data visualization
- Building ML-powered dashboards with Streamlit
- Applying machine learning to a real-world problem

---

## 🔮 Future Improvements

Possible extensions include:

- Testing the approach on real-world smart-meter datasets
- Incorporating additional temporal and seasonal features
- Comparing multiple anomaly detection algorithms
- Developing more robust household-level behavioural profiles
- Improving anomaly scoring and risk classification
- Adding model evaluation using labelled datasets
- Deploying the dashboard as a web application

---

## 👩‍💻 Author

**Samiya Budye**

Computer Science Engineering (AI & ML)  
Finolex Academy of Management & Technology

[GitHub](https://github.com/samiyak14)
