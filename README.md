# IBB Wi-Fi Network IoT Analytics Platform 📡

**Course:** Internet of Things and Applied Data Science (Fall 2025)  
**Professor:** Dr. Mehmet Ali Akyol  
**Team:** Data Wave  
**Live Dashboard:** [http://13.49.241.124:8501](http://13.49.241.124:8501)  
**GitHub Repository:** [(https://github.com/baraa1muslah-jpg/IBB-IoT-Project)]


📋 Project Overview

This project implements an end-to-end IoT data science solution for monitoring and forecasting subscriber growth density across Istanbul Metropolitan Municipality's city-wide Wi-Fi network. The system addresses the challenge of network infrastructure planning by analyzing data from two IoT sensor streams with (87.3%) accuracy.

Key Requirements Met:
1- IoT Data Problem: Wi-Fi AP connection logs as sensor data
2- Data Cleaning & Preparation: Processed 2016-2025 IBB Open Data
3- Visualization & Dashboard: Interactive Streamlit dashboard
4- Baseline Modeling: Random Forest forecasting (>85% accuracy)
5- Cloud Deployment: AWS EC2 Free Tier instance



## 🏗️ Repository Structure
project-root/
├── dashboards/
│ └── app.py # Main Streamlit application
├── data/
│ ├── raw/ # Original data (.gitignored)
│ └── processed/ # Cleaned dataset
├── docs/
│ ├── images/             # Screenshots for project
│ └── report.md # Final project report
├── requirements.txt # Python dependencies
└── README.md # This file


## 🚀 Quick Start

### Option 1: View Live (Recommended)
Click the link above to view the running dashboard on AWS EC2.

### Option 2: Run Locally

1. **Clone the repository:**
git clone https://github.com/baraa1muslah-jpg/IBB-IoT-Project.git cd IBB-IoT-Project

2. **Install dependencies:**
pip install -r requirements.txt

3. **Run the dashboard:**
streamlit run dashboards/app.py

## 🛠️ Technical Implementation

**Data Sources**
● Primary Dataset: IBB Open Data Portal (Wi-Fi subscription logs)
● Period: 2016-2025
● Records: ~3.4 million connection logs
● Sensors: Access Point connection logs (two streams)

**Technologies**
● Backend: Python
● Visualization: Streamlit, Plotly, Mapbox
● Deployment: AWS EC2 (Free Tier), Ubuntu 24.04
● Version Control: GitHub



## 👥 Team Contributions

**Team Member	Contribution**
Al-Baraa Al-Qaisi :	Cloud Deployment (AWS), GitHub Management, Dashboard Integration & Model Development.
Farag Gaffar :	Data Cleaning , EDA & Pre-processing.



