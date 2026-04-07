# InfraSense AI 🏗️🤖
**Predictive Analytics for Government Infrastructure Projects**

### 📌 Overview
InfraSense AI is an AI-powered decision-support engine developed for the **Mini Smart India Hackathon 2026** (Problem Statement: MSIH25030). The system addresses systemic inefficiencies in major infrastructure projects, where cost overruns typically average 20–35% and delays can span several years.

### 🚀 Key Features
* **Predictive Risk Engine:** Leverages Machine Learning to forecast the probability of delays and budget overruns at the pre-sanction and mid-execution stages.
* **Real-Time What-if Simulator:** An interactive simulation environment where users can adjust variables—such as budget, timeline, and contractor ratings—to see instant impacts on project risk.
* **Dynamic Analytics:** Visualizes historical trends and risk distribution using integrated data charts to assist in proactive decision-making.
* **Contractor Performance Profiling:** Analyzes executing agencies based on historical performance to identify high-risk vendors before contract awards.

### 🛠️ Tech Stack
* **Language:** Python
* **Web Interface:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **Data Processing:** Pandas, NumPy

### 🧠 System Logic
The model utilizes a **Random Forest** architecture to process non-linear relationships between project variables. By optimizing the learning process through concepts like **Gradient Descent** (minimizing the loss function), the engine provides stable and interpretable risk scores tailored for government auditing workflows.

### ⚙️ Installation & Usage
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/safwan447/InfraSense-AI.git](https://github.com/safwan447/InfraSense-AI.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas scikit-learn
    ```
3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

---
*Developed for the Mini SIH 2026 Grand Finale at Presidency University, Bengaluru.*