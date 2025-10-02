# Audible Insights: Intelligent Book Recommendations

## Project Overview
This project builds an intelligent book recommendation system to help users discover personalized book suggestions based on their preferences. The system leverages **data cleaning, NLP feature extraction, clustering, and recommendation models**, and presents results via a **Streamlit web application deployed on AWS**.

---

## Domain
**Recommendation Systems**

---

## Problem Statement
Readers often struggle to find books that match their interests. This project aims to:
- Analyze book datasets for ratings, genres, and user interactions.
- Apply NLP and clustering to group similar books.
- Build multiple recommendation models (content-based, clustering-based, hybrid).
- Deploy a Streamlit app to provide personalized recommendations.

---

## Business Use Cases
1. **Personalized Reading Experience** – Recommend books tailored to user preferences.
2. **Enhanced Library & Bookstore Systems** – Improve borrowing and sales strategies.
3. **Author/Publisher Insights** – Identify popular genres and trending books.
4. **Reader Engagement** – Suggest trending or highly rated books across categories.

---

## Datasets
**Dataset 1:** `Audible_Catlog.csv`  
**Dataset 2:** `Audible_Catlog_Advanced_Features.csv`  

**Key Columns Include:**
- `Book Name`, `Author`, `Rating`, `Number of Reviews`, `Price`, `Description`
- `Listening Time`, `Ranks`, `Genre`
- Dataset 2 adds advanced features like additional ratings or metadata.

> Note: Large datasets can be added via `.gitignore` or hosted on AWS S3 to avoid GitHub size limits.

---

## Approach

### 1. Data Preparation
- Merge the two datasets based on `Book Name` and `Author`.
- Handle missing values, duplicates, and inconsistent data.

### 2. Exploratory Data Analysis (EDA)
- Analyze genres, ratings, publication trends, and author popularity.
- Visualize insights using bar charts, heatmaps, and line charts.

### 3. NLP & Clustering
- Extract text features from book titles, descriptions, and reviews.
- Cluster books using algorithms like K-Means or DBSCAN.

### 4. Recommendation Models
- **Content-Based Filtering** – Based on book attributes and user preferences.
- **Clustering-Based Recommendations** – Suggest books from similar clusters.
- **Hybrid Approaches** – Combine content-based and clustering models.
- Evaluate models using metrics like precision, recall, and RMSE.

### 5. Streamlit Application
- Input user preferences (favorite genres or books).
- Display personalized recommendations.
- Visualize data insights and trends.

---

## Results
- Cleaned, merged, and processed datasets.
- Text features extracted and books clustered.
- Multiple recommendation models developed and evaluated.
- Functional Streamlit application deployed on AWS.

---

## Project Structure

