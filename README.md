# 📊 Telegram Industry Analyzer (Persian Market)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**Telegram Industry Analyzer** is a powerful Python tool designed to crawl, scrape, and analyze Persian Telegram channels. It categorizes content into specific industrial sectors (Petrochemical, Steel, Mining, etc.), filters out noise (spam, sports, ads), and generates publication-ready visualizations and NLP insights.

## 🚀 Key Features

- **Automated Categorization**: Classifies posts into industries like *Petrochemical*, *Steel Chain*, *Non-Ferrous Metals*, *Water Industry*, and *Mining* based on a comprehensive keyword set.
- **Advanced Persian NLP**: Uses **Hazm** for normalization, tokenization, and lemmatization of Persian text.
- **Smart Noise Filtering**:
  - **Context-Aware Filters**: Removes sports news (football, leagues) and advertisements (VPN, hair transplant, etc.) to ensure data purity.
  - **Channel Blacklisting**: Automatically excludes irrelevant or spammy channels.
  - **Stopword Removal**: Cleans generic Persian words, numbers, and IDs.
- **High-Quality Visualizations**: Generates professional 16:9 charts using **Seaborn**:
  - Weekly time trends (Dynamic scaling).
  - Word Clouds for each industry.
  - Top active channels & most viewed posts.
  - Keyword frequency breakdowns.
- **Audit & Debugging**: Exports CSV reports (`channel_audit.csv`, `nlp_debug.csv`) to verify data accuracy and identify false positives.

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/yourusername/telegram-industry-analyzer.git](https://github.com/yourusername/telegram-industry-analyzer.git)
   cd telegram-industry-analyzer

2. **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt

4. **Font Setup:**

For correct Persian text rendering in charts, create an assets folder in the root directory and place the Vazirmatn font file there:

assets/Vazirmatn-Regular.ttf

## ⚙️ Configuration

Create a .env file in the root directory to configure your database connection. You can use the provided sample.env as a template:

    # Database Credentials
    DB_USER=your_db_user
    DB_PASS=your_db_password
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=telegram_db

# Database Schema

The script expects a PostgreSQL database (or a CSV fallback) with a table named telegram_posts containing at least these columns:

channel_username (Text)

text (Text: Post content)

views (Integer)

date (DateTime)

# 🖥️ Usage

Run the main script:
    
    python main.py

# The script will:

Connect to the database and fetch posts.

Clean and categorize the data.

Perform NLP analysis.

Generate PNG charts in the root directory (e.g., 1_industry_counts.png, 5_time_trend.png).

Export CSV audit files for inspection.

# 📊 Output Examples
The tool generates several visualizations:

Industry Distribution: Bar chart of total posts per sector.

Top Keywords: Which keywords are driving the categorization.

Word Clouds: Visual representation of the most frequent terms in each industry.

Time Trend: Daily/Weekly analysis of post volume.

Activity Volume: Most active channels within specific sectors.

# 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project.

Create your feature branch (git checkout -b feat/AmazingFeature).

Commit your changes (git commit -m 'feat: add some AmazingFeature').

Push to the branch (git push origin feat/AmazingFeature).

Open a Pull Request.

# 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

Developed with ❤️ for Persian Data Analysis.