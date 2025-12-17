import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from hazm import word_tokenize, Normalizer, stopwords_list
from collections import Counter
import re
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration for Database Connection
# Replace with your actual credentials in your .env based on sample.env
DB_CONFIG = {
    "DB_USER": os.getenv("DB_USER"),
    "DB_PASS": os.getenv("DB_PASS"),
    "DB_HOST": os.getenv("DB_HOST"),
    "DB_PORT": os.getenv("DB_PORT"),
    "DB_NAME": os.getenv("DB_NAME"),
}

# Industry Keywords Definition
# Organized by sectors as requested
INDUSTRY_KEYWORDS = {
    'Petrochemical': [
        'پتروشیمی خلیج فارس', 'تحریم پتروشیمی', 'خوراک پتروشیمی', 'خوراک گاز',
        'محدودیت گاز', 'قطع گاز', 'ناترازی گاز', 'اوره', 'آمونیاک', 'متانول',
        'پتروپالایش', 'رفع آلایندگی', 'بنزین پتروشیمی', 'گاز طبیعی', 'صنعت پتروشیمی'
    ],
    'Steel_Chain': [
        'صنایع فولاد', 'انرژی فولاد', 'گاز فولاد', 'آلیاژ', 'صنعت فولاد',
        'ورق فولادی', 'آهن اسفنجی', 'کنسانتره سنگ آهن', 'تیرآهن', 'فولاد ایران',
        'شمش فولاد', 'زنجیره مس', 'شمش فولادی', 'مدیریت فولاد', 'مواد اولیه',
        'فولاد خوزستان', 'فولاد مبارکه', 'ذوب آهن', 'ناترازی انرژی', 'صادرات فولاد'
    ],
    'Non_Ferrous_Metals': [
        'آلومینیوم', 'فلزات غیرآهنی', 'شمش آلومینیوم', 'کنسانتره مس', 'شمش مس',
        'توسعه زنجیره مس', 'نیکل', 'روی', 'سهام ملی صنایع مس', 'سهام فملی',
        'معدن مس سرچشمه', 'شمش روی', 'کاتد مس', 'قیمت جهانی مس'
    ],
    'Water_Industry': [
        'بحران آب', 'انتقال آب دریا', 'انتقال آب خلیج فارس به فلات مرکزی',
        'مدیریت آب', 'آلودگی آب', 'قطعی آب', 'زاینده‌رود', 'آب شیرین‌کن',
        'فرونشست زمین', 'مدیریت منابع آب', 'آبخیزداری', 'آب شیرین‌کن دریایی',
        'آبفا', 'تصفیه فاضلاب', 'لایروبی', 'بارورسازی ابرها',
        'سفره‌های آب زیرزمینی', 'حق آبه', 'بحران کم‌آبی'
    ],
    'Mining': [
        'سنگ آهن', 'کنسانتره', 'گندله', 'معدن طلا', 'ایمیدرو', 'حفاری اکتشافی',
        'ماشین‌آلات معدنی', 'دامپتراک', 'فلوتاسیون', 'لیچینگ',
        'پروانه بهره‌برداری', 'زغال سنگ'
    ]
}

class TelegramIndustryAnalyzer:
    """
    Analyzer class for processing Telegram channel posts related to specific industries.
    It handles fetching data from MariaDB, filtering based on keywords, and generating reports.
    """

    def __init__(self, DB_CONFIG, keywords):
        """
        Initialize the analyzer with database credentials and industry keywords.

        Args:
            db_config (dict): Database connection details.
            keywords (dict): Dictionary of industry names and their associated keywords.
        """
        self.keywords = keywords
        self.processed_data = None  # Will hold the final DataFrame
        
        # Create SQLAlchemy engine for database connection
        # Using mysql+pymysql connector (requires: pip install pymysql)
        connection_str = f"mysql+pymysql://{DB_CONFIG['DB_USER']}:{DB_CONFIG['DB_PASS']}@{DB_CONFIG['DB_HOST']}:{DB_CONFIG['DB_PORT']}/{DB_CONFIG['DB_NAME']}"
        self.engine = create_engine(connection_str)
        print(">> Database engine initialized successfully.")

    
    def _get_regex_patterns(self):
        """
        Internal helper to compile regex patterns for each industry.
        This optimizes the search process by creating a single regex per industry.
        
        Returns:
            dict: {industry_name: compiled_regex_pattern}
        """
        patterns = {}
        for industry, keys in self.keywords.items():
            # Join keywords with OR operator (|) and escape special characters
            # Example: (آهن|فولاد|مس)
            pattern_str = '|'.join([re.escape(k) for k in keys])
            patterns[industry] = pattern_str
        return patterns
    
    
    def fetch_and_filter_data(self, start_date, end_date):
        """
        Fetches data month-by-month to manage memory, filters by keywords immediately,
        and aggregates relevant posts.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        print(f">> Starting data fetch from {start_date} to {end_date}...")
        
        # Convert strings to datetime objects
        current_date = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_relevant_posts = []
        patterns = self._get_regex_patterns()

        # Iterate month by month
        while current_date < end:
            # Define the window for the current query (1 month)
            next_month = current_date + pd.DateOffset(months=1)
            # Ensure we don't go past the end date
            query_end = min(next_month, end)
            
            print(f"   Processing batch: {current_date.date()} to {query_end.date()}")

            # SQL Query for the specific time window
            # We select ONLY necessary columns to reduce I/O
            query = text("""
                SELECT text, full_date, channel_username, views
                FROM telegram_channel_post
                WHERE full_date >= :start AND full_date < :end
                AND text IS NOT NULL
            """)
            
            # Fetch data using pandas read_sql
            # params prevents SQL injection and handles date formatting
            df_batch = pd.read_sql(query, self.engine, params={"start": current_date, "end": query_end})
            
            if not df_batch.empty:
                # Filter logic: Check if text contains ANY keyword from ANY industry
                # We combine all industry patterns into one huge regex for the first pass filter
                # This drastically reduces rows before detailed categorization
                full_pattern = '|'.join(patterns.values())
                mask = df_batch['text'].str.contains(full_pattern, regex=True, na=False)
                
                relevant_batch = df_batch[mask].copy()
                
                # If we found relevant posts, append them to our list
                if not relevant_batch.empty:
                    all_relevant_posts.append(relevant_batch)
                    print(f"   -> Found {len(relevant_batch)} relevant posts in this batch.")
                else:
                    print("   -> No relevant posts found in this batch.")
            
            # Move to next month
            current_date = next_month

        # Concatenate all batches into a single DataFrame
        if all_relevant_posts:
            self.processed_data = pd.concat(all_relevant_posts, ignore_index=True)
            print(f">> Total relevant posts fetched: {len(self.processed_data)}")
        else:
            self.processed_data = pd.DataFrame()
            print(">> No relevant posts found in the entire period.")
            
            
    def categorize_posts(self):
        """
        Tags each post with the specific industries it belongs to based on keywords.
        Adds boolean columns for each industry (e.g., 'is_Petrochemical').
        """
        if self.processed_data is None or self.processed_data.empty:
            print("No data to categorize. Run fetch_and_filter_data first.")
            return

        print(">> Categorizing posts into industries...")
        patterns = self._get_regex_patterns()

        # Create a column for each industry indicating if the post is relevant
        for industry, pattern in patterns.items():
            # Create a boolean column: True if pattern is found, False otherwise
            col_name = f"is_{industry}"
            self.processed_data[col_name] = self.processed_data['text'].str.contains(pattern, regex=True, na=False)
        
        # Convert full_date to datetime if strictly needed for plotting later
        self.processed_data['full_date'] = pd.to_datetime(self.processed_data['full_date'])
        print(">> Categorization complete.")
        
    
    def generate_stats_report(self):
        """
        Calculates statistics for each industry, extracts top posts and top channels.
        
        Returns:
            dict: A dictionary containing stats per industry.
        """
        if self.processed_data is None:
            return {}

        report = {}
        
        for industry in self.keywords.keys():
            col_name = f"is_{industry}"
            # Filter data for this specific industry
            industry_df = self.processed_data[self.processed_data[col_name] == True]
            
            # 1. Count posts
            post_count = len(industry_df)
            
            # 2. Top posts by views (Top 20)
            top_posts = industry_df.nlargest(20, 'views')[['full_date', 'channel_username', 'views', 'text']]
            
            # 3. Top channels by total views in this industry
            top_channels = industry_df.groupby('channel_username')['views'].sum().nlargest(10)
            
            report[industry] = {
                'count': post_count,
                'top_posts': top_posts,
                'top_channels': top_channels
            }
            
            print(f"--- Stats for {industry} ---")
            print(f"Total Posts: {post_count}")
            print(f"Top Channel: {top_channels.index[0] if not top_channels.empty else 'N/A'}")

        return report
    
    
    def analyze_word_frequency(self, top_n=50):
        """
        Performs NLP analysis using 'hazm' library for normalization and stopword removal.
        """
        print(">> Starting NLP analysis (Word Frequency)...")
        
        # Initialize Hazm Normalizer
        normalizer = Normalizer()
        
        # 1. Get standard Persian stopwords from Hazm
        hazm_stops = stopwords_list()
        
        # 2. Add domain-specific stopwords (optional but recommended for clean charts)
        # These are noise words often found in news/telegram but not grammatical stopwords
        domain_stops = [
            'هزار', 'میلیون', 'میلیارد', 'تومان', 'ریال', 'دلار', 'درصد',
            'سال', 'ماه', 'روز', 'شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه',
            'گزارش', 'خبر', 'ادامه', 'تصویر', 'لینک', 'عضو', 'کانال', 'سلام', 'درود'
        ]
        
        # Combine both sets
        all_stops = set(hazm_stops + domain_stops)

        freq_report = {}

        for industry in self.keywords.keys():
            col_name = f"is_{industry}"
            # Check if dataframe exists and has the column
            if col_name not in self.processed_data.columns:
                continue

            industry_df = self.processed_data[self.processed_data[col_name] == True]
            
            if industry_df.empty:
                freq_report[industry] = []
                continue

            # Concatenate all text
            all_text = " ".join(industry_df['text'].astype(str).tolist())
            
            # Normalize text (converting Arabic chars to Persian, etc.)
            normalized_text = normalizer.normalize(all_text)
            
            # Tokenize using Hazm is implicit in many workflows, but here we do simple split first
            # Or use word_tokenize from hazm for better accuracy:
            tokens = word_tokenize(normalized_text)
            
            # Filter tokens: remove stopwords, punctuation, and short words
            clean_tokens = [
                word for word in tokens 
                if word not in all_stops 
                and len(word) > 2
                and not word.isnumeric() # Remove pure numbers
            ]
            
            # Count frequency
            counter = Counter(clean_tokens)
            freq_report[industry] = counter.most_common(top_n)
            
        print(">> NLP analysis complete.")
        return freq_report
    
    
    def plot_visualizations(self, stats_report, freq_report):
        """
        Generates and saves the requested plots using Matplotlib and Seaborn.
        """
        print(">> Generating visualizations...")
        sns.set_theme(style="whitegrid")
        # Note: You might need to set a Persian font for Matplotlib to display Farsi labels correctly
        # plt.rcParams['font.family'] = 'Vazir' # Example font
        
        # 1. Bar Chart: Compare Post Counts per Industry
        plt.figure(figsize=(10, 6))
        counts = {k: v['count'] for k, v in stats_report.items()}
        sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
        plt.title("Total Posts per Industry (Last 12 Months)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("chart_1_industry_counts.png")
        plt.close()

        # 2. Line Chart: Time Trend (Weekly)
        # We need to aggregate data by date for this
        plt.figure(figsize=(12, 6))
        for industry in self.keywords.keys():
            col_name = f"is_{industry}"
            df_ind = self.processed_data[self.processed_data[col_name] == True].copy()
            # Group by week
            weekly_counts = df_ind.resample('W', on='full_date').size()
            plt.plot(weekly_counts.index, weekly_counts.values, label=industry)
        
        plt.title("Weekly Post Trend per Industry")
        plt.legend()
        plt.tight_layout()
        plt.savefig("chart_2_time_trend.png")
        plt.close()

        # 3. Bar Chart: Top Channels for one industry (Example: Steel_Chain)
        # We can loop to create for all, but here is an example for Steel
        target_industry = 'Steel_Chain'
        if target_industry in stats_report:
            top_ch = stats_report[target_industry]['top_channels']
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_ch.values, y=top_ch.index, palette="magma")
            plt.title(f"Top 10 Channels by Views - {target_industry}")
            plt.tight_layout()
            plt.savefig(f"chart_3_top_channels_{target_industry}.png")
            plt.close()

        print(">> Charts saved successfully.")
        
# --- Usage Example ---
if __name__ == "__main__":
    # Initialize
    analyzer = TelegramIndustryAnalyzer(DB_CONFIG, INDUSTRY_KEYWORDS)
    
    # Run Pipeline (Example for 1 year)
    analyzer.fetch_and_filter_data('2024-01-01', '2025-01-01')
    analyzer.categorize_posts()
    
    # Reports
    stats = analyzer.generate_stats_report()
    freq = analyzer.analyze_word_frequency()
    
    # Plots
    analyzer.plot_visualizations(stats, freq)