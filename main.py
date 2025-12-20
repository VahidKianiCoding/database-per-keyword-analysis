import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from hazm import word_tokenize, Normalizer, stopwords_list
from collections import Counter
import re
from datetime import datetime, timedelta
import os
import logging
from tqdm import tqdm
from urllib.parse import quote_plus
from dotenv import load_dotenv
from wordcloud import WordCloud

# Libraries for correct Persian text rendering in plots
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_RESHAPER = True
except ImportError:
    HAS_RESHAPER = False
    print("Warning: 'arabic-reshaper' or 'python-bidi' not installed. Persian text in plots might look disjointed.")

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

# Setup basic logging to see what's getting dropped
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        'توسعه زنجیره مس', 'نیکل', 'سرب و روی', 'سهام ملی صنایع مس', 'سهام فملی',
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

def make_farsi_text_readable(text):
    """
    Helper to reshape Farsi text for Matplotlib so letters correspond correctly (RTL).
    Requires arabic-reshaper and python-bidi.
    """
    if HAS_RESHAPER:
        reshaped_text = arabic_reshaper.reshape(text) # type: ignore
        return get_display(reshaped_text) # type: ignore
    return text

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
        self.processed_data = None
        self.engine = None
        
        # Database setup - Only attempt if config is provided
        # This allows offline mode without errors if DB credentials are missing
        # URL encode the username and password to handle special characters like '@', ':', '/'
        if DB_CONFIG.get('DB_HOST'):    
            try:
                safe_user = quote_plus(DB_CONFIG['DB_USER'])
                safe_pass = quote_plus(DB_CONFIG['DB_PASS'])
            
                # Create SQLAlchemy engine for database connection
                # Using mysql+pymysql connector (requires: pip install pymysql)
                connection_str = (
                    f"mysql+pymysql://{safe_user}:{safe_pass}"
                    f"@{DB_CONFIG['DB_HOST']}:{DB_CONFIG['DB_PORT']}/{DB_CONFIG['DB_NAME']}"
                )
                self.engine = create_engine(connection_str)
                print(">> Database engine initialized successfully.")
            except Exception as e:
                print(f"Warning: Database connection failed ({e}). Running in offline mode only.")

    
    def _get_regex_patterns(self):
        """
        Internal helper to compile regex patterns for each industry.
        This optimizes the search process by creating a single regex per industry.
        
        Returns:
            dict: {industry_name: compiled_regex_pattern}
        """
        patterns = {}
        for industry, keys in self.keywords.items(): # type: ignore
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
            
            try:
                if self.engine:    
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
            
            except Exception as e:
                print(f"   Error in batch: {e}")
            
            # Move to next month
            current_date = next_month

        # Concatenate all batches into a single DataFrame
        if all_relevant_posts:
            self.processed_data = pd.concat(all_relevant_posts, ignore_index=True)
            print(f">> Total relevant posts fetched: {len(self.processed_data)}")
        else:
            self.processed_data = pd.DataFrame()
            
            
    def categorize_posts(self):
        """
        Tags each post with the specific industries it belongs to based on keywords.
        Adds boolean columns for each industry (e.g., 'is_Petrochemical').
        """
        if self.processed_data is None or self.processed_data.empty:
            return

        print(">> Categorizing posts into industries...")
        patterns = self._get_regex_patterns()

        # Create a column for each industry indicating if the post is relevant
        for industry, pattern in patterns.items():
            # Create a boolean column: True if pattern is found, False otherwise
            col_name = f"is_{industry}"
            self.processed_data[col_name] = self.processed_data['text'].str.contains(pattern, regex=True, na=False)
        self.processed_data['full_date'] = pd.to_datetime(self.processed_data['full_date'])
        
        # Convert full_date to datetime if strictly needed for plotting later
        self.processed_data['full_date'] = pd.to_datetime(self.processed_data['full_date'])
        print(">> Categorization complete.")
        
    
    def analyze_keyword_breakdown(self):
        """
        Calculates which specific keywords triggered the matches for each industry.
        Returns: dict {industry: {keyword: count}}
        """
        print(">> Analyzing keyword breakdown (Chart 2 Data)...")
        breakdown = {}
        
        for industry, keys in self.keywords.items(): # type: ignore
            col_name = f"is_{industry}"
            if col_name not in self.processed_data.columns: # type: ignore
                continue
            
            df_ind = self.processed_data[self.processed_data[col_name] == True] # type: ignore
            if df_ind.empty:
                continue
                
            key_counts = {}
            for k in keys:
                # Count occurrences of specific keywords
                count = df_ind['text'].str.contains(re.escape(k), regex=True).sum()
                if count > 0:
                    key_counts[k] = count
            
            # Sort by count descending
            breakdown[industry] = dict(sorted(key_counts.items(), key=lambda item: item[1], reverse=True))
            
        return breakdown
    
    def generate_stats_report(self):
        """
        Calculates statistics for each industry, extracts top posts and top channels.
        
        Returns:
            dict: A dictionary containing stats per industry.
        """
        if self.processed_data is None: return {}
        
        report = {}
        
        for industry in self.keywords.keys(): # type: ignore
            col_name = f"is_{industry}"
            if col_name not in self.processed_data.columns: continue
            
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
            print(f"--- {industry}: {post_count} posts ---")
        return report
    
    
    def analyze_word_frequency(self, top_n=50):
        """
        Performs NLP analysis:
        1. Per Industry
        2. Global (All relevant posts combined)
        Returns a dictionary containing frequency lists for both.
        """
        print(">> Starting NLP analysis (Global & Per Industry)...")
        
        # Initialize Hazm Normalizer
        normalizer = Normalizer(
            True, # correct_spacing
            True, # remove_diacritics
            True, # remove_specials_chars
            True, # decrease_repeated_chars
            True, # persian_style
            True, # persian_numbers
            True, # unicodes_replacement
            True, # seperate_mi
            )
        
        # 1. Get standard Persian stopwords from Hazm
        hazm_stops = stopwords_list()
        
        # 2. Add domain-specific stopwords (optional but recommended for clean charts)
        # These are noise words often found in news/telegram but not grammatical stopwords
        domain_stops = [
            'هزار', 'میلیون', 'میلیارد', 'تومان', 'ریال', 'دلار', 'درصد', 'عدد', 'شماره',
            'سال', 'ماه', 'روز', 'شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه',
            'گزارش', 'خبر', 'ادامه', 'تصویر', 'لینک', 'عضو', 'کانال', 'سلام', 'درود', 'جهت', 'مطلب',
            'افزایش', 'کاهش', 'نیز', 'باید', 'شدن', 'داد', 'کرد', 'کند', 'است', 'بود', 'شد', 'گفت', 'وی',
            'این', 'آن', 'با', 'بر', 'برای', 'که', 'از', 'به', 'در', 'را', 'تا', 'چون', 'چه', 'اگر',
            'هست', 'نیست', 'دارد', 'داشت', 'می', 'نمی', 'های', 'ها', 'تر', 'ترین', 'می‌شود', 'می‌باشد'
        ]
        
        # Combine both sets
        all_stops = set(hazm_stops + domain_stops)
        
        freq_report = {}
        
        # --- Helper Function for Tokenization ---
        def get_tokens_counter(texts_list):
            local_counter = Counter()
            for txt in tqdm(texts_list, leave=False):
                normalized = normalizer.normalize(txt)
                tokens = normalized.split() # Fast split
                clean_tokens = [
                    t for t in tokens 
                    if t not in all_stops 
                    and len(t) > 2 
                    and not t.isnumeric()
                ]
                local_counter.update(clean_tokens)
            return local_counter

        # 1. Per Industry Analysis
        for industry in self.keywords.keys(): # type: ignore
            col_name = f"is_{industry}"
            # Check if dataframe exists and has the column
            if col_name not in self.processed_data.columns: continue # type: ignore
             
            industry_df = self.processed_data[self.processed_data[col_name] == True] # type: ignore
            if industry_df.empty: continue
            
            print(f"   -> Analyzing Industry: {industry}...")
            texts = industry_df['text'].dropna().astype(str).tolist()
            cnt = get_tokens_counter(texts)
            freq_report[industry] = dict(cnt.most_common(top_n))            
            
            # 2. Global Analysis (All Industries Combined)
            print("   -> Analyzing Global (All Industries)...")
            
            # Filter rows where AT LEAST one industry flag is True
            industry_cols = [f"is_{k}" for k in self.keywords.keys() if f"is_{k}" in self.processed_data.columns] # type: ignore
            if industry_cols:
                # Create a mask for rows that belong to at least one industry
                global_mask = self.processed_data[industry_cols].any(axis=1) # type: ignore
                global_df = self.processed_data[global_mask] # type: ignore
            
                if not global_df.empty:
                    global_texts = global_df['text'].dropna().astype(str).tolist()
                    global_cnt = get_tokens_counter(global_texts)
                    freq_report['Global_All_Industries'] = dict(global_cnt.most_common(top_n))
            
        print(">> NLP analysis complete.")
        return freq_report
    
    
    def plot_visualizations(self, stats_report, freq_report, keyword_breakdown):
        """
        Generates and saves the requested plots using Matplotlib and Seaborn.
        """
        print(">> Generating 5 visualizations...")
        sns.set_theme(style="whitegrid")
        # Ensure a font that supports Arabic/Persian script is available on your OS
        # 'Arial' usually supports it on Windows, otherwise install 'Vazir' or 'Tahoma'
        plt.rcParams['font.family'] = 'Arial'
        
        # 1. Bar Chart: Total Posts per Industry
        if stats_report:
            plt.figure(figsize=(10, 6))
            counts = {make_farsi_text_readable(k): v['count'] for k, v in stats_report.items()}
            sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
            plt.title(make_farsi_text_readable("Total Posts per Industry")) # type: ignore
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("1_industry_counts.png")
            plt.close()

        # 2. Bar Chart: Keyword Breakdown (Top 10 keywords per industry)
        if keyword_breakdown:
            for industry, keys_data in keyword_breakdown.items():
                if not keys_data: continue
                top_keys = dict(list(keys_data.items())[:10])
                plt.figure(figsize=(12, 6))
                labels = [make_farsi_text_readable(k) for k in top_keys.keys()]
                sns.barplot(x=list(top_keys.values()), y=labels, palette="rocket")
                plt.title(make_farsi_text_readable(f"Most-Used Keywords in {industry}")) # type: ignore
                plt.xlabel("Count")
                plt.tight_layout()
                plt.savefig(f"2_keywords_{industry}.png")
                plt.close()

        # 3. Word Frequencies (Cloud + Bar Chart)
        if freq_report:
            # Font config
            font_path = "C:\\Windows\\Fonts\\tahoma.ttf" 
            if not os.path.exists(font_path): font_path = "arial.ttf"

            for group_name, freqs in freq_report.items():
                if not freqs: continue
                
                # A. Word Cloud (Decorative)
                if HAS_RESHAPER:
                    reshaped_freqs = {get_display(arabic_reshaper.reshape(k)): v for k, v in freqs.items()} # type: ignore
                else:
                    reshaped_freqs = freqs
                
                wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path)
                wc.generate_from_frequencies(reshaped_freqs)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                plt.title(make_farsi_text_readable(f"WordCloud in {group_name}")) # type: ignore
                plt.savefig(f"4_wordcloud_{group_name}.png")
                plt.close()
                
                # B. Bar Chart (Analytical) - NEW FEATURE
                # Take top 15 words for the bar chart
                top_words = dict(list(freqs.items())[:15])
                plt.figure(figsize=(12, 8))
                
                words = [make_farsi_text_readable(k) for k in top_words.keys()]
                counts = list(top_words.values())
                
                # Use a special color for Global analysis
                palette = "magma" if group_name == 'Global_All_Industries' else "crest"
                
                sns.barplot(x=counts, y=words, palette=palette)
                plt.title(make_farsi_text_readable(f"Most-Used Words in {group_name}")) # type: ignore
                plt.xlabel("Frequency")
                plt.tight_layout()
                plt.savefig(f"4_barchart_freq_{group_name}.png")
                plt.close()

        # 4. Top Channels
        if stats_report:
            for industry, data in stats_report.items():
                top_ch = data['top_channels']
                if top_ch.empty: continue
                plt.figure(figsize=(10, 6))
                sns.barplot(x=top_ch.values, y=top_ch.index, palette="mako")
                plt.title(make_farsi_text_readable(f"Top Channels by View in {industry}")) # type: ignore
                plt.xlabel("Total Views")
                plt.tight_layout()
                plt.savefig(f"3_top_channels_{industry}.png")
                plt.close()

        # 5. Weekly Trend
        plt.figure(figsize=(12, 6))
        for industry in self.keywords.keys():
            col_name = f"is_{industry}"
            if col_name in self.processed_data.columns: # type: ignore
                df_ind = self.processed_data[self.processed_data[col_name] == True].copy() # type: ignore
                if not df_ind.empty:
                    weekly_counts = df_ind.resample('W', on='full_date').size()
                    plt.plot(weekly_counts.index, weekly_counts.values, label=make_farsi_text_readable(industry)) # type: ignore
        
        plt.title(make_farsi_text_readable("Weekly Trend of Relevant Posts")) # type: ignore
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("5_time_trend.png")
        plt.close()
        print(">> All charts saved.")


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Robustly loads the CSV file, handling parsing errors, repeated headers,
    and invalid datetime formats automatically.
    """
    try:
        # Attempt 1: Try reading with Python engine which is more forgiving with quotes
        # on_bad_lines='skip' will ignore lines with too many fields (like line 512 error)
        df = pd.read_csv(
            file_path, 
            engine='python', 
            on_bad_lines='skip',
            encoding='utf-8'
        )
    except Exception as e:
        logger.error(f"Failed to load CSV normally: {e}")
        return pd.DataFrame()

    initial_count = len(df)
    
    # 1. Fix: Remove rows that are actually repeated headers
    # This happens when multiple CSVs are merged or file is corrupted
    if 'full_date' in df.columns:
        df = df[df['full_date'] != 'full_date']
    
    # 2. Fix: robust datetime conversion
    # errors='coerce' turns unparseable dates into NaT (Not a Time) instead of crashing
    df['full_date'] = pd.to_datetime(df['full_date'], errors='coerce')
    
    # 3. Fix: Drop rows where date conversion failed (essential for time-series analysis)
    df = df.dropna(subset=['full_date'])
    
    # 4. Ensure numeric types for views
    if 'views' in df.columns:
        df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)

    final_count = len(df)
    dropped_count = initial_count - final_count
    
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} invalid or corrupted rows during cleaning.")
        
    logger.info(f"Successfully loaded {final_count} clean records.")
    
    return df


        
if __name__ == "__main__":
    # --- Configuration ---
    CSV_FILENAME = "telegram_industry_data.csv"
    FORCE_FETCH = False  # Set to True to ignore cache and fetch fresh data from DB
    CSV_SEPARATOR = ','  # Note: The new loader assumes standard CSV format
    
    # Initialize Analyzer
    analyzer = TelegramIndustryAnalyzer(DB_CONFIG, INDUSTRY_KEYWORDS)
    
    try:
        # DECISION LOGIC: Fetch from DB or Load from Disk?
        data_loaded = False
        
        # Condition 1: Fetch if forced OR file doesn't exist
        if FORCE_FETCH or not os.path.exists(CSV_FILENAME):
            print(f">> Mode: ONLINE (Reason: FORCE_FETCH={FORCE_FETCH} or File Missing)")
            
            # Calculate dynamic dates (Last n days)
            today = datetime.now()
            # You can adjust this window as needed
            start_date = today - timedelta(days=365) 
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = today.strftime('%Y-%m-%d')
            
            # 1. Fetch
            analyzer.fetch_and_filter_data(start_str, end_str)
            
            # 2. Save to CSV (Cache)
            if analyzer.processed_data is not None and not analyzer.processed_data.empty:
                print(f">> Saving fetched data to {CSV_FILENAME}...")
                analyzer.processed_data.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig', sep=CSV_SEPARATOR)
                data_loaded = True
            else:
                print("!! Warning: No data fetched from Database.")
                
        # Condition 2: Load from local cache (Offline Mode)
        else:
            print(f">> Mode: OFFLINE (Loading {CSV_FILENAME})")
            
            # HERE IS THE CHANGE: Use the robust loader function we created
            # This replaces the complex try/except block for reading CSVs
            analyzer.processed_data = load_and_clean_data(CSV_FILENAME)
            
            if analyzer.processed_data is not None and not analyzer.processed_data.empty:
                data_loaded = True
                print(f">> Loaded {len(analyzer.processed_data)} records from disk.")
            else:
                print("!! Warning: Loaded file is empty or corrupted.")

        # --- ANALYSIS PIPELINE ---
        if data_loaded and analyzer.processed_data is not None and not analyzer.processed_data.empty:
            
            # Step 1: Categorize Posts
            # Note: This method also ensures 'full_date' is datetime, which is safe to run again
            analyzer.categorize_posts()
            
            # Step 2: Generate Statistics
            stats = analyzer.generate_stats_report()
            
            # Step 3: Specific Keyword Breakdown
            keyword_stats = analyzer.analyze_keyword_breakdown()
            
            # Step 4: NLP & Word Cloud Analysis
            freq_stats = analyzer.analyze_word_frequency()
            
            # Step 5: Visualize
            analyzer.plot_visualizations(stats, freq_stats, keyword_stats)
            
            print("\n>> Pipeline Finished Successfully. Check generated PNG files.")
            
        else:
            print(">> No data available to process. Check database connection or CSV file.")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()