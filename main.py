import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from hazm import word_tokenize, Normalizer, stopwords_list, Lemmatizer, InformalNormalizer
from collections import Counter
import re
from datetime import datetime, timedelta
import os
import logging
from tqdm import tqdm
from urllib.parse import quote_plus
from dotenv import load_dotenv
from wordcloud import WordCloud
import io

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
    'Steel Chain': [
        'صنایع فولاد', 'انرژی فولاد', 'گاز فولاد', 'آلیاژ', 'صنعت فولاد',
        'ورق فولادی', 'آهن اسفنجی', 'کنسانتره سنگ آهن', 'تیرآهن', 'فولاد ایران',
        'شمش فولاد', 'زنجیره مس', 'شمش فولادی', 'مدیریت فولاد', 'مواد اولیه',
        'فولاد خوزستان', 'فولاد مبارکه', 'ذوب آهن', 'ناترازی انرژی', 'صادرات فولاد'
    ],
    'Non Ferrous Metals': [
        'آلومینیوم', 'فلزات غیرآهنی', 'شمش آلومینیوم', 'کنسانتره مس', 'شمش مس',
        'توسعه زنجیره مس', 'نیکل', 'سرب و روی', 'سهام ملی صنایع مس', 'سهام فملی',
        'معدن مس سرچشمه', 'شمش روی', 'کاتد مس', 'قیمت جهانی مس'
    ],
    'Water Industry': [
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
        Also sets up NLP models lazily.
        """
        self.keywords = keywords 
        self.processed_data = None
        self.engine = None
        
        # NLP Components (Lazy loaded)
        self.normalizer = None
        self.informal_normalizer = None
        self.tokenizer = None
        self.lemmatizer = None
        self.tagger = None
        self.stopwords = None
        
        # Initialize Hazm components
        self._setup_hazm()
        
        # Database setup
        if DB_CONFIG.get('DB_HOST'):    
            try:
                safe_user = quote_plus(DB_CONFIG['DB_USER'])
                safe_pass = quote_plus(DB_CONFIG['DB_PASS'])
                connection_str = (
                    f"mysql+pymysql://{safe_user}:{safe_pass}"
                    f"@{DB_CONFIG['DB_HOST']}:{DB_CONFIG['DB_PORT']}/{DB_CONFIG['DB_NAME']}"
                )
                self.engine = create_engine(connection_str)
                print(">> Database engine initialized successfully.")
            except Exception as e:
                print(f"Warning: Database connection failed ({e}). Running in offline mode only.")

    def _setup_hazm(self):
        """
        Configures Hazm models with optimized parameters for Telegram data.
        """
        try:
            # 1. Normalizer
            self.normalizer = Normalizer(
                correct_spacing=True,
                remove_diacritics=True,
                remove_specials_chars=True,
                decrease_repeated_chars=True,
                persian_style=True,
                persian_numbers=False, 
                unicodes_replacement=True,
                seperate_mi=True
            )
            
            # 2. Informal Normalizer
            self.informal_normalizer = InformalNormalizer()
            
            # 3. Tokenizer & Lemmatizer
            self.tokenizer = word_tokenize
            self.lemmatizer = Lemmatizer()
            
            # 4. Stopwords Setup (FINAL REFINED LIST)
            hazm_stops = stopwords_list()
            
            # A. Time & Date
            time_stops = [
                'سال', 'ماه', 'روز', 'هفته', 'ساعت', 'دقیقه', 'ثانیه', 'امروز', 'دیروز', 'فردا', 'امشب',
                'شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه',
                'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 
                'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند',
                'گذشته', 'آینده', 'کنونی', 'جاری', 'مدت', 'زمان', 'تاریخ'
            ]
            
            # B. Web & Social Media
            web_stops = [
                'http', 'https', 'www', 'com', 'ir', 'org', 'net', 'link', 'join', 'channel', 
                'id', 'admin', 'bot', 'click', 'site', 'website', 'instagram', 'telegram',
                'لینک', 'سایت', 'وبسایت', 'اینستاگرام', 'تلگرام', 'واتساپ', 'یوتیوب', 'اپلیکیشن',
                'عضو', 'عضویت', 'کانال', 'گروه', 'پیج', 'ادمین', 'ایدی', 'آیدی', 'پست', 'استوری'
            ]
            
            # C. Verbs & Abstract Nouns
            general_stops = [
                'هزار', 'میلیون', 'میلیارد', 'تومان', 'ریال', 'دلار', 'درصد', 'عدد', 'شماره',
                'گزارش', 'خبر', 'ادامه', 'تصویر', 'مطلب', 'صفحه', 'نسخه', 'منتشر', 'انتشار', 'منبع', 'لینک',
                'افزایش', 'کاهش', 'نیز', 'باید', 'شدن', 'داد', 'کرد', 'کند', 'است', 'بود', 'شد', 'گفت', 'وی',
                'این', 'آن', 'با', 'بر', 'برای', 'که', 'از', 'به', 'در', 'را', 'تا', 'چون', 'چه', 'اگر',
                'هست', 'نیست', 'دارد', 'داشت', 'می', 'نمی', 'های', 'ها', 'تر', 'ترین', 'می‌شود', 'می‌باشد',
                'نمی‌شود', 'خواهد', 'نخواهد', 'بوده', 'شده', 'میشود', 'میشوم', 'دارند', 'کنند', 'می‌کنند',
                'توانست', 'توانسته', 'انجام', 'جهت', 'دریافت', 'ارسال', 'تماس', 'پاسخ', 'سوال', 'قرار',
                'پایان', 'آغاز', 'شروع', 'مورد', 'بخش', 'حوزه', 'طی', 'طبق', 'برابر', 'سوی', 'ضمن',
                'کشور', 'استان', 'شهر', 'تهران', 'ایران', 'منطقه', 'محل', 'مکان', 'سراسر',
                'توسط', 'درباره', 'بنابر', 'همچنین', 'اما', 'ولی', 'لذا', 'چرا', 'خیر', 'بله',
                'اصل', 'آخر', 'اول', 'دوم', 'سوم', 'سایر', 'دیگر', 'کل', 'تمامی', 'برخی', 'بعضی',
                'عین', 'فقط', 'تنها', 'خیلی', 'بسیار', 'کاملا', 'واقعا', 'حتما', 'شاید',
                'خود', 'خویش', 'همین', 'همان', 'آنها', 'ایشان', 'ما', 'شما',
                'مشاوره', 'رایگان', 'تحویل', 'فوری', 'تضمینی', 'اقامت', 'ویژه'
                'کیش', 'قشم', 'بازی', 'مسابقه', 'دیدار', 'رقابت', 'جدول', 'هواداران',
                'کارشناس', 'صندوق', 'شناور', 'اتاق'
            ]
            
            self.stopwords = set(hazm_stops + time_stops + web_stops + general_stops)
            
            # 5. POS Tagger
            try:
                from hazm import POSTagger
                print(">> NLP: Initializing POS Tagger...")
                self.tagger = POSTagger(repo_id="roshan-research/hazm-postagger", model_filename="pos_tagger.model") # type: ignore
                print(">> NLP: POS Tagger loaded successfully.")
            except Exception as tag_err:
                self.tagger = None
                print(f">> NLP: Warning - POS Tagger failed ({tag_err}). Fallback enabled.")
                
        except Exception as e:
            print(f"Error setting up Hazm: {e}")
    
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
        Performs advanced NLP analysis with Context Filtering (Sports/Spam removal).
        """
        print(">> Starting NLP analysis (Global & Per Industry)...")
        freq_report = {}
        
        # --- 0. PRE-PROCESSING: Blacklists ---
        
        # A. Stopwords from Channel Names
        if 'channel_username' in self.processed_data.columns: # type: ignore
            channel_names = self.processed_data['channel_username'].astype(str).str.lower().unique().tolist() # type: ignore
            self.stopwords.update(channel_names) # type: ignore
            self.stopwords.update([f"@{name}" for name in channel_names]) # type: ignore
        
        # B. Context Blacklist (To remove entire posts if they are off-topic)
        # If a post contains these words, we assume it's sports/spam, not industry news.
        SPORTS_KEYWORDS = [
            'فوتبال', 'لیگ برتر', 'جام حذفی', 'سرمربی', 'دروازه‌بان', 'هافبک', 'مهاجم', 
            'پرسپولیس', 'استقلال', 'تراکتور', 'سپاهان', 'لیگ قهرمانان', 'فدراسیون فوتبال',
            'ورزشگاه', 'المپیک', 'مدال', 'قهرمانی', 'سوت پایان', 'هواداران'
        ]
        
        ADS_KEYWORDS = [
            'مشاوره رایگان', 'فالور', 'ممبر', 'وی‌پی‌ان', 'فیلترشکن', 'کاشت مو', 'مهاجرت تضمینی',
            'تور لحظه آخری', 'لاماری', 'اقامت' # Based on your debug file
        ]
        
        full_blacklist_pattern = '|'.join(SPORTS_KEYWORDS + ADS_KEYWORDS)

        # --- Internal Processing Function ---
        def process_text_batch(texts_list):
            local_counter = Counter()
            
            for txt in tqdm(texts_list, leave=False, desc="NLP Processing"):
                if not isinstance(txt, str): continue
                
                # 1. Normalization
                normalized = self.normalizer.normalize(txt) # type: ignore
                
                # 2. Tokenization
                tokens = self.tokenizer(normalized) # type: ignore
                valid_lemmas = []
                
                # 3. Tagging & Lemmatization
                if self.tagger:
                    try:
                        tagged = self.tagger.tag(tokens)
                        for word, tag in tagged:
                            # Keep Noun (N), Adjective (AJ)
                            if tag.startswith('N') or tag.startswith('AJ'):
                                lemma = self.lemmatizer.lemmatize(word) # type: ignore
                                if '#' in lemma: lemma = lemma.split('#')[0]
                                valid_lemmas.append(lemma)
                    except:
                        for word in tokens:
                            valid_lemmas.append(word)
                else:
                    for word in tokens:
                        lemma = self.lemmatizer.lemmatize(word) # type: ignore
                        if '#' in lemma: lemma = lemma.split('#')[0]
                        valid_lemmas.append(lemma)

                # 4. FINAL FILTERING (STRICT MODE)
                clean_tokens = []
                for t in valid_lemmas:
                    t_lower = t.lower()
                    
                    # A. Basic Stopword & Length
                    if t_lower in self.stopwords or len(t) < 3: continue
                    
                    # B. Numbers
                    if re.search(r'\d', t): continue
                    
                    # C. Web/IDs/Handles
                    if any(x in t_lower for x in ['http', 'www', '.com', '.ir', '@', 'id:', 'bot']): continue
                    
                    # D. Emojis & Symbols
                    if not re.match(r'^[آ-یa-zA-Z\u200c]+$', t): continue

                    # E. ENGLISH NOISE FILTER (NEW)
                    # If word is purely English:
                    if re.match(r'^[a-zA-Z]+$', t):
                        # 1. Drop if it's too long (Usernames usually > 8 chars, e.g. heratwomanshopping)
                        #    Industry terms like 'PVC', 'Forex', 'Steel' are usually short.
                        if len(t) > 7: continue 
                        
                        # 2. Drop specific English spam found in debug
                        if t_lower in ['landing', 'saamim', 'click', 'join', 'admin']: continue

                    clean_tokens.append(t)
                        
                local_counter.update(clean_tokens)
            return local_counter

        # 1. Per Industry Analysis
        for industry in self.keywords.keys():
            col_name = f"is_{industry}"
            if col_name not in self.processed_data.columns: continue # type: ignore
             
            industry_df = self.processed_data[self.processed_data[col_name] == True].copy() # type: ignore
            if industry_df.empty: continue
            
            # --- APPLY CONTEXT FILTER (Crucial Step) ---
            # Remove rows containing sports/spam keywords
            initial_count = len(industry_df)
            mask_noise = industry_df['text'].str.contains(full_blacklist_pattern, regex=True, na=False)
            industry_df = industry_df[~mask_noise]
            
            filtered_count = len(industry_df)
            if initial_count - filtered_count > 0:
                print(f"   -> {industry}: Filtered {initial_count - filtered_count} sports/ads posts.")
            
            print(f"   -> Analyzing Industry: {industry} ({filtered_count} posts)...")
            texts = industry_df['text'].dropna().astype(str).tolist()
            cnt = process_text_batch(texts)
            freq_report[industry] = dict(cnt.most_common(top_n))            
            
        # 2. Global Analysis (Apply same filter)
        print("   -> Analyzing Global (All Industries)...")
        industry_cols = [f"is_{k}" for k in self.keywords.keys() if f"is_{k}" in self.processed_data.columns] # type: ignore
        if industry_cols:
            global_mask = self.processed_data[industry_cols].any(axis=1) # type: ignore
            global_df = self.processed_data[global_mask].copy() # type: ignore
            
            # Filter Global too
            mask_noise_global = global_df['text'].str.contains(full_blacklist_pattern, regex=True, na=False)
            global_df = global_df[~mask_noise_global]
            
            if not global_df.empty:
                global_texts = global_df['text'].dropna().astype(str).tolist()
                global_cnt = process_text_batch(global_texts)
                freq_report['Global_All_Industries'] = dict(global_cnt.most_common(top_n))
            
        print(">> NLP analysis complete.")
        return freq_report
    
    
    def save_frequency_report(self, freq_report, filename="nlp_analysis_results.csv"):
        """
        Exports the word frequency analysis to a CSV file for manual inspection.
        Columns: Category (Industry), Word, Count
        """
        print(f">> Exporting NLP results to {filename}...")
        
        all_rows = []
        
        # Iterate through the dictionary structure
        for category, words_data in freq_report.items():
            if not words_data: continue
            
            # words_data is a dict like {'word': count, ...}
            for word, count in words_data.items():
                all_rows.append({
                    'Category': category,
                    'Word': word,
                    'Count': count
                })
        
        if all_rows:
            df_export = pd.DataFrame(all_rows)
            # encoding='utf-8-sig' is crucial for opening Persian CSVs in Excel correctly
            df_export.to_csv(filename, index=False, encoding='utf-8-sig')
            print(">> Export successful.")
        else:
            print(">> Warning: No data to export.")
    
    
    def plot_visualizations(self, stats_report, freq_report, keyword_breakdown):
        """
        Generates high-quality, publication-ready visualizations using Seaborn.
        """
        import matplotlib.font_manager as fm
        import numpy as np
        
        print(">> Generating professional visualizations...")
        
        # 1. Setup Style & Font
        sns.set_theme(style="whitegrid", context="talk") 
        
        # Path to the font file (Make sure to create 'assets' folder)
        font_path = os.path.join("assets", "Vazirmatn-Regular.ttf")
        if not os.path.exists(font_path):
            font_path = "Vazirmatn-Regular.ttf" # Fallback
            
        try:
            # Create FontProperties object for Matplotlib
            persian_font = fm.FontProperties(fname=font_path)
            print(f"   -> Using font: {os.path.basename(font_path)}")
        except:
            print("   -> Warning: Custom font not found. Persian text might be disjointed.")
            persian_font = None

        # Helper to apply font to all chart elements
        def apply_chart_style(ax, title, x_label, y_label):
            ax.set_title(make_farsi_text_readable(title), fontproperties=persian_font, fontsize=20, pad=20)
            ax.set_xlabel(make_farsi_text_readable(x_label), fontproperties=persian_font, fontsize=16)
            ax.set_ylabel(make_farsi_text_readable(y_label), fontproperties=persian_font, fontsize=16)
            
            for label in ax.get_xticklabels():
                label.set_fontproperties(persian_font)
            for label in ax.get_yticklabels():
                label.set_fontproperties(persian_font)

        # Helper to add value labels on bars
        def add_value_labels(ax, orient='h'):
            for p in ax.patches:
                if p.get_width() == 0 and p.get_height() == 0: continue
                
                if orient == 'h':
                    value = int(p.get_width())
                    x = p.get_width()
                    y = p.get_y() + p.get_height() / 2
                    ha = 'left'
                    va = 'center'
                    offset = (5, 0)
                else:
                    value = int(p.get_height())
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ha = 'center'
                    va = 'bottom'
                    offset = (0, 5)
                
                ax.annotate(f'{value:,}', (x, y), xytext=offset, 
                            textcoords='offset points', ha=ha, va=va, fontsize=12)
                
        # ---------------------------------------------------------
        # 1. Bar Chart: Total Posts per Industry
        # ---------------------------------------------------------
        if stats_report:
            data = []
            for k, v in stats_report.items():
                data.append({'Industry': k, 'Count': v['count']})
            df_chart = pd.DataFrame(data).sort_values('Count', ascending=False)
            
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(data=df_chart, x='Industry', y='Count', palette="viridis")
            
            # Fix X-axis labels
            labels = [make_farsi_text_readable(name) for name in df_chart['Industry']]
            ax.set_xticklabels(labels, rotation=45) # type: ignore
            
            apply_chart_style(ax, "تعداد کل پست‌ها به تفکیک صنعت", "صنعت", "تعداد پست")
            add_value_labels(ax, orient='v')
            
            plt.tight_layout()
            plt.savefig("1_industry_counts.png", dpi=300)
            plt.close()

        # ---------------------------------------------------------
        # 2. Bar Chart: Top Keywords per Industry
        # ---------------------------------------------------------
        if keyword_breakdown:
            for industry, keys_data in keyword_breakdown.items():
                if not keys_data: continue
                
                top_keys = dict(list(keys_data.items())[:15])
                df_keys = pd.DataFrame({
                    'Keyword': [make_farsi_text_readable(k) for k in top_keys.keys()],
                    'Count': list(top_keys.values())
                })
                
                # Dynamic Height
                fig_h = max(6, len(df_keys) * 0.5 + 2)
                plt.figure(figsize=(10, fig_h))
                
                ax = sns.barplot(data=df_keys, x='Count', y='Keyword', palette="rocket")
                
                apply_chart_style(ax, f"کلمات کلیدی پرتکرار در صنعت {industry}", "تعداد تکرار", "کلمه کلیدی")
                add_value_labels(ax, orient='h')
                
                plt.tight_layout()
                plt.savefig(f"2_keywords_{industry}.png", dpi=300)
                plt.close()

        # ---------------------------------------------------------
        # 3. Bar Chart: Top Channels per Industry
        # ---------------------------------------------------------
        if stats_report:
            for industry, data in stats_report.items():
                top_ch = data['top_channels']
                if top_ch.empty: continue
                
                df_ch = pd.DataFrame({
                    'Channel': top_ch.index,
                    'Views': top_ch.values
                })
                
                fig_h = max(6, len(df_ch) * 0.5 + 2)
                plt.figure(figsize=(10, fig_h))
                
                ax = sns.barplot(data=df_ch, x='Views', y='Channel', palette="mako")
                
                apply_chart_style(ax, f"برترین کانال‌ها (بازدید کل) در صنعت {industry}", "مجموع بازدید", "نام کانال")
                add_value_labels(ax, orient='h')
                
                plt.tight_layout()
                plt.savefig(f"3_top_channels_{industry}.png", dpi=300)
                plt.close()
                
        # ---------------------------------------------------------
        # 4. Word Clouds & Freq Bar Charts
        # ---------------------------------------------------------
        if freq_report:
            for group_name, freqs in freq_report.items():
                if not freqs: continue
                
                # A. Bar Chart (Top 20 Words)
                top_words = dict(list(freqs.items())[:20])
                df_words = pd.DataFrame({
                    'Word': [make_farsi_text_readable(k) for k in top_words.keys()],
                    'Count': list(top_words.values())
                })
                
                fig_h = max(8, len(df_words) * 0.5 + 2)
                plt.figure(figsize=(12, fig_h))
                
                palette = "magma" if group_name == 'Global_All_Industries' else "crest"
                ax = sns.barplot(data=df_words, x='Count', y='Word', palette=palette)
                
                title_text = f"پرتکرارترین کلمات در {group_name}"
                if group_name == 'Global_All_Industries':
                    title_text = "پرتکرارترین کلمات در کل صنایع (Global)"
                    
                apply_chart_style(ax, title_text, "تعداد تکرار", "کلمه")
                add_value_labels(ax, orient='h')
                
                plt.tight_layout()
                plt.savefig(f"4_barchart_freq_{group_name}.png", dpi=300)
                plt.close()

                # B. Word Cloud
                reshaped_freqs = {}
                if HAS_RESHAPER:
                    for k, v in freqs.items():
                        reshaped_k = get_display(arabic_reshaper.reshape(k)) # type: ignore
                        reshaped_freqs[reshaped_k] = v
                else:
                    reshaped_freqs = freqs
                
                wc = WordCloud(
                    width=1600, height=900, 
                    background_color='white', 
                    font_path=font_path,
                    colormap='viridis',
                    max_words=100
                )
                wc.generate_from_frequencies(reshaped_freqs)
                
                plt.figure(figsize=(16, 9))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                plt.title(make_farsi_text_readable(f"ابر کلمات: {group_name}"), # type: ignore
                          fontproperties=persian_font, fontsize=24, pad=20)
                
                plt.savefig(f"4_wordcloud_{group_name}.png", dpi=300)
                plt.close()

        # ---------------------------------------------------------
        # 5. Time Trend (Weekly)
        # ---------------------------------------------------------
        plt.figure(figsize=(14, 7))
        ax = plt.gca()
        
        has_data = False
        for industry in self.keywords.keys():
            col_name = f"is_{industry}"
            if col_name in self.processed_data.columns: # type: ignore
                df_ind = self.processed_data[self.processed_data[col_name] == True].copy() # type: ignore
                if not df_ind.empty:
                    weekly_counts = df_ind.resample('W', on='full_date').size()
                    label_text = make_farsi_text_readable(industry)
                    sns.lineplot(x=weekly_counts.index, y=weekly_counts.values, label=label_text, linewidth=2.5)
                    has_data = True
        
        if has_data:
            apply_chart_style(ax, "روند هفتگی تعداد پست‌ها", "تاریخ", "تعداد پست")
            
            leg = plt.legend(fontsize=12)
            if persian_font:
                for text in leg.get_texts():
                    text.set_fontproperties(persian_font)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("5_time_trend.png", dpi=300)
            plt.close()
            
        # ---------------------------------------------------------
        # 6. Bar Chart: Top Channels by Activity (Post Count) - Global & Industry
        # ---------------------------------------------------------
        # Prepare list of categories: Industries + Global
        categories_to_plot = list(self.keywords.keys()) + ['Global']
        
        for category in categories_to_plot:
            if category == 'Global':
                # Logic for Global: Posts that match ANY industry
                industry_cols = [f"is_{k}" for k in self.keywords.keys() if f"is_{k}" in self.processed_data.columns] # type: ignore
                if not industry_cols: continue
                mask = self.processed_data[industry_cols].any(axis=1) # type: ignore
                df_target = self.processed_data[mask] # type: ignore
                title_suffix = "در کل صنایع (Global)"
                color_palette = "viridis"
            else:
                # Logic for specific Industry
                col_name = f"is_{category}"
                if col_name not in self.processed_data.columns: continue # type: ignore
                df_target = self.processed_data[self.processed_data[col_name] == True] # type: ignore
                title_suffix = f"در صنعت {category}"
                color_palette = "flare"

            if df_target.empty: continue

            # Count posts per channel
            top_active_channels = df_target['channel_username'].value_counts().head(15)
            
            df_active = pd.DataFrame({
                'Channel': top_active_channels.index,
                'PostCount': top_active_channels.values
            })
            
            # Plot
            fig_h = max(6, len(df_active) * 0.5 + 2)
            plt.figure(figsize=(10, fig_h))
            
            ax = sns.barplot(data=df_active, x='PostCount', y='Channel', palette=color_palette)
            
            # Styling
            chart_title = f"فعال‌ترین کانال‌ها (تعداد پست مرتبط) {title_suffix}"
            apply_chart_style(ax, chart_title, "تعداد پست", "نام کانال")
            add_value_labels(ax, orient='h')
            
            plt.tight_layout()
            # File name handling
            safe_cat = "Global" if category == 'Global' else category
            plt.savefig(f"6_top_active_channels_{safe_cat}.png", dpi=300)
            plt.close()

        print(">> New chart (Activity Volume) saved successfully.")


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Robustly loads a malformed CSV where record separators are literal '\\n' strings
    instead of actual newlines.
    """
    logger.info(f"Loading and repairing raw file: {file_path}")
    
    try:
        # 1. Read the entire file as a raw string
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            
        # 2. SURGICAL FIX: Replace literal "\n" sequence with actual newline
        # The pattern in your file is: "views_value"\n"text_next_row"
        fixed_content = raw_content.replace('"\\n"', '"\n"')
        
        # 3. Parse the fixed content from memory
        df = pd.read_csv(
            io.StringIO(fixed_content),
            sep=',',
            quotechar='"',
            on_bad_lines='skip', # Skip truly broken lines if any remain
            encoding='utf-8'
        )
        
    except Exception as e:
        logger.error(f"Failed to load or fix CSV: {e}")
        return pd.DataFrame()

    initial_count = len(df)
    
    # 4. Standard Cleaning Pipeline
    
    # Ensure columns match expectations (fix any whitespace in names)
    df.columns = [c.strip() for c in df.columns]
    
    # Check if critical columns exist
    if 'full_date' not in df.columns or 'views' not in df.columns:
        logger.error(f"Critical columns missing. Found: {df.columns.tolist()}")
        return pd.DataFrame()

    # Convert Datetime
    df['full_date'] = pd.to_datetime(df['full_date'], errors='coerce')
    
    # Filter valid rows
    df = df.dropna(subset=['full_date'])
    
    # Convert Views
    df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)

    final_count = len(df)
    logger.info(f"Successfully repaired & loaded {final_count} records (dropped {initial_count - final_count}).")
    
    return df

        
if __name__ == "__main__":
    # --- Configuration ---
    CSV_FILENAME = "telegram_industry_data.csv"
    FORCE_FETCH = False  
    CSV_SEPARATOR = ',' 
    
    # Initialize Analyzer
    analyzer = TelegramIndustryAnalyzer(DB_CONFIG, INDUSTRY_KEYWORDS)
    
    try:
        # DECISION LOGIC: Fetch from DB or Load from Disk?
        data_loaded = False
        
        # Condition 1: Fetch if forced OR file doesn't exist
        if FORCE_FETCH or not os.path.exists(CSV_FILENAME):
            print(f">> Mode: ONLINE (Reason: FORCE_FETCH={FORCE_FETCH} or File Missing)")
            
            # Calculate dynamic dates (Last 1 year for example)
            today = datetime.now()
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
            
            # Robust Load using the new function
            analyzer.processed_data = load_and_clean_data(CSV_FILENAME)
            
            if analyzer.processed_data is not None and not analyzer.processed_data.empty:
                data_loaded = True
                print(f">> Loaded {len(analyzer.processed_data)} records from disk.")
            else:
                print("!! Warning: Loaded file is empty or corrupted.")

        # --- ANALYSIS PIPELINE ---
        if data_loaded and analyzer.processed_data is not None and not analyzer.processed_data.empty:
            
            # Step 1: Categorize Posts
            analyzer.categorize_posts()
            
            # Step 2: Generate Statistics
            stats = analyzer.generate_stats_report()
            
            # Step 3: Specific Keyword Breakdown
            keyword_stats = analyzer.analyze_keyword_breakdown()
            
            # Step 4: NLP & Word Cloud Analysis
            freq_stats = analyzer.analyze_word_frequency()
            
            # <--- NEW LINE: Export the results ---
            analyzer.save_frequency_report(freq_stats, "nlp_debug_output.csv")
            
            # Step 5: Visualize
            analyzer.plot_visualizations(stats, freq_stats, keyword_stats)
            
            print("\n>> Pipeline Finished Successfully. Check generated PNG files.")
            
        else:
            print(">> No data available to process. Check database connection or CSV file.")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()