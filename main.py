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
        'محدودیت گاز', 'قطع گاز', 'ناترازی گاز', 'آمونیاک', 'متانول',
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
            # 1. Normalizer & Models
            self.normalizer = Normalizer(correct_spacing=True, remove_diacritics=True, 
                                       remove_specials_chars=True, decrease_repeated_chars=True, 
                                       persian_style=True, persian_numbers=False, 
                                       unicodes_replacement=True, seperate_mi=True)
            self.informal_normalizer = InformalNormalizer()
            self.tokenizer = word_tokenize
            self.lemmatizer = Lemmatizer()
            
            # 2. Stopwords Setup
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
                'کشور', 'استان', 'شهر', 'منطقه', 'محل', 'مکان', 'سراسر',
                'توسط', 'درباره', 'بنابر', 'همچنین', 'اما', 'ولی', 'لذا', 'چرا', 'خیر', 'بله',
                'اصل', 'آخر', 'اول', 'دوم', 'سوم', 'سایر', 'دیگر', 'کل', 'تمامی', 'برخی', 'بعضی',
                'عین', 'فقط', 'تنها', 'خیلی', 'بسیار', 'کاملا', 'واقعا', 'حتما', 'شاید',
                'خود', 'خویش', 'همین', 'همان', 'آنها', 'ایشان', 'ما', 'شما',
                'مشاوره', 'رایگان', 'تحویل', 'فوری', 'تضمینی', 'اقامت', 'ویژه',
                'کیش', 'قشم', 'بازی', 'مسابقه', 'دیدار', 'رقابت', 'جدول', 'هواداران',
                'کارشناس', 'صندوق', 'شناور', 'اتاق'
            ]
            
            self.stopwords = set(hazm_stops + time_stops + web_stops + general_stops)
            

            # D. CONTEXT BLACKLISTS (Global Definition)
            self.sports_keywords = [
                'فوتبال', 'لیگ برتر', 'جام حذفی', 'سرمربی', 'دروازه‌بان', 'هافبک', 'مهاجم', 'مدافع',
                'پرسپولیس', 'استقلال', 'تراکتور', 'سپاهان', 'لیگ قهرمانان', 'فدراسیون فوتبال',
                'ورزشگاه', 'المپیک', 'مدال', 'قهرمانی', 'سوت پایان',
                'پنالتی', 'گل', 'بیفوما', 'آلومینیوم اراک', 'هتریک', 'هت‌تریک', 'آفساید', 'کرنر',
                'نیمه اول', 'نیمه دوم', 'وقت اضافه', 'داور', 'سوت', 'شوت', 'پاس گل', 'تیم ملی'
                'کمیته انضباطی', 'دربی', 'شهرآورد', 'میزبان', 'مهمان', 'تیم', 
                'قایقرانی', 'ورزشی', 'باشگاه'
            ]
            
            # E. ADVERTISEMENTS (Global Definition)
            self.ads_keywords = [
                'مشاوره رایگان', 'فالور', 'ممبر', 'وی‌پی‌ان', 'فیلترشکن', 'کاشت مو', 'مهاجرت تضمینی',
                'تور لحظه آخری', 'لاماری', 'اقامت', 'مهاجرت'
            ]
            
            # F. Channels to completely ignore in ALL stats and charts
            self.channel_blacklist = [
                'perspolisirfans', 
                # Add others here later if needed (e.g. 'IranintlTV' pending decision)
            ]
            
            # Pre-compile the regex pattern for performance
            self.blacklist_pattern = '|'.join(self.sports_keywords + self.ads_keywords)
            
            # 3. POS Tagger
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
        Calculates statistics for each industry, extracted AFTER removing:
        1. Blacklisted channels
        2. Context noise (sports/ads)
        """
        if self.processed_data is None: return {}
        
        print(">> Generating clean statistics (applying filters)...")
        report = {}
        
        self.translations = {
            'Petrochemical': 'پتروشیمی',
            'Steel_Chain': 'زنجیره فولاد',
            'Non_Ferrous_Metals': 'فلزات غیرآهنی',
            'Water_Industry': 'صنعت آب',
            'Mining': 'معدن',
            'Global': 'کل صنایع'
        }
        
        for industry in self.keywords.keys():
            col_name = f"is_{industry}"
            if col_name not in self.processed_data.columns: continue
            
            # 1. Select Industry Data
            industry_df = self.processed_data[self.processed_data[col_name] == True].copy()
            
            # 2. FILTER: Remove Blacklisted Channels
            if hasattr(self, 'channel_blacklist') and self.channel_blacklist:
                # Normalizing to lower case for comparison
                mask_channel = industry_df['channel_username'].astype(str).str.lower().isin([x.lower() for x in self.channel_blacklist])
                industry_df = industry_df[~mask_channel]
            
            # 3. FILTER: Apply Context Filter (Sports/Ads)
            if hasattr(self, 'blacklist_pattern') and self.blacklist_pattern:
                mask_noise = industry_df['text'].str.contains(self.blacklist_pattern, regex=True, na=False)
                industry_df = industry_df[~mask_noise]
            
            if industry_df.empty: continue

            # 4. Compute Stats on Cleaned Data
            post_count = len(industry_df)
            top_posts = industry_df.nlargest(20, 'views')[['full_date', 'channel_username', 'views', 'text']]
            top_channels = industry_df.groupby('channel_username')['views'].sum().nlargest(15)
            
            report[industry] = {
                'count': post_count,
                'top_posts': top_posts,
                'top_channels': top_channels,
                'clean_df': industry_df
            }
            
        return report
    
    
    def analyze_word_frequency(self, top_n=50):
        """
        Performs advanced NLP analysis with Channel & Context Filtering.
        """
        print(">> Starting NLP analysis (Global & Per Industry)...")
        freq_report = {}
        
        # --- 0. PRE-PROCESSING: Dynamic Blacklists ---
        
        # A. Stopwords from Channel Names
        if 'channel_username' in self.processed_data.columns: # type: ignore
            channel_names = self.processed_data['channel_username'].astype(str).str.lower().unique().tolist() # type: ignore
            self.stopwords.update(channel_names) # type: ignore
            self.stopwords.update([f"@{name}" for name in channel_names]) # type: ignore
        
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
                    
                    # A. Stopword & Length
                    if t_lower in self.stopwords or len(t) < 3: continue
                    
                    # B. Numbers
                    if re.search(r'\d', t): continue
                    
                    # C. Web/IDs
                    if any(x in t_lower for x in ['http', 'www', '.com', '.ir', '@', 'id:', 'bot']): continue
                    
                    # D. Emojis & Symbols
                    if not re.match(r'^[آ-یa-zA-Z\u200c]+$', t): continue

                    # E. English Noise (Usernames)
                    if re.match(r'^[a-zA-Z]+$', t):
                        if len(t) > 7: continue 
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
            
            # --- FILTER 1: Channel Blacklist ---
            if hasattr(self, 'channel_blacklist') and self.channel_blacklist:
                mask_channel = industry_df['channel_username'].astype(str).str.lower().isin([x.lower() for x in self.channel_blacklist])
                industry_df = industry_df[~mask_channel]
            
            if industry_df.empty: continue

            # --- FILTER 2: Context Filter (Sports/Ads) ---
            initial_count = len(industry_df)
            mask_noise = industry_df['text'].str.contains(self.blacklist_pattern, regex=True, na=False)
            industry_df = industry_df[~mask_noise]
            
            filtered_count = len(industry_df)
            # Only print if significantly filtered to reduce noise
            if initial_count - filtered_count > 5:
                print(f"   -> {industry}: NLP Filtered {initial_count - filtered_count} posts.")
            
            print(f"   -> Analyzing Industry: {industry} ({filtered_count} posts)...")
            texts = industry_df['text'].dropna().astype(str).tolist()
            cnt = process_text_batch(texts)
            freq_report[industry] = dict(cnt.most_common(top_n))            
            
        # 2. Global Analysis
        print("   -> Analyzing Global (All Industries)...")
        industry_cols = [f"is_{k}" for k in self.keywords.keys() if f"is_{k}" in self.processed_data.columns] # type: ignore
        if industry_cols:
            global_mask = self.processed_data[industry_cols].any(axis=1) # type: ignore
            global_df = self.processed_data[global_mask].copy() # type: ignore
            
            # --- FILTER 1: Channel Blacklist (Global) ---
            if hasattr(self, 'channel_blacklist') and self.channel_blacklist:
                mask_channel = global_df['channel_username'].astype(str).str.lower().isin([x.lower() for x in self.channel_blacklist])
                global_df = global_df[~mask_channel]

            # --- FILTER 2: Context Filter (Global) ---
            mask_noise_global = global_df['text'].str.contains(self.blacklist_pattern, regex=True, na=False)
            global_df = global_df[~mask_noise_global]
            
            if not global_df.empty:
                global_texts = global_df['text'].dropna().astype(str).tolist()
                global_cnt = process_text_batch(global_texts)
                freq_report['Global_All_Industries'] = dict(global_cnt.most_common(top_n))
            
        print(">> NLP analysis complete.")
        return freq_report
    
    
    def plot_visualizations(self, stats_report, freq_report, keyword_breakdown):
        """
        Generates publication-ready charts with fixed labels, Persian names, and thousands separators.
        """
        import matplotlib.font_manager as fm
        from matplotlib.ticker import FuncFormatter
        import numpy as np
        
        print(">> Generating professional visualizations...")
        sns.set_theme(style="white", context="talk")
        
        # Font Setup
        font_paths = [
            os.path.join("assets", "Vazirmatn-Regular.ttf"),
            "Vazirmatn-Regular.ttf"
        ]
        persian_font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    persian_font = fm.FontProperties(fname=fp)
                    print(f"   -> Found font: {fp}")
                    break
                except: continue
        
        if not persian_font:
            print("   -> Warning: Vazirmatn font not found. Charts may have issues.")

        # --- Helper: Thousands Separator Formatter ---
        def thousands_fmt(x, pos):
            return '{:,.0f}'.format(x)
        
        fmt_ticker = FuncFormatter(thousands_fmt)

        # --- Helper: Apply Style ---
        def apply_style_and_save(ax, title, xlabel, ylabel, filename, x_fmt=False, y_fmt=False):
            # Title: Centered, High Padding
            ax.set_title(make_farsi_text_readable(title), fontproperties=persian_font, 
                         fontsize=26, pad=30, loc='center', fontweight='bold')
            # Labels: Extra Padding
            ax.set_xlabel(make_farsi_text_readable(xlabel), fontproperties=persian_font, fontsize=18, labelpad=20)
            ax.set_ylabel(make_farsi_text_readable(ylabel), fontproperties=persian_font, fontsize=18, labelpad=20)
            
            # Ticks Font
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(persian_font)
                label.set_fontsize(14)
                
            # Ticks Font
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(persian_font)
                label.set_fontsize(14)
            
            if x_fmt: ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:,.0f}'.format(x)))
            if y_fmt: ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:,.0f}'.format(x)))
            
            sns.despine(left=True, bottom=False)
            ax.spines['bottom'].set_color('#333333')
            ax.grid(axis='x', color='#e0e0e0', linestyle='--', linewidth=0.8)
            
            # Save with tight bounding box to include padding
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()

        # --- Helper: Add Labels & Fix Overflow ---
        def add_labels(ax, orient='h'):
            # 1. Add Labels
            max_val = 0
            for p in ax.patches:
                val = p.get_width() if orient == 'h' else p.get_height()
                if val > max_val: max_val = val
                if val <= 0: continue
                
                # Format with comma
                label_text = f'{int(val):,}'
                if orient == 'h':
                    x, y = val, p.get_y() + p.get_height()/2
                    ax.annotate(label_text, (x, y), xytext=(8, 0), 
                                textcoords='offset points', ha='left', va='center', fontsize=12)
                else:
                    x, y = p.get_x() + p.get_width()/2, val
                    ax.annotate(label_text, (x, y), xytext=(0, 8), 
                                textcoords='offset points', ha='center', va='bottom', fontsize=12)
            
            # Tighten Axis Limits to avoid ugly empty space
            if orient == 'h': ax.set_xlim(0, max_val * 1.15)
            else: ax.set_ylim(0, max_val * 1.15)

        # ---------------------------------------------------------
        # 1. Bar Chart: Total Posts
        # ---------------------------------------------------------
        try:
            if stats_report:
                data = []
                for k, v in stats_report.items():
                    # Use Persian Name
                    fa_name = self.translations.get(k, k)
                    data.append({'Industry': fa_name, 'Count': v['count']})
                
                df_chart = pd.DataFrame(data).sort_values('Count', ascending=False)
                
                # 16:9 Ratio (approx 16 width, 9 height)
                plt.figure(figsize=(16, 9))
                ax = sns.barplot(data=df_chart, x='Industry', y='Count', palette="viridis")
                
                # Fix RTL Labels
                labels = [make_farsi_text_readable(name) for name in df_chart['Industry']]
                ax.set_xticklabels(labels, rotation=0) # No rotation needed if 16:9 # type: ignore
                
                add_labels(ax, orient='v')
                apply_style_and_save(ax, "تعداد کل پست‌ها به تفکیک صنعت", "صنعت", "تعداد پست", "1_industry_counts.png", y_fmt=True)
                print("   -> Chart 1 generated.")
                
        except Exception as e:
            print(f"!! Error generating Chart 1: {e}")

        # ---------------------------------------------------------
        # 2. Bar Chart: Top Keywords
        # ---------------------------------------------------------
        try:
            if keyword_breakdown:
                for industry, keys_data in keyword_breakdown.items():
                    if not keys_data: continue
                    
                    top_keys = dict(list(keys_data.items())[:15])
                    df_keys = pd.DataFrame({
                        'Keyword': [make_farsi_text_readable(k) for k in top_keys.keys()],
                        'Count': list(top_keys.values())
                    })
                    
                    # Aspect Ratio Logic: Fixed width 16, Height adapts but min 9
                    fig_h = max(9, len(df_keys) * 0.6) 
                    plt.figure(figsize=(16, fig_h))
                    
                    ax = sns.barplot(data=df_keys, x='Count', y='Keyword', palette="rocket")
                    
                    fa_ind = self.translations.get(industry, industry)
                    add_labels(ax, orient='h')
                    apply_style_and_save(ax, f"کلمات کلیدی پرتکرار در {fa_ind}", "تعداد تکرار", "کلمه کلیدی", 
                                       f"2_keywords_{industry}.png", x_fmt=True)
                print("   -> Chart 2 (Keywords) generated.")
        
        except Exception as e:
            print(f"!! Error generating Chart 2: {e}")

        # ---------------------------------------------------------
        # 3. Bar Chart: Top Channels by Views
        # ---------------------------------------------------------
        try:
            if stats_report:
                for industry, data in stats_report.items():
                    top_ch = data['top_channels']
                    if top_ch.empty: continue
                    
                    df_ch = pd.DataFrame({'Channel': top_ch.index, 'Views': top_ch.values})
                    fa_industry = self.translations.get(industry, industry)
                    
                    fig_h = max(9, len(df_ch) * 0.6) # Increased spacing factor to 0.6
                    plt.figure(figsize=(16, fig_h))
                    
                    ax = sns.barplot(data=df_ch, x='Views', y='Channel', palette="mako")
                    
                    add_labels(ax, orient='h')
                    apply_style_and_save(ax, f"پربازدیدترین کانال‌ها در {fa_industry}", "مجموع بازدید", "نام کانال", 
                                       f"3_top_channels_{industry}.png", x_fmt=True)
                print("   -> Chart 3 (Top Channels) generated.")
        except Exception as e:
            print(f"!! Error generating Chart 3: {e}")

        # ---------------------------------------------------------
        # 4. Frequency Charts & WordClouds
        # ---------------------------------------------------------
        try:
            if freq_report:
                for group_name, freqs in freq_report.items():
                    if not freqs: continue
                    
                    # A. Bar Chart
                    top_words = dict(list(freqs.items())[:20])
                    df_words = pd.DataFrame({
                        'Word': [make_farsi_text_readable(k) for k in top_words.keys()],
                        'Count': list(top_words.values())
                    })
                    
                    fig_h = max(9, len(df_words) * 0.6)
                    plt.figure(figsize=(16, fig_h))
                    
                    palette = "magma" if group_name == 'Global_All_Industries' else "crest"
                    ax = sns.barplot(data=df_words, x='Count', y='Word', palette=palette)
                    
                    # Translation logic for Global
                    fa_group = "کل صنایع" if group_name == 'Global_All_Industries' else self.translations.get(group_name, group_name)
                    
                    add_labels(ax, orient='h')
                    apply_style_and_save(ax, f"پرتکرارترین کلمات در {fa_group}", "تعداد تکرار", "کلمه", 
                                       f"4_barchart_freq_{group_name}.png", x_fmt=True)

                    # B. Word Cloud
                    if HAS_RESHAPER:
                        reshaped_freqs = {get_display(arabic_reshaper.reshape(k)): v for k, v in freqs.items()} # type: ignore
                    else:
                        reshaped_freqs = freqs
                    
                    # Ensure font path is valid
                    wc_font = font_paths[0] if os.path.exists(font_paths[0]) else (font_paths[1] if os.path.exists(font_paths[1]) else None)

                    wc = WordCloud(width=1920, height=1080, background_color='white', 
                                 font_path=wc_font,
                                 colormap='viridis', max_words=100)
                    wc.generate_from_frequencies(reshaped_freqs)
                    
                    plt.figure(figsize=(16, 9))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis("off")
                    
                    # Title with padding
                    plt.title(make_farsi_text_readable(f"ابر کلمات: {fa_group}"), # type: ignore
                              fontproperties=persian_font, fontsize=28, pad=30, loc='center')
                    
                    # CRITICAL FIX: Save with padding
                    plt.savefig(f"4_wordcloud_{group_name}.png", dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                print("   -> Chart 4 (NLP) generated with proper margins.")
        except Exception as e:
            print(f"!! Error generating Chart 4: {e}")

        # ---------------------------------------------------------
        # 5. Time Trend
        # ---------------------------------------------------------
        try:
            plt.figure(figsize=(16, 9))
            ax = plt.gca()
            has_data = False
            
            for industry in self.keywords.keys():
                col_name = f"is_{industry}"
                if col_name in self.processed_data.columns: # type: ignore
                    df_ind = self.processed_data[self.processed_data[col_name] == True].copy() # type: ignore
                    if not df_ind.empty:
                        if not pd.api.types.is_datetime64_any_dtype(df_ind['full_date']):
                            df_ind['full_date'] = pd.to_datetime(df_ind['full_date'], errors='coerce')
                        
                        # DYNAMIC FREQUENCY LOGIC
                        # 1. Calculate total duration
                        min_date = df_ind['full_date'].min()
                        max_date = df_ind['full_date'].max()
                        days_span = (max_date - min_date).days
                        
                        # 2. Divide by 7 to get roughly 7 intervals
                        # If span is small (e.g. 7 days), freq = 1 Day.
                        # If span is large (e.g. 70 days), freq = 10 Days.
                        freq_days = max(1, days_span // 7)
                        resample_rule = f'{freq_days}D'
                        
                        weekly_counts = df_ind.resample(resample_rule, on='full_date').size()
                        
                        label_text = make_farsi_text_readable(self.translations.get(industry, industry))
                        sns.lineplot(x=weekly_counts.index, y=weekly_counts.values, label=label_text, linewidth=3, marker='o')
                        has_data = True
            
            if has_data:
                leg = plt.legend(fontsize=14)
                if persian_font:
                    for text in leg.get_texts(): text.set_fontproperties(persian_font)
                
                apply_style_and_save(ax, "روند زمانی تعداد پست‌ها", "تاریخ", "تعداد پست", "5_time_trend.png", y_fmt=True)
                print("   -> Chart 5 (Dynamic Trend) generated.")
            else:
                print("   -> Chart 5 skipped (No data).")
        except Exception as e:
            print(f"!! Error generating Chart 5: {e}")

        # ---------------------------------------------------------
        # 6. Activity Volume
        # ---------------------------------------------------------
        try:
            # (Same logic as before, just updated styling)
            categories_to_plot = list(self.keywords.keys()) + ['Global']
            for category in categories_to_plot:
                if category == 'Global':
                    industry_cols = [f"is_{k}" for k in self.keywords.keys() if f"is_{k}" in self.processed_data.columns] # type: ignore
                    if not industry_cols: continue
                    # Apply regex filter to Global
                    mask_rel = self.processed_data[industry_cols].any(axis=1) # type: ignore
                    if hasattr(self, 'blacklist_pattern'):
                        mask_clean = ~self.processed_data['text'].str.contains(self.blacklist_pattern, regex=True, na=False) # type: ignore
                        df_target = self.processed_data[mask_rel & mask_clean] # type: ignore
                    else:
                        df_target = self.processed_data[mask_rel] # type: ignore
                    
                    title_suffix = "در کل صنایع"
                    color_palette = "viridis"
                    cat_name = "Global"
                else:
                    # Use cleaned data from stats_report if available
                    if category in stats_report:
                        df_target = stats_report[category]['clean_df']
                    else:
                        col_name = f"is_{category}"
                        if col_name not in self.processed_data.columns: continue # type: ignore
                        df_target = self.processed_data[self.processed_data[col_name] == True] # type: ignore
                    
                    title_suffix = f"در {self.translations.get(category, category)}"
                    color_palette = "flare"
                    cat_name = category

                if df_target.empty: continue

                top_active = df_target['channel_username'].value_counts().head(15)
                df_active = pd.DataFrame({'Channel': top_active.index, 'PostCount': top_active.values})
                
                fig_h = max(9, len(df_active) * 0.6)
                plt.figure(figsize=(16, fig_h))
                
                ax = sns.barplot(data=df_active, x='PostCount', y='Channel', palette=color_palette)
                
                add_labels(ax, orient='h')
                safe_cat = "Global" if cat_name == 'Global' else cat_name
                apply_style_and_save(ax, f"فعال‌ترین کانال‌ها (تعداد پست) {title_suffix}", "تعداد پست", "نام کانال", 
                                   f"6_active_channels_{safe_cat}.png", x_fmt=True)
            print("   -> Chart 6 (Activity) generated.")
        except Exception as e:
            print(f"!! Error generating Chart 6: {e}")
        
    
    def export_channel_audit(self, stats_report, filename="channel_audit_debug.csv"):
        """
        Exports sample posts from top channels to verify relevance.
        Use this to debug why 'IranIntl' or 'Perspolis' might appear in industry stats.
        """
        print(f">> Exporting channel audit data to {filename}...")
        audit_rows = []
        
        for industry, data in stats_report.items():
            top_channels = data['top_channels'].index.tolist()
            clean_df = data.get('clean_df')
            
            if clean_df is None or clean_df.empty: continue
            
            # For each top channel, grab their top 3 viewed posts in this industry
            for channel in top_channels:
                channel_posts = clean_df[clean_df['channel_username'] == channel].nlargest(3, 'views')
                
                for _, row in channel_posts.iterrows():
                    audit_rows.append({
                        'Industry': industry,
                        'Channel': channel,
                        'Views': row['views'],
                        'Date': row['full_date'],
                        'Snippet': row['text'][:100].replace('\n', ' ') + "..." # First 100 chars
                    })
        
        if audit_rows:
            pd.DataFrame(audit_rows).to_csv(filename, index=False, encoding='utf-8-sig')
            print(">> Audit file created. Check content to identify false positives.")
            
    
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
            
    
    def debug_specific_keyword(self, industry, keyword, filename="keyword_debug.csv"):
        """
        Exports posts containing a specific keyword within a specific industry.
        Useful for debugging why a generic word (like 'Aluminium') ranks high.
        """
        print(f">> Debugging keyword '{keyword}' in industry '{industry}'...")
        col_name = f"is_{industry}"
        
        if col_name not in self.processed_data.columns: # type: ignore
            print(f"   Error: Industry column {col_name} not found.")
            return

        # Filter: Industry matches AND text contains keyword
        mask_ind = self.processed_data[col_name] == True # type: ignore
        mask_key = self.processed_data['text'].str.contains(keyword, na=False) # type: ignore
        
        # Apply Channel Blacklist & Context Filter too (to see what remains AFTER filtering)
        clean_df = self.processed_data[mask_ind & mask_key].copy() # type: ignore
        
        if hasattr(self, 'channel_blacklist') and self.channel_blacklist:
             mask_ch = clean_df['channel_username'].astype(str).str.lower().isin([x.lower() for x in self.channel_blacklist])
             clean_df = clean_df[~mask_ch]

        if hasattr(self, 'blacklist_pattern') and self.blacklist_pattern:
             mask_noise = clean_df['text'].str.contains(self.blacklist_pattern, regex=True, na=False)
             clean_df = clean_df[~mask_noise]
        
        # Select relevant columns
        result = clean_df[['full_date', 'channel_username', 'text']].copy()
        
        if not result.empty:
            result.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f">> Exported {len(result)} posts containing '{keyword}' to {filename}.")
        else:
            print(f">> No posts found with keyword '{keyword}' after filtering.")



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
            
            # Check why some words are out of proportion
            analyzer.save_frequency_report(freq_stats, "nlp_debug_output.csv")
            
            # Check why 'Aluminium' is #1 in Non_Ferrous_Metals
            analyzer.debug_specific_keyword('Non_Ferrous_Metals', 'آلومینیوم', 'aluminium_debug.csv')
            
            # Step 5: Visualize
            analyzer.plot_visualizations(stats, freq_stats, keyword_stats)
            
            # Step 6: Export Channel Audit
            analyzer.export_channel_audit(stats, "channel_audit.csv")
            
            print("\n>> Pipeline Finished Successfully. Check generated PNG files.")
            
        else:
            print(">> No data available to process. Check database connection or CSV file.")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()