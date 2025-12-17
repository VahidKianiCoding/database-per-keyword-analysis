import pandas as pd
import sqlalchemy
from sqlalchemy import text
from datetime import datetime, timedelta
import re
from tqdm import tqdm # For progress bar

class TelegramDataPipeline:
    def __init__(self, db_connection_str):
        """
        Initialize the pipeline with database connection string.
        Format: mysql+pymysql://user:password@host:port/dbname
        """
        self.engine = sqlalchemy.create_engine(db_connection_str)
        
        # Define keywords for each industry
        self.industries = {
            "Petrochemical": [
                "پتروشیمی خلیج فارس", "تحریم پتروشیمی", "خوراک پتروشیمی", "خوراک گاز",
                "محدودیت گاز", "قطع گاز", "ناترازی گاز", "اوره", "آمونیاک", "متانول",
                "پتروپالایش", "رفع آلایندگی", "بنزین پتروشیمی", "گاز طبیعی", "صنعت پتروشیمی"
            ],
            "Steel_Chain": [
                "صنایع فولاد", "انرژی فولاد", "گاز فولاد", "آلیاژ", "صنعت فولاد",
                "ورق فولادی", "آهن اسفنجی", "کنسانتره سنگ آهن", "تیرآهن", "فولاد ایران",
                "شمش فولاد", "زنجیره مس", "شمش فولادی", "مدیریت فولاد", "مواد اولیه",
                "فولاد خوزستان", "فولاد مبارکه", "ذوب آهن", "ناترازی انرژی", "صادرات فولاد"
            ],
            "Non_Ferrous_Metals": [
                "آلومینیوم", "فلزات غیرآهنی", "شمش آلومینیوم", "کنسانتره مس", "شمش مس",
                "توسعه زنجیره مس", "نیکل", "روی", "سهام ملی صنایع مس", "سهام فملی",
                "معدن مس سرچشمه", "شمش روی", "کاتد مس", "قیمت جهانی مس"
            ],
            "Water_Industry": [
                "بحران آب", "انتقال آب دریا", "انتقال آب خلیج فارس به فلات مرکزی", "مدیریت آب",
                "آلودگی آب", "قطعی آب", "زاینده‌رود", "آب شیرین‌کن", "فرونشست زمین",
                "مدیریت منابع آب", "آبخیزداری", "آب شیرین‌کن دریایی", "آبفا", "تصفیه فاضلاب",
                "لایروبی", "بارورسازی ابرها", "سفره‌های آب زیرزمینی", "حق آبه", "بحران کم‌آبی"
            ],
            "Mining": [
                "سنگ آهن", "کنسانتره", "گندله", "معدن طلا", "ایمیدرو", "حفاری اکتشافی",
                "ماشین‌آلات معدنی", "دامپتراک", "فلوتاسیون", "لیچینگ", "پروانه بهره‌برداری", "زغال سنگ"
            ]
        }
        
        # Pre-compile regex patterns for performance (Huge speedup)
        self.compiled_patterns = {
            industry: re.compile('|'.join(keywords)) 
            for industry, keywords in self.industries.items()
        }

    
    def fetch_data_by_month(self, start_date, months_back=12):
            """
            Fetches data month by month to avoid memory/network overload.
            """
            all_relevant_data = []
            
            end_date = datetime.strptime(start_date, "%Y-%m-%d")
            # Go back 'months_back' months from start_date
            # We process from Past -> Present or Present -> Past. 
            # Let's do month by month chunks.
            
            print(f"🚀 Starting extraction for the last {months_back} months...")
            
            for i in tqdm(range(months_back)):
                # Calculate time window for this chunk
                month_end = end_date - timedelta(days=30 * i)
                month_start = month_end - timedelta(days=30)
                
                query = f"""
                SELECT text, full_date, channel_username, views
                FROM telegram_channel_post
                WHERE full_date >= '{month_start.strftime('%Y-%m-%d')}' 
                  AND full_date < '{month_end.strftime('%Y-%m-%d')}'
                """
                
                try:
                    # Execute query
                    df_chunk = pd.read_sql(query, self.engine)
                    
                    if not df_chunk.empty:
                        # Filter in memory (Fast with 64GB RAM)
                        processed_chunk = self._filter_and_tag(df_chunk)
                        if not processed_chunk.empty:
                            all_relevant_data.append(processed_chunk)
                            
                except Exception as e:
                    print(f"❌ Error fetching data for {month_start} to {month_end}: {e}")
    
            if all_relevant_data:
                final_df = pd.concat(all_relevant_data, ignore_index=True)
                print(f"✅ Data Extraction Complete. Total relevant posts found: {len(final_df)}")
                return final_df
            else:
                print("⚠️ No relevant data found.")
                return pd.DataFrame()
    
    def _filter_and_tag(self, df):
        """
        Checks each row against all industry patterns.
        A post can belong to multiple industries.
        """
        # We need to drop rows with no text first
        df = df.dropna(subset=['text'])
        
        # We will create a list to store indices of rows that match ANY category
        # And also store the categories they matched
        
        matches = []
        
        for index, row in df.iterrows():
            post_text = row['text']
            matched_industries = []
            
            for industry, pattern in self.compiled_patterns.items():
                if pattern.search(post_text):
                    matched_industries.append(industry)
            
            if matched_industries:
                # If the post matches at least one industry, keep it
                # We add a new column 'industries' which is a list of matched categories
                row_data = row.to_dict()
                row_data['matched_industries'] = matched_industries
                matches.append(row_data)
        
        return pd.DataFrame(matches)
