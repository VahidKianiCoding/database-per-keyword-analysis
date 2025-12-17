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
