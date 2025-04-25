
# estimate_manager.py

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheet 백업 함수
def backup_to_google_sheet(df, sheet_name='금형백업'):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("project11-457901-d742c683d428.json", scope)
        client = gspread.authorize(creds)

        try:
            sheet = client.open(sheet_name).sheet1
        except gspread.SpreadsheetNotFound:
            sh = client.create(sheet_name)
            sh.share('your_email@example.com', perm_type='user', role='writer')
            sheet = sh.sheet1

        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
        st.success("✅ Google Sheet에 백업 완료되었습니다.")
    except Exception as e:
        st.error(f"❌ Google Sheet 백업 중 오류 발생: {e}")

# DB 초기화
@st.cache_resource
def init_db():
    conn = sqlite3.connect("estimate.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS molds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT,
            name TEXT,
            make_date TEXT,
            manufacturer TEXT,
            status TEXT,
            location TEXT,
            note TEXT,
            standard TEXT,
            category TEXT,
            part TEXT,
            model_name TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mold_location_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mold_id INTEGER,
            이전위치 TEXT,
            변경위치 TEXT,
            변경일시 TEXT
        )
    """)
    conn.commit()
    return conn

# Streamlit 페이지 설정
st.set_page_config(page_title="금형 관리 시스템", layout="wide")

# 메인 실행
def main():
    st.title("🛠 금형 관리 시스템")
    st.info("본 시스템은 금형 데이터를 관리하고 Google Sheet 백업을 지원합니다.")

    conn = init_db()
    df = pd.read_sql_query("SELECT * FROM molds", conn)

    if st.button("🔁 Google Sheet로 백업"):
        backup_to_google_sheet(df)

    st.write("금형 목록")
    st.dataframe(df)

if __name__ == "__main__":
    main()
