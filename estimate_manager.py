import streamlit as st
# 페이지 설정
st.set_page_config(page_title="견적서 관리 시스템", layout="wide")
import sqlite3
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SHEET_CREDENTIALS = "project11-457901-d742c683d428.json"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

credentials = Credentials.from_service_account_file(SHEET_CREDENTIALS, scopes=SCOPES)
gc = gspread.authorize(credentials)

try:
    sheet_estimate = gc.open("견적서백업").sheet1
    sheet_mold = gc.open("금형백업").sheet1
except Exception as e:
    st.error(f"❌ Google Sheet 연결 실패: {e}")


from datetime import datetime


# DB 초기화
import sqlite3
import streamlit as st

# DB 초기화 함수
@st.cache_resource
def init_db():
    conn = sqlite3.connect("estimate.db", check_same_thread=False)
    cursor = conn.cursor()

    # 견적 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS estimates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT,
            date TEXT,
            model TEXT,
            category TEXT,
            product TEXT,
            price REAL,
            final_price REAL
        )
    """)

    # 위치 변경 이력 테이블 생성
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

# 실제 연결 및 커서 객체 생성
conn = init_db()
cursor = conn.cursor()


# 견적서 등록
def add_estimate():
    st.subheader("📋 견적서 등록")
    with st.form("register_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            company = st.text_input("상호")
            model = st.text_input("모델")
            product = st.text_input("품명")
        with col2:
            category = st.text_input("카테고리")
            price = st.number_input("견적가", min_value=0)
            final_price = st.number_input("결정가", min_value=0)
        with col3:
            date = st.date_input("날짜", value=datetime.today())

        submitted = st.form_submit_button("등록하기")
        if submitted:
            cursor.execute("""
                INSERT INTO estimates (company, date, model, category, product, price, final_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (company, date.strftime("%Y-%m-%d"), model, category, product, price, final_price))
            conn.commit()
            st.success("✅ 견적서가 등록되었습니다.")

# 엑셀 업로드
def upload_excel():
    st.subheader("📤 엑셀 업로드")
    uploaded_files = st.file_uploader("엑셀 파일을 업로드하세요 (다중 선택 가능)", type=["xlsx"], accept_multiple_files=True)

    if uploaded_files:
        if 'uploaded_log' not in st.session_state:
            st.session_state.uploaded_log = []

        for file in uploaded_files:
            if file.name in st.session_state.uploaded_log:
                continue  # 중복 업로드 방지
            st.session_state.uploaded_log.append(file.name)

            df = pd.read_excel(file, header=None)
            target_columns = ["상호", "날짜", "모델", "카테고리", "구분", "품명", "견적가", "결정가"]
            header_idx = None
            for i, row in df.iterrows():
                match_count = sum(str(cell).strip() in target_columns for cell in row)
                if match_count >= 5:
                    header_idx = i
                    break

            if header_idx is None:
                st.error(f"❌ [{file.name}] 헤더를 찾을 수 없습니다.")
                continue

            df.columns = df.iloc[header_idx]
            df = df.iloc[header_idx + 1:].copy()
            df.columns = df.columns.str.strip()
            df = df.dropna(subset=["모델", "품명", "견적가"])

            rename_map = {
                "상호": "company",
                "날짜": "date",
                "모델": "model",
                "카테고리": "category",
                "구분": "category",
                "품명": "product",
                "견적가": "price",
                "결정가": "final_price"
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            expected = ["company", "date", "model", "category", "product", "price", "final_price"]
            missing = [col for col in expected if col not in df.columns]
            if missing:
                st.error(f"❌ [{file.name}] 누락된 필수 컬럼: {missing}")
                continue

            df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.strftime('%Y-%m-%d')
            df["price"] = pd.to_numeric(df["price"], errors='coerce').fillna(0)
            df["final_price"] = pd.to_numeric(df["final_price"], errors='coerce').fillna(0)
            df = df.drop_duplicates()

            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO estimates (company, date, model, category, product, price, final_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["company"], row["date"], row["model"], row["category"],
                    row["product"], row["price"], row["final_price"]
                ))
            conn.commit()
            st.success(f"✅ '{file.name}' 업로드 및 등록 완료")

# 견적서 목록 보기
def show_estimates():
    st.subheader("📄 견적서 목록 보기")

    df = pd.read_sql_query("SELECT * FROM estimates", conn)

    if df.empty:
        st.info("등록된 견적서가 없습니다.")
        return

    # 최고가/최저가 표시
    max_prices = df.groupby(['model', 'product'])['price'].transform('max')
    min_prices = df.groupby(['model', 'product'])['price'].transform('min')
    df['max_price_flag'] = df['price'] == max_prices
    df['min_price_flag'] = df['price'] == min_prices

    def style_price(row):
        val = int(round(row['price']))
        if row['max_price_flag']:
            return f"<span style='color:red;font-weight:bold'>{val:,}</span>"
        elif row['min_price_flag']:
            return f"<span style='color:blue;font-weight:bold'>{val:,}</span>"
        else:
            return f"{val:,}"

    df['견적가'] = df.apply(style_price, axis=1)
    df['결정가'] = df['final_price'].apply(lambda x: f"{int(round(x)):,}")
    df['견적가(숫자)'] = df['price'].apply(lambda x: int(round(x)))
    df['결정가(숫자)'] = df['final_price'].apply(lambda x: int(round(x)))

    # ✅ 선택 삭제용 Editor
    editor_df = df[['id', 'company', 'model', 'category', 'product', '견적가(숫자)', '결정가(숫자)', 'date']].copy()
    editor_df.columns = ['ID', '상호', '모델', '구분', '품명', '견적가', '결정가', '날짜']
    editor_df.insert(1, '선택', False)

    selected = st.data_editor(editor_df, use_container_width=True, hide_index=True)
    selected_ids = selected[selected['선택'] == True]['ID'].tolist()

    if selected_ids:
        if st.button("🗑️ 선택 항목 삭제"):
            cursor.executemany("DELETE FROM estimates WHERE id = ?", [(i,) for i in selected_ids])
            conn.commit()
            st.success(f"✅ {len(selected_ids)}개 항목이 삭제되었습니다.")
            st.rerun()

    with st.expander("⚠ 전체 삭제", expanded=False):
        st.warning("모든 견적서를 삭제합니다. 정말 삭제하시겠습니까?")
        if st.button("🔴 전체 견적서 삭제"):
            cursor.execute("DELETE FROM estimates")
            conn.commit()
            st.success("📛 전체 견적서가 삭제되었습니다.")
            st.rerun()

    # ✅ 보기용 테이블
    st.markdown("---")
    st.subheader("👁 견적서 보기 (모델 필터 + 강조)")

    model_list = sorted(df['model'].dropna().unique())
    selected_models = st.multiselect("모델을 선택하세요", model_list, default=model_list)

    filtered_df = df[df['model'].isin(selected_models)]
    styled_df = filtered_df[['company', 'model', 'category', 'product', '견적가', '결정가', 'date']].copy()
    styled_df.columns = ['상호', '모델', '구분', '품명', '견적가', '결정가', '날짜']

    st.markdown("""
        <style>table td span { font-weight: bold; }</style>
        <div>※ 최고가는 <span style='color:red'>빨간색</span>, 최저가는 <span style='color:blue'>파란색</span>으로 표시됩니다.</div>
    """, unsafe_allow_html=True)
    st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ✅ 견적가 비교표 (차이 포함, 카테고리별 소계 + 총합계 강조)
    import re

    st.markdown("---")
    st.subheader("📊 견적가 비교표 (모델 + 구분 + 품명 + 날짜 기준, 차이 금액 포함 + 소계/총합계 강조)")

    model_options = sorted(df['model'].dropna().unique())
    category_options = sorted(df['category'].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect("📌 모델 선택", model_options, default=model_options)
    with col2:
        selected_categories = st.multiselect("📌 구분(카테고리) 선택", category_options, default=category_options)

    filtered = df[
        df['model'].isin(selected_models) &
        df['category'].isin(selected_categories)
    ]

    if filtered.empty:
        st.info("조건에 맞는 견적서가 없습니다.")
        return

    pivot = filtered.pivot_table(
        index=["model", "category", "product", "date"],
        columns="company",
        values="price",
        aggfunc="min"
    ).reset_index()

    price_cols = pivot.columns.difference(["model", "category", "product", "date"])
    min_price = pivot[price_cols].min(axis=1)
    diff_df = pivot[price_cols].subtract(min_price, axis=0)

    def format_price_with_diff(price, diff):
        if pd.isna(price):
            return ""
        price_fmt = f"{int(round(price)):,}"
        if diff > 0:
            return f"{price_fmt} <span style='color:red'>(+{int(round(diff)):,})</span>"
        return price_fmt

    styled_rows = []
    for category, group_df in pivot.groupby("category"):
        group_diff = diff_df.loc[group_df.index]

        for i in group_df.index:
            row = group_df.loc[i, ["model", "category", "product", "date"]].to_dict()
            for col in price_cols:
                row[col] = format_price_with_diff(group_df.at[i, col], group_diff.at[i, col])
            styled_rows.append(row)

        subtotal = {
            'model': '소계', 'category': category, 'product': '', 'date': ''
        }
        for col in price_cols:
            subtotal[col] = format_price_with_diff(group_df[col].sum(), group_diff[col].sum())
        styled_rows.append(subtotal)

    total_row = {
        'model': '총합계', 'category': '', 'product': '', 'date': ''
    }
    for col in price_cols:
        total_row[col] = format_price_with_diff(pivot[col].sum(), diff_df[col].sum())
    styled_rows.append(total_row)

    styled_df = pd.DataFrame(styled_rows)

    # ✅ HTML 스타일 강조 (소계 노란색, 총합계 하늘색)
    html = styled_df.to_html(escape=False, index=False)

    html = re.sub(
        r'(<tr[^>]*?>\s*<td[^>]*?>소계</td>)',
        r'<tr style="background-color:#fff8b3;font-weight:bold">\1',
        html,
        flags=re.DOTALL
    )

    html = re.sub(
        r'(<tr[^>]*?>\s*<td[^>]*?>총합계</td>)',
        r'<tr style="background-color:#d9edf7;font-weight:bold">\1',
        html,
        flags=re.DOTALL
    )

    st.markdown("※ 견적가 + 차이금액 형식 (예: 13,000 <span style='color:red'>(+2,000)</span>)", unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)




    
    # ✅ HTML 테이블 출력 이후 피벗 테이블이 추가되었다면
    pivot = filtered.pivot_table(
        index=["model", "category", "product", "date"],
        columns="company",
        values="price",
        aggfunc='min'
    )

    pivot = pivot.apply(pd.to_numeric, errors='coerce')

    styled_pivot = pivot.style.format("{:,.0f}").highlight_min(axis=1, props='color:blue;font-weight:bold')

    st.markdown("※ 동일 모델 + 구분 + 품명 조합에 대해 상호별 견적가를 비교합니다. 최저가는 파란색으로 강조됩니다.")
    st.dataframe(styled_pivot, use_container_width=True)


# 비교 분석
def compare_estimates():
    st.subheader("📊 견적서 비교 분석 (모델 + 구분별 총합 및 참여상호 수)")

    df = pd.read_sql_query("SELECT * FROM estimates", conn)
    if df.empty:
        st.info("비교할 견적서가 없습니다.")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ 날짜 필터
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 시작 날짜", value=df['date'].min().date())
    with col2:
        end_date = st.date_input("📅 종료 날짜", value=df['date'].max().date())

    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    # ✅ 모델 필터
    model_list = sorted(df['model'].dropna().unique())
    selected_models = st.multiselect("📌 모델 선택", model_list, default=model_list)
    df = df[df['model'].isin(selected_models)]

    if df.empty:
        st.warning("선택한 조건에 맞는 데이터가 없습니다.")
        return

    # ✅ 제품별 최저가/최고가 계산
    min_prices = df.groupby(['model', 'category', 'product'])['price'].min().reset_index(name='최저가')
    max_prices = df.groupby(['model', 'category', 'product'])['price'].max().reset_index(name='최고가')
    merged = pd.merge(min_prices, max_prices, on=['model', 'category', 'product'])

    # ✅ 모델 + 카테고리별 총합 및 평균가 계산
    summary = merged.groupby(['model', 'category']).agg(
        최저가합계=('최저가', 'sum'),
        최고가합계=('최고가', 'sum')
    ).reset_index()
    summary['평균가'] = ((summary['최저가합계'] + summary['최고가합계']) / 2).astype(int)

    # ✅ 합계 행 추가
    total_row = {
        'model': '총합계',
        'category': '',
        '최저가합계': summary['최저가합계'].sum(),
        '최고가합계': summary['최고가합계'].sum(),
        '평균가': summary['평균가'].sum()
    }
    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    # ✅ 금액 포맷
    for col in ['최저가합계', '최고가합계', '평균가']:
        summary[col] = summary[col].apply(lambda x: f"{int(round(x)):,}")

    # ✅ 요약 테이블 출력
    st.markdown("### 📊 모델 + 구분별 견적 비교 요약 (하단에 총합 포함)")
    st.markdown("※ 평균가는 (최저가합계 + 최고가합계) ÷ 2 기준입니다.")
    st.dataframe(summary, use_container_width=True)

       # ✅ 상세 보기: 제품별 최저/최고가 참여 상호명 표시
    st.markdown("---")
    st.subheader("📂 상세 보기: 제품별 최저/최고가 참여 상호명")

    for (model, category) in summary[['model', 'category']].dropna().drop_duplicates().values:
        with st.expander(f"📦 {model} / {category} 참여 상호 보기"):
            subset = df[(df['model'] == model) & (df['category'] == category)]

            result = []
            for product in subset['product'].unique():
                product_df = subset[subset['product'] == product]
                min_price = product_df['price'].min()
                max_price = product_df['price'].max()

                # ✅ 참여한 실제 상호명만 표시
                min_companies = product_df[product_df['price'] == min_price]['company'].unique()
                max_companies = product_df[product_df['price'] == max_price]['company'].unique()

                min_display = ', '.join(sorted(min_companies))
                max_display = ', '.join(sorted(max_companies))

                result.append({
                    '제품': product,
                    '최저가 상호': min_display,
                    '최고가 상호': max_display
                })

            detail_df = pd.DataFrame(result)
            st.dataframe(detail_df, use_container_width=True)

# 금형관리 기능
# 금형 테이블 생성 함수 (전역으로 분리)

# 금형관리 화면
def mold_management():
    st.subheader("🛠 금형관리")
def init_mold_db():
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
    conn.commit()
def backup_estimate_to_sheet(row_dict):
    row = [
        row_dict.get("company"), row_dict.get("date"), row_dict.get("model"),
        row_dict.get("category"), row_dict.get("product"),
        row_dict.get("price"), row_dict.get("final_price")
    ]
    sheet_estimate.append_row(row)

def backup_mold_to_sheet(row_dict):
    row = [
        row_dict.get("code"), row_dict.get("name"), row_dict.get("make_date"),
        row_dict.get("manufacturer"), row_dict.get("status"), row_dict.get("location"),
        row_dict.get("note"), row_dict.get("standard"), row_dict.get("category"),
        row_dict.get("part"), row_dict.get("model_name")
    ]
    sheet_mold.append_row(row)
def mold_management():
    st.subheader("🛠 금형관리")
    init_mold_db()

    # 📤 엑셀 업로드
    st.markdown("### 📥 엑셀로 금형정보 업로드")
    uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        required_cols = ['금형코드', '금형명', '제작일자', '제작사', '사용상태',
                         '보관위치', '품명', '기준값', '상품군', '파트부', '모델명']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ 필수 컬럼 누락: {required_cols}")
            return

        if st.button("📥 엑셀 정보 등록"):
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO molds (code, name, make_date, manufacturer, status, location, note, standard, category, part, model_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['금형코드'], row['금형명'], str(row['제작일자'])[:10],
                    row['제작사'], row['사용상태'], row['보관위치'], row['품명'],
                    row['기준값'], row['상품군'], row['파트부'], row['모델명']
                ))
            conn.commit()
            st.success("✅ 금형 정보가 등록되었습니다.")
            st.rerun()

    # ➕ 수동 등록
    st.markdown("### ➕ 수동 금형 등록")
    with st.form("manual_mold"):
        cols = st.columns(3)
        code = cols[0].text_input("금형코드")
        name = cols[1].text_input("금형명")
        make_date = cols[2].text_input("제작일자 (YYYY-MM-DD)")

        manufacturer = cols[0].text_input("제작사")
        status = cols[1].text_input("사용상태")
        location = cols[2].text_input("보관위치")

        note = cols[0].text_input("품명")
        standard = cols[1].text_input("기준값")
        category = cols[2].text_input("상품군")

        part = cols[0].text_input("파트부")
        model_name = cols[1].text_input("모델명")

        if st.form_submit_button("등록"):
            cursor.execute("""
                INSERT INTO molds (code, name, make_date, manufacturer, status, location, note, standard, category, part, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                code, name, make_date, manufacturer, status, location,
                note, standard, category, part, model_name
            ))
            conn.commit()
            st.success("✅ 금형 정보가 등록되었습니다.")
            st.rerun()

    # 📋 등록된 금형 목록
    st.markdown("---")
    st.subheader("📋 등록된 금형 목록 (수정 및 삭제 가능)")

    df_mold = pd.read_sql_query("SELECT * FROM molds", conn)
    if df_mold.empty:
        st.info("등록된 금형 정보가 없습니다.")
        return

    df_edit = df_mold.copy()
    df_edit['선택'] = False
    df_edit_display = df_edit.rename(columns={
        'code': '금형코드', 'name': '금형명', 'make_date': '제작일자',
        'manufacturer': '제작사', 'status': '사용상태', 'location': '보관위치',
        'note': '품명', 'standard': '기준값', 'category': '상품군',
        'part': '파트부', 'model_name': '모델명'
    })

    edited = st.data_editor(df_edit_display[['선택', 'id', '금형코드', '금형명', '제작일자', '제작사',
                                             '사용상태', '보관위치', '품명', '기준값', '상품군', '파트부', '모델명']],
                            use_container_width=True, hide_index=True, num_rows="dynamic")

    # ✅ 삭제 기능
    selected_ids = edited[edited['선택'] == True]['id'].tolist()
    if selected_ids and st.button("🗑️ 선택 항목 삭제"):
        cursor.executemany("DELETE FROM molds WHERE id = ?", [(i,) for i in selected_ids])
        conn.commit()
        st.success(f"✅ {len(selected_ids)}개의 금형 정보가 삭제되었습니다.")
        st.rerun()

    # ✅ 수정 기능
    if st.button("💾 수정사항 저장"):
        for _, row in edited.iterrows():
            cursor.execute("""
                UPDATE molds SET
                    code = ?, name = ?, make_date = ?, manufacturer = ?, status = ?,
                    location = ?, note = ?, standard = ?, category = ?, part = ?, model_name = ?
                WHERE id = ?
            """, (
                row['금형코드'], row['금형명'], row['제작일자'], row['제작사'], row['사용상태'],
                row['보관위치'], row['품명'], row['기준값'], row['상품군'], row['파트부'], row['모델명'], row['id']
            ))
        conn.commit()
        st.success("✅ 수정사항이 저장되었습니다.")
        st.rerun()

import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3

def mold_analysis():
    st.subheader("📊 금형 데이터 분석 (보관위치별 모델/파트 구성 + 요약)")

    conn = sqlite3.connect("estimate.db", check_same_thread=False)
    df = pd.read_sql_query("SELECT * FROM molds", conn)

    if df.empty:
        st.warning("등록된 금형 정보가 없습니다.")
        return

    # 컬럼 한글화
    df = df.rename(columns={
        'standard': '기준값',
        'category': '상품군',
        'part': '파트부',
        'model_name': '모델명',
        'location': '보관위치',
        'name': '금형명'
    })

    # ✅ 계단식 필터
    기준값s = df['기준값'].dropna().unique().tolist()
    선택_기준값 = st.multiselect("1️⃣ 기준값", 기준값s, default=기준값s)
    df = df[df['기준값'].isin(선택_기준값)]

    상품군s = df['상품군'].dropna().unique().tolist()
    선택_상품군 = st.multiselect("2️⃣ 상품군", 상품군s, default=상품군s)
    df = df[df['상품군'].isin(선택_상품군)]

    파트부s = df['파트부'].dropna().unique().tolist()
    선택_파트부 = st.multiselect("3️⃣ 파트부", 파트부s, default=파트부s)
    df = df[df['파트부'].isin(선택_파트부)]

    모델명s = df['모델명'].dropna().unique().tolist()
    선택_모델명 = st.multiselect("4️⃣ 모델명", 모델명s, default=모델명s)
    df = df[df['모델명'].isin(선택_모델명)]

    보관위치s = df['보관위치'].dropna().unique().tolist()
    선택_보관위치 = st.multiselect("5️⃣ 보관위치", 보관위치s, default=보관위치s)
    df = df[df['보관위치'].isin(선택_보관위치)]

    st.markdown("---")

    if df.empty:
        st.info("조건에 맞는 금형 데이터가 없습니다.")
        return

    # ✅ 도넛 차트 (보관위치별 상품군/파트부 비중)
    st.markdown("### 🍩 도넛 차트: 보관위치별 상품군/파트 구성")

    donut_cols = st.columns(min(4, len(선택_보관위치)))
    for i, loc in enumerate(선택_보관위치):
        loc_df = df[df['보관위치'] == loc]
        if loc_df.empty:
            with donut_cols[i]:
                st.info(f"{loc} 보관소: 데이터 없음")
            continue

        donut_data = loc_df.groupby(['상품군', '파트부']).size().reset_index(name='수량')
        donut_data['구성'] = donut_data['상품군'] + " / " + donut_data['파트부']

        fig = px.pie(donut_data, names='구성', values='수량', title=f"{loc} 구성 비율", hole=0.4)
        fig.update_traces(textinfo='percent+label')

        with donut_cols[i]:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ✅ 트리맵: 모델/파트 조합 + 금형명
    st.markdown("### 🌳 트리맵: 보관위치 → 모델명 → 파트부 구성")

    df['조합'] = df['모델명'].astype(str) + " / " + df['파트부'].astype(str)
    grouped = df.groupby(['보관위치', '모델명', '파트부']).agg(
        금형수량=('금형명', 'count'),
        금형목록=('금형명', lambda x: '<br>'.join(x))
    ).reset_index()

    fig = px.treemap(
        grouped,
        path=['보관위치', '모델명', '파트부'],
        values='금형수량',
        hover_data={'금형수량': True, '금형목록': True},
        color='보관위치'
    )
    fig.update_traces(root_color="lightgrey")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ✅ 요약 테이블 개선: 보관위치별 금형 수량 + 비율
    st.markdown("### 📋 요약: 보관위치별 금형 수량 및 상세 정보")

    location_summary = df.groupby('보관위치').size().reset_index(name='금형 수량')
    total = location_summary['금형 수량'].sum()
    location_summary['비율(%)'] = location_summary['금형 수량'] / total * 100
    location_summary['비율(%)'] = location_summary['비율(%)'].map("{:.1f}%".format)

    st.dataframe(location_summary, use_container_width=True)

    st.markdown("### 🔍 보관위치별 상세 현황 (모델명 + 파트부 기준)")
    for loc in 선택_보관위치:
        with st.expander(f"📦 {loc} 보관소 상세보기"):
            sub_df = df[df['보관위치'] == loc]
            detail = sub_df.groupby(['모델명', '파트부']).agg(
                금형수량=('금형명', 'count'),
                금형목록=('금형명', lambda x: ' / '.join(x))
            ).reset_index()
            st.dataframe(detail, use_container_width=True)
def mold_location_change():
    st.subheader("📦 금형 보관위치 변경")

    # 데이터 로드
    df = pd.read_sql_query("SELECT * FROM molds", conn)

    if df.empty:
        st.info("등록된 금형 정보가 없습니다.")
        return

    # 컬럼 한글화
    df_display = df.copy()
    df_display['선택'] = False
    df_display = df_display.rename(columns={
        'id': 'ID', 'code': '금형코드', 'name': '금형명',
        'location': '보관위치', 'model_name': '모델명',
        'part': '파트부', 'category': '상품군'
    })

    # ✅ 선택 박스 포함 테이블
    st.markdown("### ✅ 보관위치 변경 대상 선택")
    edited = st.data_editor(
        df_display[['선택', 'ID', '금형코드', '금형명', '모델명', '파트부', '상품군', '보관위치']],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic"
    )

    선택_ids = edited[edited['선택'] == True]['ID'].tolist()

    # ✅ 새 위치 선택
    if 선택_ids:
        new_location = st.selectbox("📍 변경할 보관위치 선택", sorted(df['location'].dropna().unique()))
        if st.button("🚚 선택 항목 위치 변경"):
            for i in 선택_ids:
                old_loc = df[df['id'] == i]['location'].values[0]
                cursor.execute("UPDATE molds SET location = ? WHERE id = ?", (new_location, i))
                cursor.execute("""
                    INSERT INTO mold_location_history (mold_id, 이전위치, 변경위치, 변경일시)
                    VALUES (?, ?, ?, ?)
                """, (
                    i, old_loc, new_location,
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                ))
            conn.commit()
            st.success(f"✅ 선택된 {len(선택_ids)}개 금형의 보관위치가 '{new_location}'(으)로 변경되었습니다.")
            st.rerun()
    else:
        st.info("🔎 먼저 금형을 선택해주세요.")

    # ✅ 이력 테이블 조회
    st.markdown("---")
    st.subheader("📜 보관위치 변경 이력")

    history = pd.read_sql_query("""
        SELECT h.*, m.code AS 금형코드, m.name AS 금형명
        FROM mold_location_history h
        LEFT JOIN molds m ON h.mold_id = m.id
        ORDER BY h.변경일시 DESC
    """, conn)

    if not history.empty:
        st.dataframe(history[["금형코드", "금형명", "이전위치", "변경위치", "변경일시"]], use_container_width=True)
    else:
        st.info("📭 아직 보관위치 변경 이력이 없습니다.")

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

def connect_to_google_sheets():
    import gspread
    from google.oauth2.service_account import Credentials

    creds_info = st.secrets["google_service_account"]

    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials = Credentials.from_service_account_info(
        creds_info, scopes=scopes
    )
    gc = gspread.authorize(credentials)
    return gc
    
    except Exception as e:
        st.error(f"❌ Google Sheet 연결 실패: {e}")
# -----------------------------------------------------------------------
if st.sidebar.button("🗂 Google Sheets 백업 실행"):
    backup_to_google_sheets()


def main():
    menu = st.sidebar.selectbox("📂 메뉴 선택", [
        "견적서 등록", "엑셀 업로드", "견적서 목록 보기", "견적서 비교 분석",
        "금형관리", "금형데이터 분석", "📦 보관위치 변경"
    ])

    if menu == "견적서 등록":
        add_estimate()
    elif menu == "엑셀 업로드":
        upload_excel()
    elif menu == "견적서 목록 보기":
        show_estimates()
    elif menu == "견적서 비교 분석":
        compare_estimates()
    elif menu == "금형관리":
        mold_management()
    elif menu == "금형데이터 분석":
        mold_analysis()
    elif menu == "📦 보관위치 변경":
        mold_location_change()

if __name__ == "__main__":
    main()

 
