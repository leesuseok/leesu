import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="견적서 관리 시스템", layout="wide")

# DB 초기화
@st.cache_resource
def init_db():
    conn = sqlite3.connect("estimate.db", check_same_thread=False)
    cursor = conn.cursor()
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
    conn.commit()
    return conn

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
                         '보관위치', '비고', '기준값', '상품군', '파트부', '모델명']
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
                    row['제작사'], row['사용상태'], row['보관위치'], row['비고'],
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

        note = cols[0].text_input("비고")
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
        'note': '비고', 'standard': '기준값', 'category': '상품군',
        'part': '파트부', 'model_name': '모델명'
    })

    edited = st.data_editor(df_edit_display[['선택', 'id', '금형코드', '금형명', '제작일자', '제작사',
                                             '사용상태', '보관위치', '비고', '기준값', '상품군', '파트부', '모델명']],
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
                row['보관위치'], row['비고'], row['기준값'], row['상품군'], row['파트부'], row['모델명'], row['id']
            ))
        conn.commit()
        st.success("✅ 수정사항이 저장되었습니다.")
        st.rerun()

import streamlit as st
import pandas as pd
import plotly.express as px

def mold_analysis():
    st.subheader("📊 금형 데이터 분석 (보관위치 + 조합 기준 시각화)")

    df = pd.read_sql_query("SELECT * FROM molds", conn)
    if df.empty:
        st.warning("등록된 금형 정보가 없습니다.")
        return

    # 컬럼 정리 (영문 → 한글)
    df = df.rename(columns={
        'standard': '기준값',
        'category': '상품군',
        'part': '파트부',
        'model_name': '모델명',
        'location': '보관위치'
    })

    # 🔍 계단식 필터
    st.markdown("### 🎯 조건 선택 (계단식 필터)")

    df_filtered = df.copy()

    기준값s = df_filtered['기준값'].dropna().unique().tolist()
    선택_기준 = st.multiselect("1️⃣ 기준값", 기준값s, default=기준값s)
    df_filtered = df_filtered[df_filtered['기준값'].isin(선택_기준)]

    상품군s = df_filtered['상품군'].dropna().unique().tolist()
    선택_상품군 = st.multiselect("2️⃣ 상품군", 상품군s, default=상품군s)
    df_filtered = df_filtered[df_filtered['상품군'].isin(선택_상품군)]

    파트부s = df_filtered['파트부'].dropna().unique().tolist()
    선택_파트부 = st.multiselect("3️⃣ 파트부", 파트부s, default=파트부s)
    df_filtered = df_filtered[df_filtered['파트부'].isin(선택_파트부)]

    모델명s = df_filtered['모델명'].dropna().unique().tolist()
    선택_모델명 = st.multiselect("4️⃣ 모델명", 모델명s, default=모델명s)
    df_filtered = df_filtered[df_filtered['모델명'].isin(선택_모델명)]

    보관위치s = df_filtered['보관위치'].dropna().unique().tolist()
    선택_보관위치 = st.multiselect("5️⃣ 보관위치", 보관위치s, default=보관위치s)
    df_filtered = df_filtered[df_filtered['보관위치'].isin(선택_보관위치)]

    st.markdown("---")

    # 🍩 도넛 차트: 보관위치별 상품군 + 파트부 조합
    st.markdown("## 🍩 보관위치별 도넛 차트 (상품군 + 파트부 기준)")

    if 선택_보관위치:
        donut_cols = st.columns(min(4, len(선택_보관위치)))
        for idx, loc in enumerate(선택_보관위치):
            loc_df = df_filtered[df_filtered['보관위치'] == loc].copy()
            if loc_df.empty:
                with donut_cols[idx]:
                    st.info(f"{loc} 보관소에 금형이 없습니다.")
                continue

            loc_df['조합'] = loc_df['상품군'].astype(str) + " - " + loc_df['파트부'].astype(str)
            combo_counts = loc_df['조합'].value_counts().reset_index()
            combo_counts.columns = ['조합', '수량']

            fig = px.pie(
                combo_counts,
                names='조합',
                values='수량',
                title=f"{loc} 보관소",
                hole=0.5
            )
            fig.update_traces(textinfo='label+percent')

            with donut_cols[idx]:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("※ 보관위치를 선택해주세요.")

    # 📊 바 차트: 모델명 + 파트부 조합 수량
    st.markdown("## 📊 모델 + 파트부 조합별 금형 수량")

    if not df_filtered.empty:
        df_filtered['조합'] = df_filtered['모델명'].astype(str) + " - " + df_filtered['파트부'].astype(str)
        model_counts = df_filtered['조합'].value_counts().reset_index()
        model_counts.columns = ['모델+파트부', '수량']

        fig_bar = px.bar(
            model_counts,
            x='모델+파트부',
            y='수량',
            text='수량',
            title="모델 + 파트부별 금형 수량",
            labels={'모델+파트부': '모델+파트부', '수량': '금형 수량'},
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("해당 조건에 맞는 금형 데이터가 없습니다.")

    # 📋 조건별 상세 테이블
    st.markdown("### 📋 조건별 금형 상세 보기")
    st.dataframe(df_filtered.drop(columns=['id']), use_container_width=True)

    st.markdown("---")
    st.info("📢 업무 자동화 플랫폼 👉 [gptonline.ai/ko](https://gptonline.ai/ko/)에서 더 많은 기능 확인하세요.")


def main():
    menu = st.sidebar.selectbox("📂 메뉴 선택", [
        "견적서 등록", "엑셀 업로드", "견적서 목록 보기", "견적서 비교 분석", "금형관리", "금형데이터 분석"
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


if __name__ == "__main__":
    main()

 