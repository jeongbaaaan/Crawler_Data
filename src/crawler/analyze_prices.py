import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm
import os

# ====== 한글 폰트 설정 (폰트 파일 직접 로드) ======
# 'analyze_prices.py' 파일과 같은 폴더에 폰트를 복사해두면 편리합니다.
try:
    font_path = Path(__file__).parent / '나눔손글씨 암스테르담.ttf'

    if not font_path.exists():
        raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {font_path}")

    # 폰트 매니저에 폰트를 추가합니다.
    fm.fontManager.addfont(str(font_path))
    # 폰트의 이름으로 폰트 패밀리를 설정합니다.
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    
    print(f"폰트 설정 성공: {plt.rcParams['font.family']}")

except Exception as e:
    print(f"오류: 한글 폰트 설정 중 문제가 발생했습니다: {e}")
    print("기본 폰트로 그래프를 그립니다. 글자가 깨질 수 있습니다.")
    plt.rcParams['font.family'] = 'sans-serif' # 기본 폰트로 대체

plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

# ====== 경로 설정 ======
# 현재 파일 위치를 기준으로 프로젝트 루트 경로를 찾습니다.
BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "processed" / "gym_prices.csv"

# ====== 데이터 로드 ======
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"오류: {CSV_PATH} 파일을 찾을 수 없습니다. 크롤러를 먼저 실행해 CSV 파일을 생성하세요.")
    exit()

# '스포애니 선릉역점' 데이터만 필터링합니다.
spoany_df = df[df['name'] == '스포애니 선릉역점']

# 유효하지 않은 가격 데이터(예: 'FETCH_FAILED')를 제거합니다.
spoany_df = spoany_df[pd.to_numeric(spoany_df['price_krw'], errors='coerce').notna()]
spoany_df['price_krw'] = spoany_df['price_krw'].astype(int)

# 3. 'term_months'와 'price_text'를 조합하여 막대 그래프의 레이블을 만듭니다.
def create_label(row):
    if pd.isna(row['term_months']) or row['term_months'] == '':
        return f'{row["price_text"]}'
    else:
        return f'{int(row["term_months"]):.0f}개월'

spoany_df['plan_label'] = spoany_df.apply(create_label, axis=1)

# 데이터프레임을 가격 순으로 재정렬합니다.
spoany_df = spoany_df.sort_values(by='price_krw', ascending=True)

if spoany_df.empty:
    print("오류: '스포애니 선릉역점'의 유효한 가격 데이터가 없습니다. CSV 파일을 확인하세요.")
    exit()

# ====== 그래프 그리기 ======
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(spoany_df['plan_label'], spoany_df['price_krw'], color='skyblue')

# 제목 및 레이블
ax.set_title('스포애니 선릉역점 회원권 가격 비교', fontsize=16, pad=15)
ax.set_xlabel('회원권 종류', fontsize=12)
ax.set_ylabel('가격 (KRW)', fontsize=12)
ax.ticklabel_format(style='plain', axis='y')

plt.xticks(rotation=0, ha='center')

# 막대 위에 값 표시
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.0f}원', ha='center', va='bottom', fontsize=10, color='dimgray')

plt.tight_layout()
output_path = Path(__file__).parent / 'spoany_prices_bar_chart.png'
plt.savefig(output_path)
print(f"데이터 시각화 완료: {output_path}")