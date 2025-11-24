"""
Theme configuration for multi-themed dashboard.
프로페셔널하고 눈에 편한 색상 팔레트 + 다크모드 완벽 지원
"""

from typing import Dict, Any

THEMES = {
    "📊 전체 현황": {
        "name": "Overview",
        "primary_color": "#3b82f6",
        "secondary_color": "#60a5fa",
        "accent_color": "#93c5fd",
        "light_bg": "#dbeafe",
        "sidebar_bg": "#f8fafc",
        "sidebar_border": "#3b82f6",
        "button_color": "#2563eb",
        "button_hover": "#1d4ed8",
        "page_gradient": "linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)",
        "section_bg": "rgba(59,130,246,0.05)",
        "accent_gradient": "linear-gradient(135deg, rgba(37,99,235,0.05) 0%, rgba(59,130,246,0.05) 100%)",
        "dark_bg": "#1a2332",
        "dark_surface": "#253447",
        "dark_card": "#3f4d5e",
        "dark_text": "#f1f5f9",
        "dark_secondary": "#cbd5e1",
        "dark_border": "#5a6b7f",
    },
    
    "⏰ 시간대 분석": {
        "name": "Time Analysis",
        "primary_color": "#06b6d4",
        "secondary_color": "#22d3ee",
        "accent_color": "#67e8f9",
        "light_bg": "#cffafe",
        "sidebar_bg": "#f0fdfa",
        "sidebar_border": "#06b6d4",
        "button_color": "#0891b2",
        "button_hover": "#0e7490",
        "page_gradient": "linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%)",
        "section_bg": "rgba(6,182,212,0.05)",
        "accent_gradient": "linear-gradient(135deg, rgba(8,145,178,0.05) 0%, rgba(6,182,212,0.05) 100%)",
        "dark_bg": "#1a2332",
        "dark_surface": "#253447",
        "dark_card": "#3f4d5e",
        "dark_text": "#f1f5f9",
        "dark_secondary": "#cbd5e1",
        "dark_border": "#5a6b7f",
    },
    
    "📈 활동 패턴": {
        "name": "Activity Pattern",
        "primary_color": "#10b981",
        "secondary_color": "#34d399",
        "accent_color": "#6ee7b7",
        "light_bg": "#d1fae5",
        "sidebar_bg": "#f0fdf4",
        "sidebar_border": "#10b981",
        "button_color": "#059669",
        "button_hover": "#047857",
        "page_gradient": "linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%)",
        "section_bg": "rgba(16,185,129,0.05)",
        "accent_gradient": "linear-gradient(135deg, rgba(5,150,105,0.05) 0%, rgba(16,185,129,0.05) 100%)",
        "dark_bg": "#1a2332",
        "dark_surface": "#253447",
        "dark_card": "#3f4d5e",
        "dark_text": "#f1f5f9",
        "dark_secondary": "#cbd5e1",
        "dark_border": "#5a6b7f",
    },
    
    "💬 톤 분석": {
        "name": "Sentiment Analysis",
        "primary_color": "#a78bfa",
        "secondary_color": "#c4b5fd",
        "accent_color": "#ddd6fe",
        "light_bg": "#ede9fe",
        "sidebar_bg": "#faf5ff",
        "sidebar_border": "#8b5cf6",
        "button_color": "#7c3aed",
        "button_hover": "#6d28d9",
        "page_gradient": "linear-gradient(135deg, #faf5ff 0%, #ede9fe 100%)",
        "section_bg": "rgba(167,139,250,0.05)",
        "accent_gradient": "linear-gradient(135deg, rgba(124,58,237,0.05) 0%, rgba(139,92,246,0.05) 100%)",
        "dark_bg": "#1a2332",
        "dark_surface": "#253447",
        "dark_card": "#3f4d5e",
        "dark_text": "#f1f5f9",
        "dark_secondary": "#cbd5e1",
        "dark_border": "#5a6b7f",
    },
    
    "👥 사용자 분석": {
        "name": "User Analysis",
        "primary_color": "#f87171",
        "secondary_color": "#fca5a5",
        "accent_color": "#fecaca",
        "light_bg": "#fee2e2",
        "sidebar_bg": "#fef2f2",
        "sidebar_border": "#ef4444",
        "button_color": "#dc2626",
        "button_hover": "#b91c1c",
        "page_gradient": "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)",
        "section_bg": "rgba(248,113,113,0.05)",
        "accent_gradient": "linear-gradient(135deg, rgba(220,38,38,0.05) 0%, rgba(239,68,68,0.05) 100%)",
        "dark_bg": "#1a2332",
        "dark_surface": "#253447",
        "dark_card": "#3f4d5e",
        "dark_text": "#f1f5f9",
        "dark_secondary": "#cbd5e1",
        "dark_border": "#5a6b7f",
    },
}


def get_theme(page_name: str) -> Dict[str, Any]:
    """페이지명으로 테마 가져오기"""
    return THEMES.get(page_name, THEMES["📊 전체 현황"])


def get_all_pages() -> list:
    """모든 페이지 목록"""
    return list(THEMES.keys())


def get_theme_css(theme: Dict[str, Any], dark_mode: bool = False) -> str:
    """테마별 CSS 생성 - 라이트/다크 모드 완벽 지원"""
    
    if dark_mode:
        bg_color = theme['dark_bg']
        card_bg = theme['dark_card']
        text_color = theme['dark_text']
        secondary_text = theme['dark_secondary']
        border_color = theme['dark_border']
    else:
        bg_color = "#f8fafc"
        card_bg = "#ffffff"
        text_color = theme['primary_color']
        secondary_text = theme['secondary_color']
        border_color = "#e2e8f0"
    
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* 메인 콘텐츠 배경 */
    .stApp {{
        background: {bg_color};
    }}
    
    .main {{
        background: {bg_color};
    }}
    
    .block-container {{
        background: {bg_color};
    }}
    
    /* 사이드바 */
    [data-testid="stSidebar"] {{
        background: {theme['sidebar_bg']};
        border-right: 2px solid {border_color};
    }}
    
    /* 사이드바의 모든 텍스트 - 검은색 고정 */
    [data-testid="stSidebar"] * {{
        color: #000000 !important;
    }}
    
    [data-testid="stSidebar"] p {{
        color: #000000 !important;
    }}
    
    [data-testid="stSidebar"] label {{
        color: #000000 !important;
    }}
    
    [data-testid="stSidebar"] div {{
        color: #000000 !important;
    }}
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #000000 !important;
    }}
    
    .sidebar-logo {{
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid {border_color};
        margin-bottom: 1.5rem;
    }}
    
    .sidebar-logo h2 {{
        font-size: 24px;
        font-weight: 700;
        color: {theme['primary_color']};
        margin: 0.5rem 0 0 0;
    }}
    
    .sidebar-logo p {{
        font-size: 12px;
        color: #000000;
        margin: 0.25rem 0 0 0;
    }}
    
    /* 라디오 버튼 */
    [data-testid="stRadio"] label {{
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        cursor: pointer;
        font-size: 14px;
        font-weight: 600;
        color: #1f2937;
    }}
    
    [data-testid="stRadio"] label:hover {{
        background: #f3f4f6;
        border-color: {theme['primary_color']};
    }}
    
    /* 선택된 라디오 버튼 */
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] + label {{
        background: {theme['primary_color']};
        border-color: {theme['primary_color']};
        color: #ffffff;
        font-weight: 700;
    }}
    
    /* 제목 */
    h1, h2, h3 {{
        color: {text_color};
    }}
    
    /* 메트릭 카드 */
    [data-testid="metric-container"] {{
        background: {card_bg};
        border: 2px solid {border_color};
        border-radius: 16px;
        padding: 1.5rem;
    }}
    
    /* 버튼 */
    .stButton > button {{
        background: {theme['button_color']};
        color: white;
        border: none;
        border-radius: 10px;
    }}
    
    /* 차트 컨테이너 */
    div[data-testid="stPlotlyChart"] {{
        background: {card_bg};
        border: 2px solid {border_color};
        border-radius: 16px;
        padding: 1rem;
    }}
    
    /* 데이터프레임 */
    [data-testid="stDataFrame"] {{
        background: {card_bg};
        border: 2px solid {border_color};
        border-radius: 12px;
    }}
    
    /* 텍스트 */
    .stMarkdown {{
        color: {text_color};
    }}
    
    .stMarkdown p {{
        color: {text_color};
    }}
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {text_color};
    }}
    
    label {{
        color: {text_color};
    }}
    
    /* 체크박스 */
    [data-testid="stCheckbox"] {{
        color: {text_color};
    }}
    
    [data-testid="stCheckbox"] label {{
        color: {text_color};
    }}
    
    /* 셀렉트박스 */
    [data-testid="stSelectbox"] {{
        color: {text_color};
    }}
    
    [data-testid="stSelectbox"] label {{
        color: {text_color};
    }}
    
    /* 메트릭 값 */
    [data-testid="stMetricValue"] {{
        color: {theme['primary_color']};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {text_color};
    }}
    
    [data-testid="stMetricDelta"] {{
        color: {theme['accent_color']};
    }}
    
    </style>
    """