"""
Multi-Themed Bookend Dashboard
프로페셔널한 색상 + 다크모드 지원
"""

import streamlit as st
from datetime import datetime, timedelta
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import components
try:
    from dashboard.utils.data_loader_v1 import get_data_loader
    from dashboard.utils.theme_config import get_theme, get_all_pages, get_theme_css
    from dashboard.components.charts_themed import (
        create_themed_hourly_chart,
        create_themed_daily_chart,
        create_themed_distribution_pie,
        create_themed_radar_chart,
        create_themed_gauge_chart,
        create_themed_bar_chart,
        create_themed_timeline_chart
    )
except ImportError as e:
    st.error(f"❌ 임포트 오류: {e}")
    st.stop()

# ========================================
# 🌿 페이지 설정
# ========================================

st.set_page_config(
    page_title="Bookend Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# 📋 사이드바
# ========================================

with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size: 42px;">📚</div>
            <h2>Bookend</h2>
            <p>데이터 분석 대시보드</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📍 페이지")
    
    page = st.radio(
        "페이지 선택",
        get_all_pages(),
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # 다크모드 토글
    st.markdown("### 🌙 테마")
    dark_mode = st.checkbox("다크모드", value=True)
    
    st.divider()
    
    st.markdown("### ⏱️ 기간")
    time_range = st.selectbox(
        "기간",
        ["최근 7일", "최근 30일", "최근 90일", "전체"],
        index=3,
        label_visibility="collapsed"
    )
    
    time_range_map = {
        "최근 7일": 7,
        "최근 30일": 30,
        "최근 90일": 90,
        "전체": 365 * 10
    }
    days = time_range_map[time_range]
    
    st.divider()
    
    st.markdown("### 🔍 필터")
    
    # 사용자 선택
    try:
        data_loader_temp = get_data_loader()
        users = data_loader_temp.get_available_users() if data_loader_temp else []
        
        selected_user_id = None
        if users and len(users) > 0:
            user_options = {"전체 사용자": None}
            user_options.update({f"👤 {distinct_id[:10]}...": uid for uid, distinct_id in users})
            
            selected = st.selectbox(
                "사용자 선택",
                options=list(user_options.keys()),
                index=0,
                label_visibility="collapsed"
            )
            selected_user_id = user_options[selected]
        else:
            st.info("💡 사용자 데이터 로딩 중...")
    except Exception as e:
        st.warning(f"⚠️ 사용자 로딩 오류: {str(e)[:50]}...")
        selected_user_id = None
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄", use_container_width=True, help="새로고침"):
            st.cache_resource.clear()
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("자동", value=False, help="5분마다 자동 새로고침")

# ========================================
# 🎨 테마 적용
# ========================================

theme = get_theme(page)
st.markdown(get_theme_css(theme, dark_mode=dark_mode), unsafe_allow_html=True)

# ========================================
# 🌱 데이터 로더 초기화
# ========================================

@st.cache_resource(ttl=300)
def init_data_loader():
    """데이터 로더를 캐싱과 함께 초기화합니다."""
    try:
        loader = get_data_loader()
        if not loader.health_check():
            st.error("❌ 데이터베이스 연결 실패")
            return None
        return loader
    except Exception as e:
        st.error(f"❌ 데이터 로더 초기화 실패: {e}")
        return None

data_loader = init_data_loader()

if data_loader is None:
    st.stop()

# ========================================
# 📊 페이지 1: 전체 현황
# ========================================

if page == "📊 전체 현황":
    st.markdown("# 📊 전체 현황")
    st.markdown("""
        <p style="font-size: 15px; color: """ + theme['secondary_color'] + """; margin-bottom: 2rem;">
        전체 사용 통계와 주요 지표를 확인하세요
        </p>
    """, unsafe_allow_html=True)
    
    kpis = data_loader.get_kpi_summary(days=days)
    
    if not kpis:
        st.warning("⚠️ 선택한 기간에 데이터가 없습니다")
    else:
        st.markdown("""
            <div class="section-header">
                <h3>📊 핵심 지표</h3>
                <p>최근 활동의 주요 수치를 확인하세요</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_interactions = kpis.get('total_interactions', 0)
        total_users = kpis.get('total_users', 0)
        active_users = kpis.get('active_users', 0)
        unique_events = kpis.get('unique_events', 0)
        
        with col1:
            st.metric("총 활동 수", f"{total_interactions:,}")
        with col2:
            st.metric("전체 사용자", f"{total_users:,}")
        with col3:
            activity_rate = (active_users / total_users * 100) if total_users > 0 else 0
            st.metric("활성 사용자", f"{active_users:,}", delta=f"{activity_rate:.1f}%")
        with col4:
            st.metric("기능 수", f"{unique_events:,}")
        
        # 시간대별 활동
        st.markdown("""
            <div class="section-header">
                <h3>🌞 시간대별 활동</h3>
                <p>하루 중 언제 가장 활발한가요?</p>
            </div>
        """, unsafe_allow_html=True)
        
        time_dist = data_loader.get_time_of_day_distribution(days=days)
        
        if time_dist:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig = create_themed_distribution_pie(time_dist, theme, "시간대 활동 분포", dark_mode=dark_mode)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown("#### 📈 시간대 통계")
                st.markdown("")
                
                total = sum(time_dist.values())
                for period, count in sorted(time_dist.items(), key=lambda x: -x[1])[:3]:
                    percentage = (count / total * 100) if total > 0 else 0
                    st.metric(period, f"{percentage:.1f}%", delta=f"{count}회")
        
        # 상위 사용자
        st.markdown("""
            <div class="section-header">
                <h3>👥 활발한 사용자</h3>
                <p>가장 많이 활동하는 사용자 TOP 10</p>
            </div>
        """, unsafe_allow_html=True)
        
        top_users = data_loader.get_top_users(limit=10, days=days)
        
        if not top_users.empty:
            st.dataframe(
                top_users[['distinct_id', 'interaction_count', 'last_interaction']],
                width='stretch',
                hide_index=True
            )

# ========================================
# ⏰ 페이지 2: 시간대 분석
# ========================================

elif page == "⏰ 시간대 분석":
    st.markdown("# ⏰ 시간대 분석")
    st.markdown("""
        <p style="font-size: 15px; color: """ + theme['secondary_color'] + """; margin-bottom: 2rem;">
        시간에 따른 사용자 활동 패턴을 분석합니다
        </p>
    """, unsafe_allow_html=True)
    
    hourly_df = data_loader.get_hourly_activity(days=days, user_id=selected_user_id)
    
    if not hourly_df.empty:
        st.markdown("""
            <div class="section-header">
                <h3>🕐 24시간 활동 리듬</h3>
                <p>시간대별 활동 분포를 확인하세요</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = create_themed_hourly_chart(hourly_df, theme, "시간대별 활동", dark_mode=dark_mode)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            peak_hour = hourly_df.loc[hourly_df['count'].idxmax()]
            
            st.markdown("#### 🌟 피크 타임")
            st.markdown("")
            
            st.metric("최고 활동 시간", f"{int(peak_hour['hour']):02d}:00")
            st.metric("평균 응답", f"{peak_hour['avg_response_time_ms']:.0f}ms")
            st.metric("활성 사용자", f"{int(peak_hour['unique_users'])}명")
    else:
        st.info("⚠️ 선택한 기간에 시간대별 데이터가 없습니다")
    
    # 일별 추세
    daily_df = data_loader.get_daily_rhythm(days=min(days, 90))
    
    if not daily_df.empty:
        st.markdown("""
            <div class="section-header">
                <h3>📅 일별 활동 추이</h3>
                <p>날짜별 사용 패턴을 추적합니다</p>
            </div>
        """, unsafe_allow_html=True)
        
        daily_summary = daily_df.groupby('date').agg({'count': 'sum', 'unique_users': 'max'}).reset_index()
        daily_summary.rename(columns={'count': 'daily_count'}, inplace=True)
        
        fig = create_themed_daily_chart(daily_summary, theme, "일별 활동", dark_mode=dark_mode)
        st.plotly_chart(fig, width='stretch')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_daily = daily_summary['daily_count'].mean()
            st.metric("일평균", f"{avg_daily:.1f}회")
        
        with col2:
            total_days = len(daily_summary)
            st.metric("활동 일수", f"{total_days}일")
        
        with col3:
            max_daily = daily_summary['daily_count'].max()
            st.metric("최고 기록", f"{int(max_daily)}회")
    else:
        st.info("⚠️ 선택한 기간에 일별 데이터가 없습니다")

# ========================================
# 📈 페이지 3: 활동 패턴
# ========================================

elif page == "📈 활동 패턴":
    st.markdown("# 📈 활동 패턴")
    st.markdown("""
        <p style="font-size: 15px; color: """ + theme['secondary_color'] + """; margin-bottom: 2rem;">
        반복되는 행동과 다양성을 분석합니다
        </p>
    """, unsafe_allow_html=True)
    
    if not selected_user_id:
        st.info("👈 좌측 사이드바에서 사용자를 선택하면 개인 패턴을 볼 수 있습니다")
        
        # 전체 활동 통계
        st.markdown("""
            <div class="section-header">
                <h3>📊 전체 활동 통계</h3>
                <p>모든 사용자의 활동 패턴을 확인하세요</p>
            </div>
        """, unsafe_allow_html=True)
        
        event_dist = data_loader.get_event_distribution(days=days)
        if event_dist:
            fig = create_themed_distribution_pie(event_dist, theme, "전체 기능 사용 분포", dark_mode=dark_mode)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # 활동 다양성 게이지
        diversity_score = data_loader.get_diversity_score(
            user_id=selected_user_id,
            days=days
        )
        
        st.markdown("""
            <div class="section-header">
                <h3>🌈 활동 다양성</h3>
                <p>다양한 기능을 얼마나 고르게 사용하는지 분석합니다</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 중앙에 큰 게이지 차트
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig_gauge = create_themed_gauge_chart(diversity_score, theme, "다양성 점수", dark_mode=dark_mode)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 전체 기능 사용 분포
        st.markdown("""
            <div class="section-header">
                <h3>📊 전체 기능 사용 분포</h3>
                <p>어떤 기능을 주로 사용하시나요?</p>
            </div>
        """, unsafe_allow_html=True)
        
        event_dist = data_loader.get_event_distribution(
            days=days,
            user_id=selected_user_id
        )
        
        if event_dist:
            fig = create_themed_distribution_pie(event_dist, theme, "기능 사용 분포", dark_mode=dark_mode)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ 선택한 사용자의 활동 데이터가 없습니다")

# ========================================
# 💬 페이지 4: 톤 분석
# ========================================

elif page == "💬 톤 분석":
    st.markdown("# 💬 톤 분석")
    st.markdown("""
        <p style="font-size: 15px; color: """ + theme['secondary_color'] + """; margin-bottom: 2rem;">
        콘텐츠에 담긴 감정과 톤을 분석합니다
        </p>
    """, unsafe_allow_html=True)
    
    tone_dist = data_loader.get_tone_distribution(days=days, user_id=selected_user_id)
    
    if tone_dist:
        st.markdown("""
            <div class="section-header">
                <h3>💬 톤 분포</h3>
                <p>어떤 톤을 선호하시나요?</p>
            </div>
        """, unsafe_allow_html=True)
        
        fig = create_themed_radar_chart(tone_dist, theme, "톤 선호도", dark_mode=dark_mode)
        st.plotly_chart(fig, use_container_width=True)
        
        # 톤별 비율 카드
        total_tone = sum(tone_dist.values())
        sorted_tones = sorted(tone_dist.items(), key=lambda x: -x[1])
        
        cols = st.columns(min(len(sorted_tones), 3))
        
        for idx, (tone_name, count) in enumerate(sorted_tones[:3]):
            percentage = (count / total_tone * 100) if total_tone > 0 else 0
            with cols[idx]:
                tone_icons = {
                    'formal': '📝',
                    'casual': '😊',
                    'poetic': '🎭',
                    'technical': '⚙️',
                    'emotional': '❤️'
                }
                icon = tone_icons.get(tone_name.lower(), '💬')
                
                card_bg = theme.get('dark_card', '#334155') if dark_mode else '#ffffff'
                text_color = theme.get('dark_text', '#e2e8f0') if dark_mode else theme['primary_color']
                
                st.markdown(f"""
                    <div class="custom-card" style="background: {card_bg}; text-align: center;">
                        <div style="font-size: 40px; margin-bottom: 0.5rem;">{icon}</div>
                        <h4 style="color: {text_color}; margin: 0.5rem 0; font-size: 16px;">{tone_name.title()}</h4>
                        <div style="font-size: 28px; font-weight: 700; color: {theme['primary_color']};">{percentage:.0f}%</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("⚠️ 선택한 기간에 톤 데이터가 없습니다")

# ========================================
# 👥 페이지 5: 사용자 분석
# ========================================

elif page == "👥 사용자 분석":
    st.markdown("# 👥 사용자 분석")
    st.markdown("""
        <p style="font-size: 15px; color: """ + theme['secondary_color'] + """; margin-bottom: 2rem;">
        사용자 행동 패턴과 세그먼트 분석
        </p>
    """, unsafe_allow_html=True)
    
    # 사용자 세그먼트 분석
    st.markdown("""
        <div class="section-header">
            <h3>👥 사용자 세그먼트</h3>
            <p>전체 사용자 활동 횟수에 따른 4단계 분류</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 전체 사용자 데이터 가져오기
    all_users = data_loader.get_all_users_with_activity()
    
    if not all_users.empty:
        # 사용자를 활동 횟수로 분류
        segment_counts = {
            '신규 사용자 (0-4회)': 0,
            '탐색 사용자 (5-19회)': 0,
            '성장 사용자 (20-49회)': 0,
            '파워 유저 (50+회)': 0
        }
        
        for _, row in all_users.iterrows():
            count = row['interaction_count']
            if count <= 4:
                segment_counts['신규 사용자 (0-4회)'] += 1
            elif count <= 19:
                segment_counts['탐색 사용자 (5-19회)'] += 1
            elif count <= 49:
                segment_counts['성장 사용자 (20-49회)'] += 1
            else:
                segment_counts['파워 유저 (50+회)'] += 1
        
        total_users_count = len(all_users)
        
        # 상단에 전체 사용자 수 크게 표시
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            card_bg = theme.get('dark_card', '#334155') if dark_mode else '#ffffff'
            text_color = theme.get('dark_text', '#e2e8f0') if dark_mode else theme['primary_color']
            
            st.markdown(f"""
                <div class="custom-card" style="text-align: center; padding: 2rem; background: {card_bg};">
                    <div style="font-size: 48px; margin-bottom: 1rem;">👥</div>
                    <h2 style="color: {text_color}; margin: 0;">전체 사용자</h2>
                    <div style="font-size: 64px; font-weight: 700; color: {theme['primary_color']}; margin: 1rem 0;">
                        {total_users_count:,}명
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 세그먼트 분포 차트만 표시
        fig = create_themed_distribution_pie(segment_counts, theme, "사용자 세그먼트 분포", dark_mode=dark_mode)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("⚠️ 사용자 데이터가 없습니다")

# 자동 새로고침
if auto_refresh:
    import time
    time.sleep(300)
    st.rerun()