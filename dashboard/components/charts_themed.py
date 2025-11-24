"""
Themed chart components for multi-themed dashboard.
다크모드 완벽 지원 + 차트 높이 보장
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any
import numpy as np


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """16진 색상을 RGBA 형식으로 변환"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def get_chart_colors(theme: Dict[str, Any], dark_mode: bool = False):
    """차트 배경색 설정"""
    if dark_mode:
        return {
            'paper_bg': theme.get('dark_card', '#3f4d5e'),
            'plot_bg': theme.get('dark_surface', '#253447'),
            'grid_color': theme.get('dark_border', '#5a6b7f'),
            'text_color': theme.get('dark_text', '#f1f5f9'),
        }
    else:
        return {
            'paper_bg': '#ffffff',
            'plot_bg': '#ffffff',
            'grid_color': hex_to_rgba(theme['secondary_color'], 0.15),
            'text_color': theme['primary_color'],
        }


def create_themed_hourly_chart(
    df: pd.DataFrame, 
    theme: Dict[str, Any], 
    title: str = "시간대별 리듬",
    dark_mode: bool = False
) -> go.Figure:
    """테마를 적용한 시간대별 차트"""
    colors = get_chart_colors(theme, dark_mode)
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(
            text="📊 데이터가 없습니다",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['text_color'])
        )
        fig.update_layout(
            height=450,
            paper_bgcolor=colors['paper_bg'],
            plot_bgcolor=colors['plot_bg']
        )
        return fig
    
    hours = list(range(24))
    counts = [0] * 24
    
    for _, row in df.iterrows():
        hour = int(row['hour'])
        count = row['count']
        counts[hour] = count
    
    hour_labels = [f"{h:02d}:00" for h in hours]
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=counts,
        mode='lines+markers',
        line=dict(
            color=theme['primary_color'],
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            color=counts,
            colorscale=[
                [0, theme['light_bg']],
                [0.5, theme['secondary_color']],
                [1, theme['primary_color']]
            ],
            showscale=False,
            line=dict(color=colors['paper_bg'], width=2)
        ),
        fill='tozeroy',
        fillcolor=hex_to_rgba(theme['primary_color'], 0.2),
        hovertemplate='<b>%{text}</b><br>활동: %{y}<extra></extra>',
        text=hour_labels
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=colors['text_color'])
        ),
        xaxis_title="시간 (Hour)",
        yaxis_title="활동 수",
        plot_bgcolor=colors['plot_bg'],
        paper_bgcolor=colors['paper_bg'],
        height=450,
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(
            tickmode='linear', tick0=0, dtick=2,
            range=[-0.5, 23.5],
            showgrid=True,
            gridcolor=colors['grid_color'],
            color=colors['text_color']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid_color'],
            color=colors['text_color']
        ),
        hovermode='x unified',
        font={'family': 'Pretendard, sans-serif', 'color': colors['text_color']}
    )
    
    return fig


def create_themed_daily_chart(
    df: pd.DataFrame, 
    theme: Dict[str, Any], 
    title: str = "일별 활동",
    dark_mode: bool = False
) -> go.Figure:
    """테마를 적용한 일별 차트"""
    colors = get_chart_colors(theme, dark_mode)
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(
            text="📊 데이터가 없습니다",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['text_color'])
        )
        fig.update_layout(
            height=350,
            paper_bgcolor=colors['paper_bg'],
            plot_bgcolor=colors['plot_bg']
        )
        return fig
    
    count_col = 'daily_count' if 'daily_count' in df.columns else 'count'
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[count_col],
        mode='lines',
        line=dict(
            color=theme['primary_color'],
            width=2.5,
            shape='spline'
        ),
        fill='tozeroy',
        fillcolor=hex_to_rgba(theme['primary_color'], 0.15),
        name='활동량',
        hovertemplate='<b>%{x}</b><br>활동: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=colors['text_color'])
        ),
        xaxis_title="날짜",
        yaxis_title="활동 수",
        hovermode='x unified',
        plot_bgcolor=colors['plot_bg'],
        paper_bgcolor=colors['paper_bg'],
        margin=dict(l=60, r=20, t=60, b=60),
        height=350,
        xaxis=dict(
            showgrid=True,
            gridcolor=colors['grid_color'],
            showline=True,
            linecolor=colors['grid_color'],
            color=colors['text_color']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid_color'],
            showline=True,
            linecolor=colors['grid_color'],
            color=colors['text_color']
        ),
        font={'family': 'Pretendard, sans-serif', 'color': colors['text_color']}
    )
    
    return fig


def create_themed_distribution_pie(
    data: Dict[str, int], 
    theme: Dict[str, Any], 
    title: str = "분포",
    dark_mode: bool = False
) -> go.Figure:
    """테마를 적용한 분포 파이 차트"""
    colors = get_chart_colors(theme, dark_mode)
    fig = go.Figure()
    
    if not data:
        fig.add_annotation(
            text="📊 데이터가 없습니다",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['text_color'])
        )
        fig.update_layout(
            height=400,
            paper_bgcolor=colors['paper_bg'],
            plot_bgcolor=colors['plot_bg']
        )
        return fig
    
    labels = list(data.keys())
    values = list(data.values())
    
    # 테마 색상 팔레트 생성
    pie_colors = [
        theme['primary_color'],
        theme['secondary_color'],
        theme['accent_color']
    ]
    pie_colors += [hex_to_rgba(theme['primary_color'], 0.7) for _ in range(max(0, len(labels) - 3))]
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(
            colors=pie_colors[:len(labels)],
            line=dict(color=colors['paper_bg'], width=3)
        ),
        textposition='auto',
        textinfo='label+percent',
        textfont=dict(color=colors['text_color'], size=13),
        hovertemplate='<b>%{label}</b><br>개수: %{value}<br>비율: %{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=colors['text_color'])
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(color=colors['text_color'])
        ),
        height=400,
        margin=dict(l=20, r=120, t=60, b=20),
        paper_bgcolor=colors['paper_bg'],
        plot_bgcolor=colors['plot_bg'],
        font={'family': 'Pretendard, sans-serif', 'color': colors['text_color']}
    )
    
    return fig


def create_themed_radar_chart(
    data: Dict[str, int], 
    theme: Dict[str, Any], 
    title: str = "분포",
    dark_mode: bool = False
) -> go.Figure:
    """테마를 적용한 레이더 차트"""
    colors = get_chart_colors(theme, dark_mode)
    fig = go.Figure()
    
    if not data:
        fig.add_annotation(
            text="📊 데이터가 없습니다",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['text_color'])
        )
        fig.update_layout(
            height=450,
            paper_bgcolor=colors['paper_bg'],
            plot_bgcolor=colors['plot_bg']
        )
        return fig
    
    categories = list(data.keys())
    values = list(data.values())
    
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        mode='lines+markers',
        line=dict(
            color=theme['primary_color'],
            width=2
        ),
        marker=dict(
            size=8,
            color=theme['primary_color'],
            line=dict(color=colors['paper_bg'], width=2)
        ),
        fill='toself',
        fillcolor=hex_to_rgba(theme['primary_color'], 0.25),
        name='분포',
        hovertemplate='<b>%{theta}</b><br>활동: %{r}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=colors['text_color'])
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                showgrid=True,
                gridcolor=colors['grid_color'],
                showticklabels=True,
                tickfont=dict(color=colors['text_color'])
            ),
            angularaxis=dict(
                showgrid=True,
                gridcolor=colors['grid_color'],
                tickfont=dict(color=colors['text_color'])
            ),
            bgcolor=colors['plot_bg']
        ),
        showlegend=False,
        height=450,
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor=colors['paper_bg'],
        font={'family': 'Pretendard, sans-serif', 'color': colors['text_color']}
    )
    
    return fig


def create_themed_gauge_chart(
    value: float, 
    theme: Dict[str, Any], 
    title: str = "점수",
    dark_mode: bool = False
) -> go.Figure:
    """테마를 적용한 게이지 차트"""
    colors = get_chart_colors(theme, dark_mode)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={
            'text': title, 
            'font': {'size': 16, 'color': colors['text_color']}
        },
        number={
            'suffix': "%", 
            'font': {'size': 28, 'color': theme['primary_color']}
        },
        gauge={
            'axis': {
                'range': [None, 100], 
                'tickwidth': 1,
                'tickcolor': colors['text_color']
            },
            'bar': {'color': theme['primary_color']},
            'bgcolor': colors['plot_bg'],
            'borderwidth': 2,
            'bordercolor': colors['grid_color'],
            'steps': [
                {'range': [0, 33], 'color': colors['grid_color']},
                {'range': [33, 66], 'color': theme['accent_color']},
                {'range': [66, 100], 'color': hex_to_rgba(theme['primary_color'], 0.4)}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=colors['paper_bg'],
        font={'family': 'Pretendard, sans-serif', 'color': colors['text_color']}
    )
    
    return fig


def create_themed_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    theme: Dict[str, Any],
    title: str = "막대 차트",
    dark_mode: bool = False
) -> go.Figure:
    """테마를 적용한 막대 차트"""
    colors = get_chart_colors(theme, dark_mode)
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(
            text="📊 데이터가 없습니다",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['text_color'])
        )
    else:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker=dict(
                color=theme['primary_color'],
                line=dict(color=colors['grid_color'], width=1)
            ),
            hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=colors['text_color'])
        ),
        xaxis_title=x_col,
        yaxis_title=y_col,
        plot_bgcolor=colors['plot_bg'],
        paper_bgcolor=colors['paper_bg'],
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor=colors['grid_color'],
            color=colors['text_color']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid_color'],
            color=colors['text_color']
        ),
        font={'family': 'Pretendard, sans-serif', 'color': colors['text_color']}
    )
    
    return fig


def create_themed_timeline_chart(
    df: pd.DataFrame,
    theme: Dict[str, Any],
    title: str = "타임라인",
    dark_mode: bool = False
) -> go.Figure:
    """테마를 적용한 타임라인 차트"""
    colors = get_chart_colors(theme, dark_mode)
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(
            text="📊 데이터가 없습니다",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['text_color'])
        )
    else:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['daily_count'],
            mode='lines+markers',
            line=dict(
                color=theme['primary_color'],
                width=2.5
            ),
            marker=dict(
                size=6,
                color=theme['primary_color'],
                line=dict(color=colors['paper_bg'], width=1)
            ),
            fill='tozeroy',
            fillcolor=hex_to_rgba(theme['primary_color'], 0.15),
            name='활동',
            hovertemplate='<b>%{x}</b><br>값: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=colors['text_color'])
        ),
        xaxis_title="날짜",
        yaxis_title="활동",
        plot_bgcolor=colors['plot_bg'],
        paper_bgcolor=colors['paper_bg'],
        height=350,
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor=colors['grid_color'],
            showline=True,
            linecolor=colors['grid_color'],
            color=colors['text_color']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid_color'],
            showline=True,
            linecolor=colors['grid_color'],
            color=colors['text_color']
        ),
        hovermode='x unified',
        font={'family': 'Pretendard, sans-serif', 'color': colors['text_color']}
    )
    
    return fig