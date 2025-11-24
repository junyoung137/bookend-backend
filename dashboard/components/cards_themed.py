"""
Themed card components for multi-themed dashboard.
눈에 편한 부드러운 카드 디자인 - 다크모드 개선
"""
import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime


def create_metric_card(
    theme: Dict[str, Any],
    title: str,
    value: str,
    delta: Optional[str] = None,
    icon: str = "📊",
    help_text: Optional[str] = None,
    dark_mode: bool = False
) -> None:
    """눈에 편한 메트릭 카드 (다크모드 개선)"""
    
    # 다크모드 색상 설정 - 더 명확한 구분
    if dark_mode:
        card_bg = '#3f4d5e'  # 더 밝은 카드 배경
        text_color = theme.get('dark_text', '#e2e8f0')
        secondary_color = theme.get('dark_secondary', '#94a3b8')
        border_color = '#5a6b7f'  # 더 밝은 테두리
    else:
        card_bg = '#ffffff'
        text_color = theme['primary_color']
        secondary_color = theme['secondary_color']
        border_color = '#e2e8f0'
    
    delta_html = ""
    if delta:
        delta_color = theme['primary_color'] if delta.startswith('+') else theme['secondary_color']
        delta_html = f"""
            <div style="
                display: inline-block;
                padding: 0.3rem 0.9rem;
                background: {delta_color}15;
                color: {delta_color};
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                margin-top: 0.8rem;
            ">
                {delta}
            </div>
        """
    
    help_html = ""
    if help_text:
        help_html = f"""
            <div style="
                font-size: 11px;
                color: {secondary_color};
                margin-top: 0.8rem;
                opacity: 0.8;
            ">
                💡 {help_text}
            </div>
        """
    
    st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 2px solid {border_color};
            border-radius: 20px;
            padding: 2rem 1.8rem;
            box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'});
            transition: all 0.3s ease;
            height: 100%;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(0,0,0,{'0.4' if dark_mode else '0.12'})';" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'})';">
            <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem;">
                <div style="font-size: 28px;">{icon}</div>
                <div style="
                    font-size: 12px;
                    color: {secondary_color};
                    font-weight: 500;
                    letter-spacing: 0.3px;
                ">
                    {title}
                </div>
            </div>
            <div style="
                font-size: 32px;
                font-weight: 700;
                color: {theme['primary_color']};
                line-height: 1.2;
            ">
                {value}
            </div>
            {delta_html}
            {help_html}
        </div>
    """, unsafe_allow_html=True)


def create_info_card(
    theme: Dict[str, Any],
    title: str,
    content: str,
    icon: str = "ℹ️",
    card_type: str = "info",
    dark_mode: bool = False
) -> None:
    """부드러운 정보 카드 (다크모드 개선)"""
    
    if dark_mode:
        card_bg = '#3f4d5e'
        text_color = theme.get('dark_text', '#e2e8f0')
        secondary_color = theme.get('dark_secondary', '#94a3b8')
    else:
        card_bg = '#ffffff'
        text_color = theme['primary_color']
        secondary_color = theme['secondary_color']
    
    type_colors = {
        "info": (theme['primary_color'], theme['light_bg']),
        "success": ("#10b981", "#d1fae5"),
        "warning": ("#f59e0b", "#fef3c7"),
        "error": ("#ef4444", "#fee2e2")
    }
    
    border_color, bg_color = type_colors.get(card_type, type_colors["info"])
    
    st.markdown(f"""
        <div style="
            background: {card_bg};
            border-left: 4px solid {border_color};
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'});
        ">
            <div style="display: flex; align-items: flex-start; gap: 1.2rem;">
                <div style="font-size: 24px; line-height: 1;">{icon}</div>
                <div style="flex: 1;">
                    <h4 style="
                        margin: 0 0 0.6rem 0;
                        font-size: 15px;
                        font-weight: 600;
                        color: {border_color};
                    ">
                        {title}
                    </h4>
                    <p style="
                        margin: 0;
                        font-size: 13px;
                        color: {secondary_color};
                        line-height: 1.7;
                    ">
                        {content}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_stat_card(
    theme: Dict[str, Any],
    stats: List[Dict[str, Any]],
    dark_mode: bool = False
) -> None:
    """부드러운 통계 카드 (다크모드 개선)"""
    
    if dark_mode:
        card_bg = '#3f4d5e'
        text_color = theme.get('dark_text', '#e2e8f0')
        border_color = '#5a6b7f'
    else:
        card_bg = '#ffffff'
        text_color = theme['primary_color']
        border_color = '#e2e8f0'
    
    stats_html = ""
    for i, stat in enumerate(stats):
        border_style = "" if i == len(stats) - 1 else f"border-bottom: 1px solid {border_color};"
        
        stats_html += f"""
            <div style="
                padding: 1.3rem 0;
                {border_style}
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="
                            font-size: 12px;
                            color: {theme['secondary_color']};
                            margin-bottom: 0.4rem;
                            font-weight: 500;
                        ">
                            {stat.get('icon', '•')} {stat['label']}
                        </div>
                        <div style="
                            font-size: 22px;
                            font-weight: 700;
                            color: {theme['primary_color']};
                        ">
                            {stat['value']}
                        </div>
                    </div>
                    {'<div style="font-size: 11px; color: ' + theme['accent_color'] + '; font-weight: 600; background: ' + theme['light_bg'] + '; padding: 0.3rem 0.8rem; border-radius: 12px;">' + stat.get('badge', '') + '</div>' if stat.get('badge') else ''}
                </div>
            </div>
        """
    
    st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 2px solid {border_color};
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'});
        ">
            {stats_html}
        </div>
    """, unsafe_allow_html=True)


def create_progress_card(
    theme: Dict[str, Any],
    title: str,
    current: float,
    total: float,
    unit: str = "",
    icon: str = "📈",
    dark_mode: bool = False
) -> None:
    """부드러운 진행률 카드 (다크모드 개선)"""
    
    if dark_mode:
        card_bg = '#3f4d5e'
        text_color = theme.get('dark_text', '#e2e8f0')
        secondary_color = theme.get('dark_secondary', '#94a3b8')
        border_color = '#5a6b7f'
        progress_bg = '#2a3441'
    else:
        card_bg = '#ffffff'
        text_color = theme['primary_color']
        secondary_color = theme['secondary_color']
        border_color = '#e2e8f0'
        progress_bg = theme['light_bg']
    
    percentage = (current / total * 100) if total > 0 else 0
    
    st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 2px solid {border_color};
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'});
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 0.7rem;">
                    <span style="font-size: 22px;">{icon}</span>
                    <span style="
                        font-size: 15px;
                        font-weight: 600;
                        color: {text_color};
                    ">{title}</span>
                </div>
                <div style="
                    font-size: 13px;
                    color: {secondary_color};
                    font-weight: 600;
                ">
                    {percentage:.1f}%
                </div>
            </div>
            
            <div style="
                background: {progress_bg};
                border-radius: 12px;
                height: 10px;
                overflow: hidden;
                margin-bottom: 1rem;
            ">
                <div style="
                    background: {theme['primary_color']};
                    height: 100%;
                    width: {percentage}%;
                    border-radius: 12px;
                    transition: width 0.6s ease;
                "></div>
            </div>
            
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: {secondary_color};
            ">
                <span>{current:,.0f} {unit}</span>
                <span>{total:,.0f} {unit}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_list_card(
    theme: Dict[str, Any],
    title: str,
    items: List[Dict[str, Any]],
    icon: str = "📋",
    dark_mode: bool = False
) -> None:
    """부드러운 리스트 카드 (다크모드 개선)"""
    
    if dark_mode:
        card_bg = '#3f4d5e'
        text_color = theme.get('dark_text', '#e2e8f0')
        secondary_color = theme.get('dark_secondary', '#94a3b8')
        border_color = '#5a6b7f'
        item_bg = '#2a3441'
    else:
        card_bg = '#ffffff'
        text_color = theme['primary_color']
        secondary_color = theme['secondary_color']
        border_color = '#e2e8f0'
        item_bg = theme['light_bg']
    
    items_html = ""
    for i, item in enumerate(items):
        items_html += f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem 0;
                border-bottom: {'1px solid ' + border_color if i < len(items) - 1 else 'none'};
            ">
                <div style="display: flex; align-items: center; gap: 1rem; flex: 1;">
                    <div style="
                        width: 36px;
                        height: 36px;
                        background: {item_bg};
                        border-radius: 12px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 16px;
                    ">
                        {item.get('icon', '•')}
                    </div>
                    <div style="flex: 1;">
                        <div style="
                            font-size: 13px;
                            font-weight: 600;
                            color: {text_color};
                            margin-bottom: 0.2rem;
                        ">
                            {item['title']}
                        </div>
                        {'<div style="font-size: 11px; color: ' + secondary_color + ';">' + item.get('subtitle', '') + '</div>' if item.get('subtitle') else ''}
                    </div>
                </div>
                {'<div style="font-size: 13px; font-weight: 700; color: ' + theme['primary_color'] + ';">' + item.get('value', '') + '</div>' if item.get('value') else ''}
            </div>
        """
    
    st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 2px solid {border_color};
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'});
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 0.8rem;
                margin-bottom: 1.5rem;
                padding-bottom: 1.5rem;
                border-bottom: 1px solid {border_color};
            ">
                <span style="font-size: 24px;">{icon}</span>
                <h3 style="
                    margin: 0;
                    font-size: 17px;
                    font-weight: 600;
                    color: {text_color};
                ">
                    {title}
                </h3>
            </div>
            {items_html}
        </div>
    """, unsafe_allow_html=True)


def create_feature_card(
    theme: Dict[str, Any],
    icon: str,
    title: str,
    description: str,
    tags: Optional[List[str]] = None,
    dark_mode: bool = False
) -> None:
    """부드러운 기능 소개 카드 (다크모드 개선)"""
    
    if dark_mode:
        card_bg = '#3f4d5e'
        text_color = theme.get('dark_text', '#e2e8f0')
        secondary_color = theme.get('dark_secondary', '#94a3b8')
        border_color = '#5a6b7f'
        tag_bg = '#2a3441'
    else:
        card_bg = '#ffffff'
        text_color = theme['primary_color']
        secondary_color = theme['secondary_color']
        border_color = '#e2e8f0'
        tag_bg = theme['light_bg']
    
    tags_html = ""
    if tags:
        tags_html = '<div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.2rem;">'
        for tag in tags:
            tags_html += f"""
                <span style="
                    display: inline-block;
                    padding: 0.3rem 0.8rem;
                    background: {tag_bg};
                    color: {theme['primary_color']};
                    border-radius: 12px;
                    font-size: 10px;
                    font-weight: 600;
                    letter-spacing: 0.3px;
                ">
                    {tag}
                </span>
            """
        tags_html += '</div>'
    
    st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 2px solid {border_color};
            border-left: 4px solid {theme['primary_color']};
            border-radius: 16px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'});
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateX(2px)'; this.style.boxShadow='0 6px 20px rgba(0,0,0,{'0.4' if dark_mode else '0.12'})';" 
           onmouseout="this.style.transform='translateX(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'})';">
            <div style="display: flex; align-items: flex-start; gap: 1.5rem;">
                <div style="
                    font-size: 42px;
                    line-height: 1;
                    flex-shrink: 0;
                ">
                    {icon}
                </div>
                <div style="flex: 1;">
                    <h3 style="
                        margin: 0 0 0.7rem 0;
                        font-size: 18px;
                        font-weight: 600;
                        color: {text_color};
                    ">
                        {title}
                    </h3>
                    <p style="
                        margin: 0;
                        font-size: 13px;
                        color: {secondary_color};
                        line-height: 1.7;
                    ">
                        {description}
                    </p>
                    {tags_html}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_timeline_item(
    theme: Dict[str, Any],
    date: str,
    title: str,
    description: str,
    icon: str = "📅",
    is_last: bool = False,
    dark_mode: bool = False
) -> None:
    """부드러운 타임라인 아이템 (다크모드 개선)"""
    
    if dark_mode:
        card_bg = '#3f4d5e'
        text_color = theme.get('dark_text', '#e2e8f0')
        secondary_color = theme.get('dark_secondary', '#94a3b8')
        border_color = '#5a6b7f'
        connector_bg = '#5a6b7f'
    else:
        card_bg = '#ffffff'
        text_color = theme['primary_color']
        secondary_color = theme['secondary_color']
        border_color = '#e2e8f0'
        connector_bg = theme['light_bg']
    
    connector_html = "" if is_last else f"""
        <div style="
            position: absolute;
            left: 19px;
            top: 48px;
            width: 2px;
            height: calc(100% - 48px);
            background: {connector_bg};
        "></div>
    """
    
    st.markdown(f"""
        <div style="
            position: relative;
            padding-left: 3.5rem;
            padding-bottom: {'0' if is_last else '2rem'};
        ">
            <div style="
                position: absolute;
                left: 0;
                top: 4px;
                width: 40px;
                height: 40px;
                background: {theme['primary_color']};
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
                {icon}
            </div>
            {connector_html}
            <div style="
                background: {card_bg};
                border: 2px solid {border_color};
                border-radius: 16px;
                padding: 1.3rem 1.5rem;
                box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'});
            ">
                <div style="
                    font-size: 11px;
                    color: {secondary_color};
                    font-weight: 500;
                    margin-bottom: 0.3rem;
                ">
                    {date}
                </div>
                <div style="
                    font-size: 15px;
                    font-weight: 600;
                    color: {text_color};
                    margin-bottom: 0.6rem;
                ">
                    {title}
                </div>
                <div style="
                    font-size: 13px;
                    color: {secondary_color};
                    line-height: 1.6;
                ">
                    {description}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_comparison_card(
    theme: Dict[str, Any],
    title: str,
    items: List[Dict[str, Any]],
    dark_mode: bool = False
) -> None:
    """부드러운 비교 카드 (다크모드 개선)"""
    
    if dark_mode:
        card_bg = '#3f4d5e'
        text_color = theme.get('dark_text', '#e2e8f0')
        border_color = '#5a6b7f'
        progress_bg = '#2a3441'
    else:
        card_bg = '#ffffff'
        text_color = theme['primary_color']
        border_color = '#e2e8f0'
        progress_bg = theme['light_bg']
    
    items_html = ""
    for item in items:
        percentage = item.get('percentage', 0)
        items_html += f"""
            <div style="margin-bottom: 1.5rem;">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 0.7rem;
                ">
                    <div style="
                        font-size: 13px;
                        font-weight: 600;
                        color: {text_color};
                    ">
                        {item.get('icon', '•')} {item['label']}
                    </div>
                    <div style="
                        font-size: 13px;
                        font-weight: 700;
                        color: {theme['primary_color']};
                    ">
                        {item['value']}
                    </div>
                </div>
                <div style="
                    background: {progress_bg};
                    border-radius: 10px;
                    height: 8px;
                    overflow: hidden;
                ">
                    <div style="
                        background: {theme['primary_color']};
                        height: 100%;
                        width: {percentage}%;
                        border-radius: 10px;
                        transition: width 0.6s ease;
                    "></div>
                </div>
            </div>
        """
    
    st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 2px solid {border_color};
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.08'});
        ">
            <h3 style="
                margin: 0 0 2rem 0;
                font-size: 17px;
                font-weight: 600;
                color: {text_color};
            ">
                {title}
            </h3>
            {items_html}
        </div>
    """, unsafe_allow_html=True)