#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Font utilities for matplotlib plots.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def configure_fonts_for_plots():
    """
    Configure matplotlib to properly display Chinese characters and special symbols.
    Sets appropriate fonts and handling for minus signs.
    
    This function should be called before creating any plots that need to display
    Chinese characters or special symbols like minus signs.
    """
    # First, set axes.unicode_minus to False to ensure minus signs display correctly
    plt.rcParams['axes.unicode_minus'] = False
    
    # Try to find a font that supports Chinese characters
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK JP', 
                    'Noto Sans CJK SC', 'Noto Sans CJK TC', 'AR PL UMing CN']
    
    # Add these fonts to the sans-serif family
    plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams.get('font.sans-serif', [])
    
    # Try to set a specific font as primary if available
    for font_name in chinese_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and os.path.exists(font_path):
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'].insert(0, font_name)  # Set as primary font
                break
        except Exception:
            continue 