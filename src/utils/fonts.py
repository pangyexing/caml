#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Font utilities for matplotlib plots.
"""

import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def fix_minus_sign_issue():
    """
    Fix the Unicode minus sign issue in matplotlib by directly patching formatters.
    This works by replacing U+2212 with ASCII hyphen in all relevant formatter classes.
    """
    # Suppress all warnings about missing glyphs first
    warnings.filterwarnings(
        "ignore", 
        message="Font .* does not have a glyph for .* \[U\+2212\].*"
    )
    
    # First, disable Unicode minus in rcParams
    plt.rcParams['axes.unicode_minus'] = False
    
    # Import the formatters
    # Also patch matplotlib.ticker directly
    import matplotlib.ticker as ticker
    from matplotlib.ticker import (
        FormatStrFormatter,
        ScalarFormatter,
        StrMethodFormatter,
    )
    
    # Replace the negative sign in the ticker._mathdefault method
    original_mathdefault = getattr(ticker, '_mathdefault', None)
    if original_mathdefault:
        def patched_mathdefault(s):
            return original_mathdefault(s).replace('\u2212', '-')
        setattr(ticker, '_mathdefault', patched_mathdefault)
    
    # Replace U+2212 with ASCII hyphen in format_data methods
    
    # 1. Patch ScalarFormatter
    orig_scalar_format_data = ScalarFormatter.format_data
    def patched_scalar_format_data(self, value):
        return orig_scalar_format_data(self, value).replace('\u2212', '-')
    ScalarFormatter.format_data = patched_scalar_format_data
    
    # 2. Patch __call__ method of ScalarFormatter
    orig_scalar_call = ScalarFormatter.__call__
    def patched_scalar_call(self, x, pos=None):
        return orig_scalar_call(self, x, pos).replace('\u2212', '-')
    ScalarFormatter.__call__ = patched_scalar_call
    
    # 3. Patch StrMethodFormatter
    orig_str_method_call = StrMethodFormatter.__call__
    def patched_str_method_call(self, x, pos=None):
        return orig_str_method_call(self, x, pos).replace('\u2212', '-')
    StrMethodFormatter.__call__ = patched_str_method_call
    
    # 4. Patch FormatStrFormatter
    orig_format_str_call = FormatStrFormatter.__call__
    def patched_format_str_call(self, x, pos=None):
        return orig_format_str_call(self, x, pos).replace('\u2212', '-')
    FormatStrFormatter.__call__ = patched_format_str_call
    
    # 5. Fix tick label formatter for all existing axes
    try:
        fig = plt.gcf()
        for ax in fig.get_axes():
            ax.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, pos: f"{x}".replace('\u2212', '-')))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, pos: f"{x}".replace('\u2212', '-')))
    except:
        # Skip if there are no figures or axes yet
        pass
        
    # 6. Create custom formatter for future use
    class AsciiFormatter(ticker.ScalarFormatter):
        def __call__(self, x, pos=None):
            result = super().__call__(x, pos)
            return result.replace('\u2212', '-')
            
    # Register a custom formatter factory for global use
    def get_safe_formatter():
        fmt = AsciiFormatter()
        fmt.set_useOffset(False)
        return fmt
        
    # Store this function for later use
    fix_minus_sign_issue.get_safe_formatter = get_safe_formatter


def configure_fonts_for_plots():
    """
    Configure matplotlib to properly display Chinese characters and special symbols.
    Sets appropriate fonts and handling for minus signs.
    
    This function should be called before creating any plots that need to display
    Chinese characters or special symbols like minus signs.
    """
    # Apply the minus sign fix first
    fix_minus_sign_issue()
    
    # Suppress font warnings
    warnings.filterwarnings('ignore', message="Font family ['SimHei'] not found.*")
    warnings.filterwarnings('ignore', message="Glyph \d+ .* missing from current font")
    
    # First check which fonts are actually available on the system
    # DejaVu Sans is already available on most systems as a fallback
    available_fonts = set(f.name for f in fm.FontManager().ttflist)
    
    # Prioritize fallback fonts first that have better Unicode coverage
    fallback_fonts = ['DejaVu Sans', 'Arial', 'Verdana', 'Tahoma', 'Helvetica']
    
    # Chinese fonts - only use if actually available
    chinese_fonts = []
    for font_name in ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', 
                      'Microsoft YaHei', 'WenQuanYi Zen Hei', 'SimHei', 'AR PL UMing CN']:
        if font_name in available_fonts:
            chinese_fonts.append(font_name)
    
    # Set the font order: available Chinese fonts first, then fallbacks
    font_order = chinese_fonts + fallback_fonts
    
    # Set the font configuration
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_order + plt.rcParams.get('font.sans-serif', [])
    
    # Explicitly set figure.autolayout to True to better handle text
    plt.rcParams['figure.autolayout'] = True
    
    # Completely disable mathtext and use plain text
    plt.rcParams['text.usetex'] = False
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'
    mpl.rcParams['mathtext.default'] = 'regular'
    
    # Force simpler tick formatting that doesn't use Unicode
    mpl.rcParams['axes.formatter.use_mathtext'] = False
    
    # Additional settings for macOS
    if sys.platform == 'darwin':
        # Try to find system fonts on macOS
        system_font_dirs = ['/System/Library/Fonts', '/Library/Fonts', os.path.expanduser('~/Library/Fonts')]
        for font_dir in system_font_dirs:
            if os.path.exists(font_dir):
                try:
                    fm.fontManager.addfont(font_dir)
                except:
                    pass
    
    # Additional settings for Windows platform
    elif sys.platform.startswith('win'):
        # Try to register additional system fonts on Windows
        try:
            fm._rebuild()
        except Exception:
            pass
        
        # Windows-specific settings
        if 'SimHei' in mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
            plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
    
    # Set custom formatter for any future axes that get created
    try:
        safe_formatter = fix_minus_sign_issue.get_safe_formatter()
        mpl.rcParams['axes.formatter.use_mathtext'] = False
        
        # Configure seaborn to use our safe formatter if it's available
        try:
            import seaborn as sns
            orig_axes_style = sns.axes_style
            
            def patched_axes_style(*args, **kwargs):
                style = orig_axes_style(*args, **kwargs)
                if 'axes.unicode_minus' in style:
                    style['axes.unicode_minus'] = False
                return style
            
            sns.axes_style = patched_axes_style
        except ImportError:
            pass
    except:
        pass
        
    # Set the backend to agg for better compatibility
    plt.switch_backend('agg') 