#!/usr/bin/env python3
"""
Quantitative Financial Analysis Model with Enhanced Valuation System and Market Conditions
Complete integrated version with all fixes for pandas compatibility
"""

import os
# Set environment variables BEFORE other imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import time
import re
import gc
from typing import List, Dict, Tuple
from scipy import stats

# ML imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor

# TextBlob for sentiment analysis
from textblob import TextBlob

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent crashes
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Reddit scraping function (outside class)
def scrape_reddit_internal_api(symbol):
    """Use Reddit's internal API that the website uses"""
    
    print(f"\nSearching Reddit for {symbol}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.reddit.com/',
    }
    
    # Try multiple subreddits
    subreddits = ['stocks', 'wallstreetbets', 'investing', 'StockMarket']
    all_posts = []
    
    for subreddit in subreddits:
        search_params = {
            'q': symbol,
            'restrict_sr': 'true',
            'sr': subreddit,
            'sort': 'relevance',
            't': 'month',
            'type': 'link',
            'include_over_18': 'on'
        }
        
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        
        try:
            response = requests.get(url, params=search_params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and 'children' in data['data']:
                    for post in data['data']['children']:
                        post_data = post['data']
                        all_posts.append({
                            'title': post_data['title'],
                            'text': post_data.get('selftext', '')[:500],
                            'score': post_data['score'],
                            'num_comments': post_data['num_comments'],
                            'created_utc': post_data['created_utc'],
                            'upvote_ratio': post_data.get('upvote_ratio', 0.8),
                            'url': f"https://reddit.com{post_data['permalink']}",
                            'subreddit': subreddit
                        })
                
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error searching r/{subreddit}: {e}")
    
    print(f"Found {len(all_posts)} total posts across subreddits")
    return all_posts

class MarketConditionsAnalyzer:
    """Analyze market conditions and economic indicators"""
    
    def __init__(self):
        self.economic_indicators = {
            'SPY': 'S&P 500 (Market)',
            '^VIX': 'Volatility Index',
            '^TNX': '10-Year Treasury Yield',
            'DX-Y.NYB': 'US Dollar Index',
            'GC=F': 'Gold Futures',
            'CL=F': 'Oil Futures',
            '^IRX': '3-Month Treasury',
            'HYG': 'High Yield Bonds',
            'TLT': 'Long-Term Treasuries'
        }
        
        # Sector-specific indicators
        self.sector_indicators = {
            'Information Technology': ['QQQ', 'VGT', 'SMH'],
            'Financials': ['XLF', 'KBE', 'KRE'],
            'Health Care': ['XLV', 'IBB', 'XBI'],
            'Consumer Discretionary': ['XLY', 'XRT', 'ITB'],
            'Energy': ['XLE', 'OIH', 'XOP'],
            'Industrials': ['XLI', 'ITA', 'JETS'],
            'Materials': ['XLB', 'GDX', 'COPX'],
            'Real Estate': ['XLRE', 'IYR', 'REZ'],
            'Consumer Staples': ['XLP', 'VDC'],
            'Utilities': ['XLU', 'IDU'],
            'Communication Services': ['XLC', 'SOCL']
        }
        
        self.macro_data = {}
        self.sector_conditions = {}
    
    def fetch_economic_data(self) -> Dict:
        """Fetch current economic indicators"""
        print("\nFetching economic indicators...")
        economic_data = {}
        
        for symbol, name in self.economic_indicators.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='6mo')
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    month_ago = hist['Close'].iloc[-22] if len(hist) > 22 else hist['Close'].iloc[0]
                    three_month_ago = hist['Close'].iloc[-66] if len(hist) > 66 else hist['Close'].iloc[0]
                    
                    economic_data[name] = {
                        'current': current,
                        'month_change': (current - month_ago) / month_ago * 100,
                        'quarter_change': (current - three_month_ago) / three_month_ago * 100,
                        'trend': 'up' if current > month_ago else 'down',
                        'volatility': hist['Close'].pct_change().std() * np.sqrt(252)
                    }
            except Exception as e:
                print(f"Error fetching {name}: {e}")
        
        self.macro_data = economic_data
        return economic_data
    
    def analyze_yield_curve(self) -> Dict:
        """Analyze yield curve for recession signals"""
        try:
            ten_year = yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]
            three_month = yf.Ticker('^IRX').history(period='1d')['Close'].iloc[-1]
            
            yield_spread = ten_year - three_month
            
            return {
                'ten_three_spread': yield_spread,
                'inverted': yield_spread < 0,
                'recession_risk': 'High' if yield_spread < 0 else ('Medium' if yield_spread < 0.5 else 'Low'),
                'curve_shape': 'Inverted' if yield_spread < 0 else ('Flat' if yield_spread < 0.5 else 'Normal')
            }
        except:
            return {'recession_risk': 'Unknown', 'curve_shape': 'Unknown', 'ten_three_spread': 0}
    
    def get_market_regime(self) -> str:
        """Determine current market regime"""
        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='6mo')
            
            returns = spy_hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            ma50 = spy_hist['Close'].rolling(50).mean().iloc[-1]
            ma200 = spy_hist['Close'].rolling(200).mean().iloc[-1] if len(spy_hist) > 200 else ma50
            current_price = spy_hist['Close'].iloc[-1]
            
            vix = yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1]
            
            if current_price > ma50 > ma200 and vix < 20:
                return "Bull Market - Low Volatility"
            elif current_price > ma50 > ma200 and vix >= 20:
                return "Bull Market - High Volatility"
            elif current_price < ma50 < ma200:
                return "Bear Market"
            elif ma50 < current_price < ma200 or ma200 < current_price < ma50:
                return "Transition/Consolidation"
            else:
                return "Neutral"
        except:
            return "Unknown"
    
    def analyze_sector_conditions(self, sector: str) -> Dict:
        """Analyze specific sector conditions and momentum"""
        print(f"\nAnalyzing {sector} sector conditions...")
        
        sector_data = {
            'sector': sector,
            'etf_performance': {},
            'relative_strength': 0,
            'momentum': 'neutral',
            'outlook': 'neutral'
        }
        
        if sector in self.sector_indicators:
            etfs = self.sector_indicators[sector]
            performances = []
            
            for etf in etfs:
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period='3mo')
                    
                    if not hist.empty:
                        returns_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-22] - 1) * 100 if len(hist) > 22 else 0
                        returns_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                        
                        sector_data['etf_performance'][etf] = {
                            '1m_return': returns_1m,
                            '3m_return': returns_3m
                        }
                        performances.append(returns_1m)
                except:
                    continue
            
            # Compare to SPY
            try:
                spy = yf.Ticker('SPY')
                spy_hist = spy.history(period='3mo')
                spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-22] - 1) * 100
                
                avg_sector_return = np.mean(performances) if performances else 0
                sector_data['relative_strength'] = avg_sector_return - spy_return
                
                if avg_sector_return > spy_return + 2:
                    sector_data['momentum'] = 'strong_positive'
                    sector_data['outlook'] = 'bullish'
                elif avg_sector_return > spy_return:
                    sector_data['momentum'] = 'positive'
                    sector_data['outlook'] = 'moderately_bullish'
                elif avg_sector_return < spy_return - 2:
                    sector_data['momentum'] = 'negative'
                    sector_data['outlook'] = 'bearish'
                else:
                    sector_data['momentum'] = 'neutral'
                    sector_data['outlook'] = 'neutral'
            except:
                pass
        
        self.sector_conditions = sector_data
        return sector_data
    
    def get_federal_reserve_stance(self) -> Dict:
        """Analyze Federal Reserve policy stance"""
        try:
            treasury = yf.Ticker('^IRX')
            hist = treasury.history(period='1y')
            
            current_rate = hist['Close'].iloc[-1]
            six_months_ago = hist['Close'].iloc[-126] if len(hist) > 126 else current_rate
            
            if current_rate > six_months_ago + 0.5:
                stance = "Tightening (Hawkish)"
                impact = "negative"
            elif current_rate < six_months_ago - 0.5:
                stance = "Easing (Dovish)"
                impact = "positive"
            else:
                stance = "Neutral/Pause"
                impact = "neutral"
            
            return {
                'current_rate': current_rate,
                'stance': stance,
                'market_impact': impact,
                '6m_change': current_rate - six_months_ago
            }
        except:
            return {'stance': 'Unknown', 'market_impact': 'neutral', 'current_rate': 0}
    
    def calculate_market_adjustment_factor(self, sector: str) -> float:
        """Calculate overall market adjustment factor for predictions"""
        factors = []
        weights = []
        
        # Market regime factor
        regime = self.get_market_regime()
        if 'Bull' in regime:
            factors.append(1.15)
        elif 'Bear' in regime:
            factors.append(0.85)
        else:
            factors.append(1.0)
        weights.append(0.25)
        
        # Yield curve factor
        yield_curve = self.analyze_yield_curve()
        if yield_curve.get('recession_risk') == 'High':
            factors.append(0.8)
        elif yield_curve.get('recession_risk') == 'Low':
            factors.append(1.1)
        else:
            factors.append(1.0)
        weights.append(0.20)
        
        # Fed policy factor
        fed = self.get_federal_reserve_stance()
        if fed.get('market_impact') == 'positive':
            factors.append(1.1)
        elif fed.get('market_impact') == 'negative':
            factors.append(0.9)
        else:
            factors.append(1.0)
        weights.append(0.15)
        
        # Sector momentum factor
        sector_analysis = self.analyze_sector_conditions(sector)
        if sector_analysis.get('momentum') == 'strong_positive':
            factors.append(1.2)
        elif sector_analysis.get('momentum') == 'positive':
            factors.append(1.1)
        elif sector_analysis.get('momentum') == 'negative':
            factors.append(0.85)
        else:
            factors.append(1.0)
        weights.append(0.30)
        
        # VIX factor
        try:
            vix = yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1]
            if vix < 15:
                factors.append(1.05)
            elif vix > 30:
                factors.append(0.9)
            else:
                factors.append(1.0)
            weights.append(0.10)
        except:
            factors.append(1.0)
            weights.append(0.10)
        
        # Calculate weighted adjustment
        adjustment = sum(f * w for f, w in zip(factors, weights))
        
        return max(0.7, min(1.3, adjustment))
    
    def generate_economic_summary(self, sector: str) -> str:
        """Generate a summary of economic conditions"""
        regime = self.get_market_regime()
        yield_curve = self.analyze_yield_curve()
        fed = self.get_federal_reserve_stance()
        sector_analysis = self.analyze_sector_conditions(sector)
        
        summary = f"""
## Current Market & Economic Conditions

### Overall Market Environment
- **Market Regime**: {regime}
- **Yield Curve**: {yield_curve.get('curve_shape', 'Unknown')} (10-3 Spread: {yield_curve.get('ten_three_spread', 0):.2f}%)
- **Recession Risk**: {yield_curve.get('recession_risk', 'Unknown')}
- **Fed Policy Stance**: {fed.get('stance', 'Unknown')}
- **Current Fed Funds Proxy**: {fed.get('current_rate', 0):.2f}%

### {sector} Sector Analysis
- **Sector Momentum**: {sector_analysis.get('momentum', 'Unknown')}
- **Relative Strength vs S&P 500**: {sector_analysis.get('relative_strength', 0):.2f}%
- **Sector Outlook**: {sector_analysis.get('outlook', 'Unknown')}

### Key Economic Indicators
"""
        
        for indicator, data in list(self.macro_data.items())[:5]:
            if isinstance(data, dict):
                summary += f"- **{indicator}**: {data.get('current', 0):.2f} ({data.get('month_change', 0):+.2f}% monthly)\n"
        
        return summary

class EnhancedValuation:
    """Enhanced valuation system with multiple methods"""
    
    def __init__(self):
        self.valuation_weights = {
            'dcf': 0.15,
            'pe_multiple': 0.15,
            'peg_multiple': 0.10,
            'ev_ebitda': 0.15,
            'price_to_book': 0.10,
            'technical': 0.10,
            'analyst_target': 0.15,
            'ml_prediction': 0.10
        }
        
    def calculate_comprehensive_valuation(self, symbol, ml_prediction=None, sentiment_score=0, market_adjustment=1.0):
        """Calculate price target using multiple valuation methods with market adjustment"""
        
        print(f"\nCalculating comprehensive valuation for {symbol}...")
        
        # Get all necessary data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        
        current_price = info.get('currentPrice', hist['Close'].iloc[-1] if len(hist) > 0 else 0)
        
        # Store individual valuations
        valuations = {}
        
        # 1. DCF Valuation
        valuations['dcf'] = self.calculate_dcf_enhanced(ticker, info, ticker.cashflow)
        
        # 2. P/E Multiple Valuation
        valuations['pe_multiple'] = self.pe_multiple_valuation(info, ticker, current_price)
        
        # 3. PEG Multiple Valuation
        valuations['peg_multiple'] = self.peg_valuation(info, current_price)
        
        # 4. EV/EBITDA Valuation
        valuations['ev_ebitda'] = self.ev_ebitda_valuation(info, financials, current_price)
        
        # 5. Price to Book Valuation
        valuations['price_to_book'] = self.price_to_book_valuation(info, ticker, current_price)
        
        # 6. Technical Analysis Target
        valuations['technical'] = self.technical_analysis_target(hist, current_price)
        
        # 7. Analyst Consensus Target
        valuations['analyst_target'] = self.get_analyst_target(ticker, current_price)
        
        # 8. ML Prediction Integration
        if ml_prediction is not None:
            valuations['ml_prediction'] = current_price * (1 + ml_prediction)
        else:
            valuations['ml_prediction'] = current_price
        
        # Calculate weighted average
        weighted_target = self.calculate_weighted_target(valuations, current_price)
        
        # Apply market adjustment
        weighted_target *= market_adjustment
        
        # Apply sentiment adjustment
        sentiment_adjusted_target = self.apply_sentiment_adjustment(
            weighted_target, sentiment_score, current_price
        )
        
        # Calculate confidence and range
        confidence, low_target, high_target = self.calculate_confidence_range(
            valuations, sentiment_adjusted_target
        )
        
        # Apply market adjustment to range
        low_target *= market_adjustment
        high_target *= market_adjustment
        
        return {
            'current_price': current_price,
            'target_price': sentiment_adjusted_target,
            'upside_potential': (sentiment_adjusted_target - current_price) / current_price if current_price > 0 else 0,
            'confidence': confidence,
            'low_target': low_target,
            'high_target': high_target,
            'valuations': valuations,
            'sentiment_impact': sentiment_score,
            'market_adjustment': market_adjustment
        }
    
    def calculate_dcf_enhanced(self, ticker, info, cash_flow):
        """Enhanced DCF calculation with better assumptions"""
        try:
            if cash_flow.empty:
                return None
                
            fcf = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else 0
            
            if fcf <= 0:
                return None
            
            revenue_growth = info.get('revenueGrowth', 0.05)
            earnings_growth = info.get('earningsGrowth', 0.05)
            
            growth_rate = min(max(np.mean([revenue_growth, earnings_growth]), 0), 0.20)
            
            beta = info.get('beta', 1.0)
            risk_free_rate = 0.045
            market_premium = 0.08
            wacc = risk_free_rate + beta * market_premium
            
            terminal_growth = 0.025
            
            projected_cf = []
            for i in range(1, 11):
                year_growth = growth_rate * (0.9 ** (i-1))
                cf = fcf * ((1 + year_growth) ** i)
                pv = cf / ((1 + wacc) ** i)
                projected_cf.append(pv)
            
            terminal_cf = projected_cf[-1] * (1 + terminal_growth)
            terminal_value = terminal_cf / (wacc - terminal_growth)
            pv_terminal = terminal_value / ((1 + wacc) ** 10)
            
            enterprise_value = sum(projected_cf) + pv_terminal
            
            cash = info.get('totalCash', 0)
            debt = info.get('totalDebt', 0)
            equity_value = enterprise_value + cash - debt
            
            shares = info.get('sharesOutstanding', 1)
            
            return equity_value / shares if shares > 0 else None
            
        except:
            return None
    
    def pe_multiple_valuation(self, info, ticker, current_price):
        """Valuation based on P/E multiples"""
        try:
            eps = info.get('trailingEps', 0)
            if eps <= 0:
                return None
            
            sector_pe = 20
            current_pe = info.get('trailingPE', 0)
            forward_pe = info.get('forwardPE', current_pe)
            
            target_pe = 0.5 * sector_pe + 0.5 * forward_pe
            
            peg = info.get('pegRatio', 1.0)
            if 0.5 < peg < 2.0:
                growth_adjustment = 2.0 - peg
                target_pe *= (1 + growth_adjustment * 0.1)
            
            return eps * target_pe
            
        except:
            return None
    
    def peg_valuation(self, info, current_price):
        """PEG ratio based valuation"""
        try:
            pe = info.get('trailingPE', 0)
            growth = info.get('earningsGrowth', 0)
            
            if pe <= 0 or growth <= 0:
                return None
            
            current_peg = pe / (growth * 100)
            target_peg = 1.25
            target_pe = target_peg * growth * 100
            eps = current_price / pe
            
            return eps * target_pe
            
        except:
            return None
    
    def ev_ebitda_valuation(self, info, financials, current_price):
        """EV/EBITDA multiple valuation"""
        try:
            if not financials.empty and 'EBITDA' in financials.index:
                ebitda = financials.loc['EBITDA'].iloc[0]
            else:
                ebit = info.get('ebitda', 0)
                if ebit <= 0:
                    return None
                ebitda = ebit
            
            market_cap = info.get('marketCap', 0)
            debt = info.get('totalDebt', 0)
            cash = info.get('totalCash', 0)
            
            current_ev = market_cap + debt - cash
            current_ev_ebitda = current_ev / ebitda if ebitda > 0 else 0
            
            target_ev_ebitda = 12.0
            
            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth > 0.15:
                target_ev_ebitda *= 1.2
            elif revenue_growth < 0.05:
                target_ev_ebitda *= 0.8
            
            target_ev = ebitda * target_ev_ebitda
            target_market_cap = target_ev - debt + cash
            shares = info.get('sharesOutstanding', 1)
            
            return target_market_cap / shares if shares > 0 else None
            
        except:
            return None
    
    def price_to_book_valuation(self, info, ticker, current_price):
        """Price to Book multiple valuation"""
        try:
            book_value = info.get('bookValue', 0)
            if book_value <= 0:
                return None
            
            current_pb = current_price / book_value
            roe = info.get('returnOnEquity', 0.10)
            target_pb = roe * 10
            target_pb = min(max(target_pb, 0.8), 5.0)
            
            return book_value * target_pb
            
        except:
            return None
    
    def technical_analysis_target(self, hist, current_price):
        """Technical analysis based price target"""
        try:
            if len(hist) < 252:
                return None
                
            high_52w = hist['High'].rolling(252).max().iloc[-1]
            low_52w = hist['Low'].rolling(252).min().iloc[-1]
            
            ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            ma_200 = hist['Close'].rolling(200).mean().iloc[-1]
            
            fib_0618 = low_52w + 0.618 * (high_52w - low_52w)
            fib_0786 = low_52w + 0.786 * (high_52w - low_52w)
            
            if current_price > ma_50 > ma_200:
                target = fib_0786
            elif current_price > ma_200:
                target = fib_0618
            else:
                target = ma_200
            
            resistance_levels = []
            
            for period in [20, 50, 100]:
                if len(hist) >= period:
                    resistance_levels.append(hist['High'].rolling(period).max().iloc[-1])
            
            resistances_above = [r for r in resistance_levels if r > current_price]
            if resistances_above:
                target = min(target, min(resistances_above) * 1.02)
            
            return target
            
        except:
            return None
    
    def get_analyst_target(self, ticker, current_price):
        """Get analyst consensus price target"""
        try:
            target = ticker.info.get('targetMeanPrice', None)
            if target and target > 0:
                return target
            
            recommendations = ticker.recommendations
            
            if recommendations is not None and not recommendations.empty:
                recent = recommendations.last('3M')
                
                if not recent.empty:
                    strong_buy = len(recent[recent['To Grade'].str.contains('Strong Buy', case=False, na=False)])
                    buy = len(recent[recent['To Grade'].str.contains('Buy', case=False, na=False)])
                    hold = len(recent[recent['To Grade'].str.contains('Hold', case=False, na=False)])
                    sell = len(recent[recent['To Grade'].str.contains('Sell', case=False, na=False)])
                    
                    total = strong_buy + buy + hold + sell
                    
                    if total > 0:
                        score = (strong_buy * 2 + buy * 1 + hold * 0 - sell * 1) / total
                        target_multiplier = 1 + (score * 0.15)
                        return current_price * target_multiplier
            
            return None
            
        except:
            return None
    
    def calculate_weighted_target(self, valuations, current_price):
        """Calculate weighted average of all valuations"""
        valid_valuations = {}
        
        for method, value in valuations.items():
            if value is not None and value > 0:
                if 0.2 * current_price < value < 3 * current_price:
                    valid_valuations[method] = value
        
        if not valid_valuations:
            return current_price
        
        total_weight = sum(self.valuation_weights[method] for method in valid_valuations.keys())
        
        weighted_sum = 0
        for method, value in valid_valuations.items():
            weight = self.valuation_weights[method] / total_weight
            weighted_sum += value * weight
        
        return weighted_sum
    
    def apply_sentiment_adjustment(self, base_target, sentiment_score, current_price):
        """Adjust target based on sentiment"""
        sentiment_multiplier = 1 + (sentiment_score * 0.10)
        
        if sentiment_score > 0 and base_target > current_price:
            sentiment_multiplier *= 1.05
        elif sentiment_score < 0 and base_target < current_price:
            sentiment_multiplier *= 0.95
        
        return base_target * sentiment_multiplier
    
    def calculate_confidence_range(self, valuations, target_price):
        """Calculate confidence level and price range"""
        valid_values = [v for v in valuations.values() if v is not None and v > 0]
        
        if len(valid_values) < 3:
            return 0.5, target_price * 0.85, target_price * 1.15
        
        std_dev = np.std(valid_values)
        mean_val = np.mean(valid_values)
        cv = std_dev / mean_val if mean_val > 0 else 1.0
        
        if cv < 0.15:
            confidence = 0.85
        elif cv < 0.25:
            confidence = 0.70
        elif cv < 0.35:
            confidence = 0.55
        else:
            confidence = 0.40
        
        range_multiplier = 1 - confidence + 0.15
        
        low_target = target_price * (1 - range_multiplier)
        high_target = target_price * (1 + range_multiplier)
        
        return confidence, low_target, high_target

class QuantFinanceMLModel:
    def __init__(self):
        print("Initializing Quantitative Finance ML Model with Market Analysis...")
        print("Using TextBlob for sentiment analysis (Mac compatible)...")
        
        self.sectors_data = {}
        self.selected_stocks = []
        self.training_data = None
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.valuation_model = EnhancedValuation()
        self.market_analyzer = MarketConditionsAnalyzer()
        self.market_adjustment = 1.0
        
    def simple_sentiment(self, text):
        """Simple sentiment analysis using TextBlob"""
        try:
            text = str(text).strip()
            if not text:
                return [{'label': 'neutral', 'score': 0.5}]
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return [{'label': 'positive', 'score': min(polarity + 0.5, 1.0)}]
            elif polarity < -0.1:
                return [{'label': 'negative', 'score': min(-polarity + 0.5, 1.0)}]
            else:
                return [{'label': 'neutral', 'score': 0.5}]
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return [{'label': 'neutral', 'score': 0.5}]
    
    def load_csv_files(self, file_paths=None):
        """Load CSV files from local directory"""
        if file_paths is None:
            print("\nPlease provide CSV file path(s).")
            print("You can drag and drop the file into terminal to get the path.")
            
            file_paths = []
            while True:
                file_path = input("\nEnter CSV file path (or 'done' to finish): ").strip()
                if file_path.lower() == 'done':
                    break
                
                file_path = file_path.strip('"').strip("'")
                
                if os.path.exists(file_path):
                    file_paths.append(file_path)
                    print(f"✓ Added: {os.path.basename(file_path)}")
                else:
                    print(f"✗ File not found: {file_path}")
        
        if not file_paths:
            raise ValueError("No CSV files provided")
        
        all_data = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded {os.path.basename(file_path)}: {len(df)} stocks")
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if all_data:
            self.master_df = pd.concat(all_data, ignore_index=True)
            self.process_gics_data()
        else:
            raise ValueError("No CSV files loaded successfully")
    
    def process_gics_data(self):
        """Process and organize GICS sectors and sub-industries"""
        self.sectors = sorted(self.master_df['GICS Sector'].dropna().unique().tolist())
        self.sub_industries = sorted(self.master_df['GICS Sub-Industry'].dropna().unique().tolist())
        
    def display_selection_menu(self):
        """Display menu for sector/sub-industry selection"""
        print("\n" + "="*50)
        print("GICS SELECTION MENU")
        print("="*50)
        print("1. Select by GICS Sector")
        print("2. Select by GICS Sub-Industry")
        
        choice = input("\nEnter your choice (1 or 2): ")
        
        if choice == '1':
            print("\nAvailable GICS Sectors:")
            for i, sector in enumerate(self.sectors, 1):
                print(f"{i:2d}. {sector}")
            selection = int(input("\nSelect sector number: ")) - 1
            return 'sector', self.sectors[selection]
        else:
            print("\nAvailable GICS Sub-Industries:")
            for i, sub in enumerate(self.sub_industries, 1):
                print(f"{i:2d}. {sub}")
            selection = int(input("\nSelect sub-industry number: ")) - 1
            return 'sub_industry', self.sub_industries[selection]
    
    def analyze_market_conditions(self, sector: str) -> Dict:
        """Analyze current market and economic conditions"""
        print("\n" + "="*50)
        print("ANALYZING MARKET CONDITIONS")
        print("="*50)
        
        # Fetch economic data
        self.market_analyzer.fetch_economic_data()
        
        # Analyze sector-specific conditions
        sector_conditions = self.market_analyzer.analyze_sector_conditions(sector)
        
        # Get market adjustment factor
        self.market_adjustment = self.market_analyzer.calculate_market_adjustment_factor(sector)
        
        print(f"\nMarket Adjustment Factor: {self.market_adjustment:.2f}x")
        print(f"Sector Momentum: {sector_conditions.get('momentum', 'neutral')}")
        
        return {
            'adjustment_factor': self.market_adjustment,
            'sector_conditions': sector_conditions,
            'market_regime': self.market_analyzer.get_market_regime(),
            'fed_stance': self.market_analyzer.get_federal_reserve_stance(),
            'yield_curve': self.market_analyzer.analyze_yield_curve()
        }
    
    def get_top_10_by_market_cap(self, selection_type: str, selection_value: str):
        """Get top 10 stocks by market cap from selected sector/sub-industry"""
        if selection_type == 'sector':
            filtered_df = self.master_df[self.master_df['GICS Sector'] == selection_value]
        else:
            filtered_df = self.master_df[self.master_df['GICS Sub-Industry'] == selection_value]
        
        symbols = filtered_df['Symbol'].tolist()
        
        market_caps = {}
        print(f"\nFetching market cap data for {len(symbols)} stocks...")
        
        for i, symbol in enumerate(symbols):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    market_caps[symbol] = market_cap
                    
                if i % 10 == 0:
                    gc.collect()
                    
            except:
                continue
        
        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:10]
        self.selected_stocks = [stock[0] for stock in sorted_stocks]
        
        print(f"\nTop 10 stocks by market cap:")
        for symbol, cap in sorted_stocks:
            print(f"{symbol}: ${cap:,.0f}")
        
        return self.selected_stocks
    
    def scrape_reddit_posts(self, symbol: str, months_back: int = 6) -> List[Dict]:
        """Scrape Reddit posts mentioning the stock symbol"""
        print(f"Scraping Reddit for {symbol}...")
        
        posts = scrape_reddit_internal_api(symbol)
        
        filtered_posts = []
        cutoff_date = datetime.now() - timedelta(days=months_back * 30)
        
        for post in posts:
            post_date = datetime.fromtimestamp(post['created_utc'])
            if post_date > cutoff_date:
                filtered_posts.append(post)
        
        print(f"Found {len(filtered_posts)} posts for {symbol} in the last {months_back} months")
        return filtered_posts
    
    def analyze_reddit_sentiment(self, posts_data: List[Dict]) -> Dict:
        """Analyze sentiment from Reddit posts using TextBlob"""
        if not posts_data:
            return {
                'avg_sentiment': 0,
                'positive_ratio': 0.5,
                'weighted_sentiment': 0,
                'post_count': 0,
                'avg_engagement': 0
            }
        
        sentiments = []
        weights = []
        
        for post in posts_data[:20]:
            try:
                text = f"{post['title']} {post.get('text', '')}"
                result = self.simple_sentiment(text)[0]
                
                if result['label'] == 'positive':
                    sentiment = result['score']
                elif result['label'] == 'negative':
                    sentiment = -result['score']
                else:
                    sentiment = 0
                
                weight = np.log1p(post['score'] + post['num_comments'])
                
                sentiments.append(sentiment)
                weights.append(weight)
                
            except Exception as e:
                print(f"Error analyzing post: {e}")
                continue
        
        if not sentiments:
            return {
                'avg_sentiment': 0,
                'positive_ratio': 0.5,
                'weighted_sentiment': 0,
                'post_count': len(posts_data),
                'avg_engagement': 0
            }
        
        sentiments = np.array(sentiments)
        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        
        return {
            'avg_sentiment': np.mean(sentiments),
            'positive_ratio': (sentiments > 0).mean(),
            'weighted_sentiment': np.sum(sentiments * weights),
            'post_count': len(posts_data),
            'avg_engagement': np.mean([p['score'] + p['num_comments'] for p in posts_data])
        }
    
    def fetch_historical_data(self, symbol: str, years: int = 3) -> pd.DataFrame:
        """Fetch 3 years of historical price and financial data"""
        ticker = yf.Ticker(symbol)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        hist_prices = ticker.history(start=start_date, end=end_date)
        
        if len(hist_prices) > 756:
            hist_prices = hist_prices.resample('W').last()
        
        hist_prices['returns'] = hist_prices['Close'].pct_change()
        hist_prices['log_returns'] = np.log(hist_prices['Close'] / hist_prices['Close'].shift(1))
        hist_prices['sma_20'] = hist_prices['Close'].rolling(window=20).mean()
        hist_prices['sma_50'] = hist_prices['Close'].rolling(window=50).mean()
        hist_prices['sma_200'] = hist_prices['Close'].rolling(window=200).mean()
        hist_prices['rsi'] = self.calculate_rsi(hist_prices['Close'])
        hist_prices['volatility'] = hist_prices['returns'].rolling(window=20).std() * np.sqrt(252)
        
        exp1 = hist_prices['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist_prices['Close'].ewm(span=26, adjust=False).mean()
        hist_prices['macd'] = exp1 - exp2
        hist_prices['signal'] = hist_prices['macd'].ewm(span=9, adjust=False).mean()
        
        return hist_prices
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_closest_value(self, hist_df, date, column, default):
        """Get closest value from historical dataframe for a given date"""
        if hist_df.empty:
            return default
        try:
            if date in hist_df.index:
                return hist_df.loc[date, column]
            else:
                # Find nearest date
                nearest_idx = hist_df.index.searchsorted(date)
                if nearest_idx == 0:
                    return hist_df.iloc[0][column]
                elif nearest_idx >= len(hist_df):
                    return hist_df.iloc[-1][column]
                else:
                    # Get the closest date
                    before = hist_df.iloc[nearest_idx - 1]
                    after = hist_df.iloc[nearest_idx]
                    if abs(date - hist_df.index[nearest_idx - 1]) < abs(date - hist_df.index[nearest_idx]):
                        return before[column]
                    else:
                        return after[column]
        except:
            return default
    
    def get_spy_trend(self, spy_hist, date):
        """Get SPY trend indicator for a given date"""
        if spy_hist.empty:
            return 0
        try:
            if date in spy_hist.index:
                current_price = spy_hist.loc[date, 'Close']
                ma50 = spy_hist['Close'].rolling(50).mean().loc[date]
                return 1 if current_price > ma50 else 0
            else:
                # Find nearest date
                nearest_idx = spy_hist.index.searchsorted(date)
                if nearest_idx == 0 or nearest_idx >= len(spy_hist):
                    return 1
                # Use nearest available data
                nearest_date = spy_hist.index[min(nearest_idx, len(spy_hist)-1)]
                current_price = spy_hist.loc[nearest_date, 'Close']
                ma50 = spy_hist['Close'].rolling(50).mean().loc[nearest_date]
                return 1 if current_price > ma50 else 0
        except:
            return 0
    
    def prepare_training_data(self, symbols: List[str]) -> pd.DataFrame:
        """Prepare ML training data with 3 years of history and market indicators"""
        print("\nPreparing training data with market indicators...")
        all_data = []
        
        # Get market indicators for the period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        
        # Fetch VIX, Treasury yields, Dollar Index
        try:
            vix_hist = yf.Ticker('^VIX').history(start=start_date, end=end_date)
            tnx_hist = yf.Ticker('^TNX').history(start=start_date, end=end_date)
            dxy_hist = yf.Ticker('DX-Y.NYB').history(start=start_date, end=end_date)
            spy_hist = yf.Ticker('SPY').history(start=start_date, end=end_date)
        except:
            vix_hist = pd.DataFrame()
            tnx_hist = pd.DataFrame()
            dxy_hist = pd.DataFrame()
            spy_hist = pd.DataFrame()
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            
            try:
                hist_data = self.fetch_historical_data(symbol, years=3)
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                for i in range(len(hist_data) - 30):
                    date = hist_data.index[i]
                    
                    features = {
                        'date': date,
                        'symbol': symbol,
                        
                        # Price features
                        'price': hist_data['Close'].iloc[i],
                        'returns_1m': hist_data['returns'].iloc[max(0, i-20):i].mean() if i > 0 else 0,
                        'returns_3m': hist_data['returns'].iloc[max(0, i-60):i].mean() if i > 0 else 0,
                        'volatility': hist_data['volatility'].iloc[i] if not pd.isna(hist_data['volatility'].iloc[i]) else 0,
                        
                        # Technical indicators
                        'rsi': hist_data['rsi'].iloc[i] if not pd.isna(hist_data['rsi'].iloc[i]) else 50,
                        'macd': hist_data['macd'].iloc[i] if not pd.isna(hist_data['macd'].iloc[i]) else 0,
                        'sma_ratio_20': hist_data['Close'].iloc[i] / hist_data['sma_20'].iloc[i] if not pd.isna(hist_data['sma_20'].iloc[i]) and hist_data['sma_20'].iloc[i] > 0 else 1,
                        'sma_ratio_50': hist_data['Close'].iloc[i] / hist_data['sma_50'].iloc[i] if not pd.isna(hist_data['sma_50'].iloc[i]) and hist_data['sma_50'].iloc[i] > 0 else 1,
                        
                        # Fundamental features
                        'pe_ratio': info.get('trailingPE', 20),
                        'peg_ratio': info.get('pegRatio', 1),
                        'price_to_book': info.get('priceToBook', 1),
                        'debt_to_equity': info.get('debtToEquity', 1),
                        'roe': info.get('returnOnEquity', 0.1),
                        'profit_margin': info.get('profitMargins', 0.1),
                        'revenue_growth': info.get('revenueGrowth', 0),
                        
                        # Market cap
                        'market_cap_log': np.log(info.get('marketCap', 1e9)),
                        
                        # Market indicators - using helper functions
                        'vix': self.get_closest_value(vix_hist, date, 'Close', 20),
                        'treasury_10y': self.get_closest_value(tnx_hist, date, 'Close', 3.5),
                        'dollar_index': self.get_closest_value(dxy_hist, date, 'Close', 100),
                        'spy_trend': self.get_spy_trend(spy_hist, date),
                        
                        # Target
                        'forward_return_1m': hist_data['returns'].iloc[i:min(i+20, len(hist_data))].sum() if i < len(hist_data) - 20 else 0
                    }
                    
                    all_data.append(features)
                
                del hist_data
                gc.collect()
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        return pd.DataFrame(all_data)
    
    def add_sentiment_features(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features to training data"""
        print("\nAdding sentiment features...")
        
        # Check if training_data is empty
        if training_data.empty:
            print("Warning: Training data is empty. Skipping sentiment features.")
            return training_data
        
        # Check if 'symbol' column exists
        if 'symbol' not in training_data.columns:
            print("Warning: 'symbol' column not found in training data. Skipping sentiment features.")
            return training_data
        
        for symbol in training_data['symbol'].unique():
            print(f"Analyzing sentiment for {symbol}...")
            
            try:
                reddit_posts = self.scrape_reddit_posts(symbol, months_back=6)
                reddit_sentiment = self.analyze_reddit_sentiment(reddit_posts)
                
                ticker = yf.Ticker(symbol)
                news = ticker.news[:10] if hasattr(ticker, 'news') else []
                
                news_sentiments = []
                for article in news:
                    title = article.get('title', '')
                    if title:
                        result = self.simple_sentiment(title)[0]
                        score = 1 if result['label'] == 'positive' else (-1 if result['label'] == 'negative' else 0)
                        news_sentiments.append(score)
                
                avg_news_sentiment = np.mean(news_sentiments) if news_sentiments else 0
                
                mask = training_data['symbol'] == symbol
                training_data.loc[mask, 'reddit_sentiment'] = reddit_sentiment['weighted_sentiment']
                training_data.loc[mask, 'reddit_posts'] = reddit_sentiment['post_count']
                training_data.loc[mask, 'reddit_engagement'] = reddit_sentiment['avg_engagement']
                training_data.loc[mask, 'news_sentiment'] = avg_news_sentiment
                
            except Exception as e:
                print(f"Error adding sentiment for {symbol}: {e}")
                mask = training_data['symbol'] == symbol
                training_data.loc[mask, 'reddit_sentiment'] = 0
                training_data.loc[mask, 'reddit_posts'] = 0
                training_data.loc[mask, 'reddit_engagement'] = 0
                training_data.loc[mask, 'news_sentiment'] = 0
        
        return training_data
    
    def train_prediction_model(self, training_data: pd.DataFrame):
        """Train ML model with market-adjusted predictions"""
        print("\n" + "="*50)
        print("TRAINING MACHINE LEARNING MODEL")
        print("="*50)
        
        # Check if training data is empty
        if training_data.empty:
            print("Error: No training data available. Cannot train model.")
            return {}
        
        feature_cols = [col for col in training_data.columns 
                       if col not in ['date', 'symbol', 'forward_return_1m']]
        
        X = training_data[feature_cols].fillna(0)
        y = training_data['forward_return_1m']
        
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        X_scaled = self.scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        cv_results = {}
        
        for name, model in models.items():
            scores = []
            mse_scores = []
            
            print(f"\nTraining {name.upper()}...")
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Apply market adjustment to predictions
                if hasattr(self, 'market_adjustment'):
                    y_pred = y_pred * self.market_adjustment
                
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                scores.append(r2)
                mse_scores.append(mse)
            
            avg_r2 = np.mean(scores)
            avg_mse = np.mean(mse_scores)
            
            cv_results[name] = {
                'r2_scores': scores,
                'mse_scores': mse_scores,
                'avg_r2': avg_r2,
                'avg_mse': avg_mse
            }
            
            print(f"  Average R² Score: {avg_r2:.4f}")
            print(f"  Average MSE: {avg_mse:.6f}")
            print(f"  R² by fold: {[f'{s:.3f}' for s in scores]}")
            
            if avg_r2 > best_score:
                best_score = avg_r2
                best_model = model
                self.ml_model = model
                self.feature_cols = feature_cols
        
        print(f"\nTraining final model ({type(best_model).__name__})...")
        best_model.fit(X_scaled, y)
        
        if hasattr(best_model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importances.head(10))
        
        return cv_results
    
    def calculate_stock_predictions(self, symbol, current_data):
        """Calculate ML predictions with market adjustment"""
        try:
            # Check if ML model exists
            if not hasattr(self, 'ml_model') or self.ml_model is None:
                print(f"No ML model available for {symbol}, using default prediction")
                return 0
            
            features = {col: 0 for col in self.feature_cols}
            
            feature_mapping = {
                'returns_1m': current_data.get('recent_returns', 0),
                'volatility': current_data.get('volatility', 0.2),
                'pe_ratio': current_data.get('pe_ratio', 20),
                'market_cap_log': np.log(current_data.get('market_cap', 1e9)),
                'reddit_sentiment': current_data.get('reddit_sentiment', 0),
                'reddit_posts': current_data.get('reddit_posts', 0),
                'reddit_engagement': current_data.get('reddit_engagement', 0),
                'news_sentiment': current_data.get('news_sentiment', 0),
                'vix': current_data.get('vix', 20),
                'treasury_10y': current_data.get('treasury_10y', 3.5),
                'dollar_index': current_data.get('dollar_index', 100),
                'spy_trend': current_data.get('spy_trend', 1)
            }
            
            for key, value in feature_mapping.items():
                if key in features:
                    features[key] = value
            
            X = pd.DataFrame([features])[self.feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            predicted_return = self.ml_model.predict(X_scaled)[0]
            
            # Apply market adjustment
            if hasattr(self, 'market_adjustment'):
                predicted_return = predicted_return * self.market_adjustment
            
            return predicted_return
            
        except Exception as e:
            print(f"Error predicting for {symbol}: {e}")
            return 0
    
    def backtest_strategy(self, test_data: pd.DataFrame, initial_capital: float = 100000):
        """Backtest the strategy with market adjustments"""
        print("\n" + "="*50)
        print("BACKTESTING STRATEGY")
        print("="*50)
        
        # Check if we have the model and features
        if not hasattr(self, 'ml_model') or self.ml_model is None:
            print("No ML model available for backtesting.")
            return pd.DataFrame({'portfolio_value': [initial_capital], 'date': [datetime.now()]})
        
        if not hasattr(self, 'feature_cols') or not self.feature_cols:
            print("No feature columns defined for backtesting.")
            return pd.DataFrame({'portfolio_value': [initial_capital], 'date': [datetime.now()]})
        
        X_test = test_data[self.feature_cols].fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.ml_model.predict(X_test_scaled)
        
        # Apply market adjustment to backtest predictions
        if hasattr(self, 'market_adjustment'):
            predictions = predictions * self.market_adjustment
        
        test_data['predicted_return'] = predictions
        
        portfolio_value = initial_capital
        portfolio_history = []
        positions = {}
        
        for date in test_data['date'].unique():
            daily_data = test_data[test_data['date'] == date]
            
            top_stocks = daily_data.nlargest(3, 'predicted_return')
            
            position_size = portfolio_value / 3
            
            new_positions = {}
            for _, stock in top_stocks.iterrows():
                symbol = stock['symbol']
                price = stock['price']
                shares = position_size / price
                new_positions[symbol] = {
                    'shares': shares,
                    'entry_price': price
                }
            
            if positions:
                for symbol, pos in positions.items():
                    current_price = daily_data[daily_data['symbol'] == symbol]['price'].values
                    if len(current_price) > 0:
                        returns = (current_price[0] - pos['entry_price']) / pos['entry_price']
                        portfolio_value += pos['shares'] * pos['entry_price'] * returns
            
            positions = new_positions
            
            portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'positions': list(positions.keys())
            })
        
        portfolio_df = pd.DataFrame(portfolio_history)
        total_return = (portfolio_value - initial_capital) / initial_capital
        
        daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        rolling_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        print(f"\nBacktest Results (3 Years):")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Value: ${portfolio_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {(1 + total_return)**(1/3) - 1:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Market Adjustment Applied: {self.market_adjustment:.2f}x")
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'])
        plt.title('Portfolio Value Over Time (3-Year Backtest with Market Conditions)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.savefig('backtest_results.png')
        plt.close()
        print("\nBacktest chart saved as 'backtest_results.png'")
        
        return portfolio_df
    
    def generate_report(self, comparative_df: pd.DataFrame, 
                       reddit_sentiments: Dict, 
                       news_sentiments: List[Dict],
                       cv_results: Dict,
                       backtest_results: pd.DataFrame,
                       selection_name: str,
                       price_targets: Dict,
                       market_conditions: Dict) -> str:
        """Generate comprehensive report with market analysis"""
        
        report = f"""
# Quantitative Financial Analysis Report with Market Conditions
## {selection_name} Sector Analysis
### Date: {datetime.now().strftime('%Y-%m-%d')}

## Executive Summary
This report analyzes the top 10 companies by market capitalization in the {selection_name} sector/sub-industry 
using machine learning models trained on 3 years of historical data, fundamental analysis, sentiment analysis,
and current market/economic conditions. Price targets incorporate macroeconomic factors and sector momentum.

"""
        
        # Add market conditions summary
        report += self.market_analyzer.generate_economic_summary(selection_name)
        
        report += f"""

## Market Adjustment Applied
Based on current economic conditions and sector momentum, we've applied a 
**{(self.market_adjustment - 1) * 100:+.2f}%** adjustment to our price targets and predictions.

- Market Regime: {market_conditions['market_regime']}
- Fed Stance: {market_conditions['fed_stance']['stance']}
- Sector Outlook: {market_conditions['sector_conditions']['outlook']}

## Machine Learning Model Performance

### Cross-Validation Results:
"""
        for model_name, results in cv_results.items():
            report += f"\n**{model_name.upper()}**:\n"
            report += f"- Average R² Score: {results['avg_r2']:.4f}\n"
            report += f"- Average MSE: {results['avg_mse']:.6f}\n"
        
        report += f"""
## Backtesting Results (3-Year Period with Market Adjustments)

- **Total Return**: {((backtest_results['portfolio_value'].iloc[-1] / 100000) - 1):.2%}
- **Annualized Return**: {(((backtest_results['portfolio_value'].iloc[-1] / 100000) ** (1/3)) - 1):.2%}
- **Number of Rebalances**: {len(backtest_results)}

## Individual Stock Analysis with Market-Adjusted Price Targets

"""
        
        for _, row in comparative_df.iterrows():
            symbol = row['symbol']
            
            reddit_sent = reddit_sentiments.get(symbol, {})
            news_sent = next((n for n in news_sentiments if n['symbol'] == symbol), {})
            
            target_info = price_targets.get(symbol, {})
            
            report += f"""
### {symbol}
**Fundamental Metrics:**
- Current Price: ${row['current_price']:.2f}
- Market Cap: ${row['market_cap']:,.0f}
- P/E Ratio: {row['pe_ratio']:.2f}
- ROE: {row['roe']:.2%}
- Profit Margin: {row['profit_margin']:.2%}

**Sentiment Analysis:**
- Reddit Sentiment: {reddit_sent.get('weighted_sentiment', 0):.3f} ({reddit_sent.get('post_count', 0)} posts analyzed)
- Reddit Engagement: {reddit_sent.get('avg_engagement', 0):.0f} average interactions
- News Sentiment: {'Positive' if news_sent.get('news_sentiment', 0) > 0 else 'Negative'}
- Combined Sentiment Score: {((reddit_sent.get('weighted_sentiment', 0) + news_sent.get('news_sentiment', 0)) / 2):.3f}

**Market-Adjusted Valuation Analysis:**
- **Target Price (Market-Adjusted): ${target_info.get('target_price', row['current_price']):.2f}**
- **Upside Potential: {target_info.get('upside_potential', 0):.2%}**
- **Confidence Level: {target_info.get('confidence', 0.5):.1%}**
- **Price Range: ${target_info.get('low_target', row['current_price']*0.9):.2f} - ${target_info.get('high_target', row['current_price']*1.1):.2f}**
- **Market Adjustment Factor: {target_info.get('market_adjustment', 1.0):.2f}x**

**ML Prediction (Market-Adjusted)**: {target_info.get('ml_return', 0):.2%} expected 1-month return

**Valuation Methods Used:**
"""
            
            if 'valuations' in target_info:
                for method, value in target_info['valuations'].items():
                    if value:
                        report += f"  - {method.replace('_', ' ').title()}: ${value:.2f}\n"
            
            report += "\n"
        
        # Top 3 recommendations based on market-adjusted upside potential
        top_3_symbols = sorted(price_targets.keys(), 
                              key=lambda x: price_targets[x].get('upside_potential', 0), 
                              reverse=True)[:3]
        
        report += """
## Top 3 Investment Recommendations (Market-Adjusted)

Based on comprehensive valuation analysis, ML predictions, sentiment, and current market conditions:

"""
        
        for i, symbol in enumerate(top_3_symbols, 1):
            row = comparative_df[comparative_df['symbol'] == symbol].iloc[0]
            target_info = price_targets[symbol]
            
            # Determine market stance
            market_stance = "Favorable" if self.market_adjustment > 1.0 else ("Challenging" if self.market_adjustment < 1.0 else "Neutral")
            
            report += f"""
### {i}. {symbol}
**Investment Thesis**: 
- Target Price: ${target_info['target_price']:.2f} ({target_info['upside_potential']:.1%} upside)
- Confidence: {target_info['confidence']:.0%}
- Fundamentals: P/E of {row['pe_ratio']:.2f}, ROE of {row['roe']:.2%}
- ML Prediction: {target_info.get('ml_return', 0):.2%} expected return (market-adjusted)
- Sentiment: {('Positive' if target_info.get('sentiment_impact', 0) > 0 else 'Negative')}
- Market Environment: {market_stance} (adjustment factor: {self.market_adjustment:.2f}x)

"""
        
        report += """
## Methodology

1. **Data Collection**: 
   - 3 years of historical price data
   - Fundamental metrics from financial statements
   - Technical indicators (RSI, MACD, SMA ratios)
   - Macroeconomic indicators (VIX, Treasury yields, Dollar Index)

2. **Market Analysis**:
   - Federal Reserve policy stance assessment
   - Yield curve analysis for recession risk
   - Sector momentum vs S&P 500
   - Market regime identification

3. **Sentiment Analysis**:
   - Reddit posts from multiple subreddits using TextBlob
   - Financial news sentiment analysis
   - Weighted by engagement metrics

4. **Machine Learning**:
   - Random Forest model with time series validation
   - Features: fundamentals + technicals + sentiment + market indicators
   - Predictions adjusted by market conditions factor

5. **Valuation Methods**:
   - Enhanced DCF with dynamic assumptions
   - P/E multiple comparison
   - PEG ratio analysis
   - EV/EBITDA multiples
   - Price-to-Book valuation
   - Technical analysis targets
   - Analyst consensus
   - ML predictions integration
   - All adjusted by market conditions factor

6. **Price Target Calculation**:
   - Weighted average of all valuation methods
   - Market conditions adjustment (0.7x to 1.3x)
   - Sentiment-based adjustments
   - Confidence intervals based on valuation dispersion

## Risk Factors
"""
        
        # Add sector-specific risks based on market conditions
        if market_conditions['yield_curve']['inverted']:
            report += "- **Inverted yield curve** signals potential recession risk\n"
        
        if market_conditions['fed_stance']['stance'] == 'Tightening (Hawkish)':
            report += "- **Rising interest rates** may pressure valuations, especially for growth stocks\n"
        
        if market_conditions['sector_conditions']['momentum'] == 'negative':
            report += f"- **Sector underperformance** - {selection_name} lagging broader market\n"
        
        try:
            vix = yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1]
            if vix > 25:
                report += f"- **Elevated volatility** (VIX: {vix:.1f}) suggests market uncertainty\n"
        except:
            pass
        
        report += """

## Disclaimer
This report is for informational purposes only and should not be considered investment advice.
Past performance does not guarantee future results. Market conditions can change rapidly.
All predictions and price targets have been adjusted for current macroeconomic conditions.
"""
        
        return report
    
    def fundamental_analysis(self, financial_data: Dict) -> Dict:
        """Perform fundamental analysis on a stock"""
        info = financial_data['info']
        
        analysis = {
            'symbol': financial_data['symbol'],
            'current_price': info.get('currentPrice', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0),
            'profit_margin': info.get('profitMargins', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'dcf_value': self.calculate_dcf(financial_data),
            'market_cap': info.get('marketCap', 0),
            'dividend_yield': info.get('dividendYield', 0)
        }
        
        return analysis
    
    def calculate_dcf(self, financial_data: Dict) -> float:
        """Calculate Discounted Cash Flow valuation"""
        try:
            cash_flow = financial_data['cash_flow']
            if cash_flow.empty:
                return 0
            
            fcf = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else 0
            
            growth_rate = 0.05
            discount_rate = 0.10
            terminal_growth = 0.02
            
            projected_cf = []
            for i in range(1, 6):
                cf = fcf * (1 + growth_rate) ** i
                pv = cf / (1 + discount_rate) ** i
                projected_cf.append(pv)
            
            terminal_cf = fcf * (1 + growth_rate) ** 5 * (1 + terminal_growth)
            terminal_value = terminal_cf / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / (1 + discount_rate) ** 5
            
            enterprise_value = sum(projected_cf) + pv_terminal
            
            shares = financial_data['info'].get('sharesOutstanding', 1)
            
            fair_value = enterprise_value / shares if shares > 0 else 0
            
            return fair_value
        except:
            return 0
    
    def comparative_analysis(self, all_analyses: List[Dict]) -> pd.DataFrame:
        """Perform comparative analysis across all stocks"""
        df = pd.DataFrame(all_analyses)
        
        df['pe_vs_median'] = df['pe_ratio'] / df['pe_ratio'].median() - 1
        df['pb_vs_median'] = df['price_to_book'] / df['price_to_book'].median() - 1
        df['roe_vs_median'] = df['roe'] / df['roe'].median() - 1
        
        df['fundamental_score'] = 0
        
        df.loc[df['pe_ratio'] > 0, 'fundamental_score'] += (1 - df['pe_vs_median']).clip(-1, 1)
        df['fundamental_score'] += df['roe_vs_median'].clip(-1, 1)
        df['fundamental_score'] += df['profit_margin'] * 2
        
        df['dcf_upside'] = (df['dcf_value'] - df['current_price']) / df['current_price']
        df['fundamental_score'] += df['dcf_upside'].clip(-1, 1)
        
        return df.sort_values('fundamental_score', ascending=False)
    
    def news_sentiment_analysis(self, data: Dict) -> Dict:
        """Analyze news sentiment for a stock"""
        symbol = data['symbol']
        news = data.get('news', [])
        
        if not news:
            return {'symbol': symbol, 'news_sentiment': 0, 'news_count': 0}
        
        sentiments = []
        for article in news:
            title = article.get('title', '')
            if title:
                result = self.simple_sentiment(title)[0]
                score = 1 if result['label'] == 'positive' else (-1 if result['label'] == 'negative' else 0)
                sentiments.append(score)
        
        return {
            'symbol': symbol,
            'news_sentiment': np.mean(sentiments) if sentiments else 0,
            'news_count': len(news)
        }
    
    def run_complete_analysis(self):
        """Main method to run the complete analysis with market conditions"""
        print("=" * 50)
        print("STARTING QUANTITATIVE FINANCIAL ANALYSIS")
        print("WITH MARKET CONDITIONS INTEGRATION")
        print("=" * 50)
        
        self.load_csv_files()
        
        selection_type, selection_value = self.display_selection_menu()
        
        # Analyze market conditions for this sector
        market_conditions = self.analyze_market_conditions(selection_value)
        print(f"\nMarket Regime: {market_conditions['market_regime']}")
        print(f"Fed Stance: {market_conditions['fed_stance']['stance']}")
        print(f"Sector Outlook: {market_conditions['sector_conditions']['outlook']}")
        
        top_10 = self.get_top_10_by_market_cap(selection_type, selection_value)
        
        training_data = self.prepare_training_data(top_10)
        
        # Check if we have training data
        if training_data.empty:
            print("\nWarning: No training data collected. Using simplified analysis...")
            # Continue with fundamental analysis only
            cv_results = {}
            backtest_results = pd.DataFrame({'portfolio_value': [100000], 'date': [datetime.now()]})
        else:
            training_data = self.add_sentiment_features(training_data)
            
            train_size = int(len(training_data) * 0.7)
            train_df = training_data[:train_size]
            test_df = training_data[train_size:]
            
            if not train_df.empty:
                cv_results = self.train_prediction_model(train_df)
                if not test_df.empty and hasattr(self, 'ml_model'):
                    backtest_results = self.backtest_strategy(test_df)
                else:
                    backtest_results = pd.DataFrame({'portfolio_value': [100000], 'date': [datetime.now()]})
            else:
                cv_results = {}
                backtest_results = pd.DataFrame({'portfolio_value': [100000], 'date': [datetime.now()]})
        
        print("\nPerforming fundamental analysis...")
        all_financial_data = []
        all_analyses = []
        
        for symbol in top_10:
            print(f"Analyzing {symbol}...")
            ticker = yf.Ticker(symbol)
            
            financial_data = {
                'symbol': symbol,
                'info': ticker.info,
                'financials': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow
            }
            
            all_financial_data.append(financial_data)
            
            analysis = self.fundamental_analysis(financial_data)
            all_analyses.append(analysis)
            
            gc.collect()
        
        comparative_df = self.comparative_analysis(all_analyses)
        
        reddit_sentiments = {}
        news_sentiments = []
        
        for symbol in top_10:
            posts = self.scrape_reddit_posts(symbol)
            reddit_sentiments[symbol] = self.analyze_reddit_sentiment(posts)
            
            news_sent = self.news_sentiment_analysis({'symbol': symbol, 'news': yf.Ticker(symbol).news[:5]})
            news_sentiments.append(news_sent)
        
        print("\nCalculating market-adjusted price targets...")
        price_targets = {}
        
        # Get current market indicators
        try:
            current_vix = yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1]
            current_tnx = yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]
            current_dxy = yf.Ticker('DX-Y.NYB').history(period='1d')['Close'].iloc[-1]
            spy_trend = 1 if yf.Ticker('SPY').history(period='1d')['Close'].iloc[-1] > yf.Ticker('SPY').history(period='60d')['Close'].mean() else 0
        except:
            current_vix = 20
            current_tnx = 3.5
            current_dxy = 100
            spy_trend = 1
        
        for symbol in top_10:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1y")
                
                current_data = {
                    'recent_returns': hist['Close'].pct_change().tail(20).mean() if len(hist) > 20 else 0,
                    'volatility': hist['Close'].pct_change().std() * np.sqrt(252) if len(hist) > 2 else 0.2,
                    'pe_ratio': info.get('trailingPE', 20),
                    'market_cap': info.get('marketCap', 1e9),
                    'reddit_sentiment': reddit_sentiments[symbol].get('weighted_sentiment', 0),
                    'reddit_posts': reddit_sentiments[symbol].get('post_count', 0),
                    'reddit_engagement': reddit_sentiments[symbol].get('avg_engagement', 0),
                    'news_sentiment': next((n['news_sentiment'] for n in news_sentiments if n['symbol'] == symbol), 0),
                    'vix': current_vix,
                    'treasury_10y': current_tnx,
                    'dollar_index': current_dxy,
                    'spy_trend': spy_trend
                }
                
                ml_prediction = self.calculate_stock_predictions(symbol, current_data)
                sentiment_score = (current_data['reddit_sentiment'] + current_data['news_sentiment']) / 2
                
                valuation_result = self.valuation_model.calculate_comprehensive_valuation(
                    symbol, ml_prediction, sentiment_score, self.market_adjustment
                )
                
                valuation_result['ml_return'] = ml_prediction
                price_targets[symbol] = valuation_result
                
                print(f"{symbol}: Target ${valuation_result['target_price']:.2f} ({valuation_result['upside_potential']:.1%} upside) [Market Adj: {self.market_adjustment:.2f}x]")
                
            except Exception as e:
                print(f"Error calculating target for {symbol}: {e}")
                price_targets[symbol] = {
                    'current_price': comparative_df[comparative_df['symbol'] == symbol]['current_price'].iloc[0],
                    'target_price': comparative_df[comparative_df['symbol'] == symbol]['current_price'].iloc[0],
                    'upside_potential': 0,
                    'confidence': 0.5,
                    'low_target': comparative_df[comparative_df['symbol'] == symbol]['current_price'].iloc[0] * 0.9,
                    'high_target': comparative_df[comparative_df['symbol'] == symbol]['current_price'].iloc[0] * 1.1,
                    'ml_return': 0,
                    'valuations': {},
                    'market_adjustment': self.market_adjustment
                }
        
        print("\nGenerating report with market analysis...")
        report = self.generate_report(comparative_df, reddit_sentiments, 
                                    news_sentiments, cv_results, 
                                    backtest_results, selection_value,
                                    price_targets, market_conditions)
        
        filename = f"ml_analysis_report_{selection_value.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"\n{'='*50}")
        print(f"Analysis complete! Report saved as {filename}")
        print(f"{'='*50}")
        
        return report, comparative_df, cv_results, backtest_results

# Main execution
if __name__ == "__main__":
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     Quantitative Financial Analysis ML Model              ║
    ║     Version 3.0 - With Market Conditions Integration      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    model = QuantFinanceMLModel()
    
    try:
        report, results, cv_results, backtest = model.run_complete_analysis()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("\nTop 3 recommendations based on market-adjusted valuation:")
        print(results[['symbol', 'fundamental_score', 'dcf_upside']].head(3))
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")