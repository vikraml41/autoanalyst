#!/usr/bin/env python3
"""
FastAPI Backend - Full Professional Hedge Fund ML Model with Quant Integration
"""

import os
import sys
import glob
import logging
import json
import time
import random
import uuid
import pickle
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import hashlib
import threading
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="21.0.0")

# Thread pool for fast parallel processing
executor = ThreadPoolExecutor(max_workers=20)

# Job storage
analysis_jobs = {}
job_lock = threading.Lock()

# Enhanced caching system
cache = {}
market_cap_cache = {}
stock_info_cache = {}
financial_cache = {}
CACHE_DURATION = 300  # 5 minutes for dynamic predictions
MARKET_CAP_CACHE_DURATION = 1800
FINANCIAL_CACHE_DURATION = 600

# Market data cache
market_data_cache = None
market_data_timestamp = 0

# ML Model training state
ml_model_trained = False
training_metrics = {}
sector_models = {}
hedge_fund_model = None
last_training_time = 0
RETRAIN_INTERVAL = 3600  # Retrain every hour

# Market Cap Ranges
MARKET_CAP_RANGES = {
    'large': {'min': 10_000_000_000, 'max': float('inf'), 'label': 'Large Cap (>$10B)'},
    'mid': {'min': 2_000_000_000, 'max': 10_000_000_000, 'label': 'Mid Cap ($2B-$10B)'},
    'small': {'min': 300_000_000, 'max': 2_000_000_000, 'label': 'Small Cap ($300M-$2B)'},
    'all': {'min': 0, 'max': float('inf'), 'label': 'All Market Caps'}
}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ OPTIMIZED CACHING WITH EXPIRY ============

def get_cached(cache_dict, key, duration=CACHE_DURATION):
    """Get from cache if not expired"""
    if key in cache_dict:
        data, timestamp = cache_dict[key]
        if time.time() - timestamp < duration:
            return data
        else:
            del cache_dict[key]
    return None

def set_cached(cache_dict, key, data):
    """Set cache with timestamp"""
    cache_dict[key] = (data, time.time())

def clear_old_cache():
    """Clear expired cache entries"""
    current_time = time.time()
    for cache_dict in [cache, market_cap_cache, stock_info_cache, financial_cache]:
        expired_keys = [k for k, (_, timestamp) in cache_dict.items() 
                       if current_time - timestamp > CACHE_DURATION * 2]
        for key in expired_keys:
            del cache_dict[key]

# ============ DATA LOADING ============

def load_csv_files():
    """Load CSV files"""
    data_path = os.environ.get('DATA_PATH', '/app/data')
    if not os.path.exists(data_path):
        data_path = 'data'
    
    csv_pattern = os.path.join(data_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        # Default stocks if no CSV
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'JPM',
                      'BAC', 'WMT', 'JNJ', 'PG', 'V', 'MA', 'HD', 'DIS', 'NFLX', 'PFE'],
            'GICS Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                          'Technology', 'Technology', 'Consumer Discretionary', 'Technology', 
                          'Technology', 'Financials', 'Financials', 'Consumer Staples',
                          'Health Care', 'Consumer Staples', 'Financials', 'Financials',
                          'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary',
                          'Health Care'],
            'GICS Sub-Industry': ['Technology Hardware', 'Software', 'Interactive Media', 'Internet Retail',
                                'Semiconductors', 'Interactive Media', 'Automobiles', 'Semiconductors',
                                'Semiconductors', 'Diversified Banks', 'Diversified Banks', 'Hypermarkets',
                                'Pharmaceuticals', 'Personal Products', 'Payment Services', 'Payment Services',
                                'Home Improvement', 'Entertainment', 'Streaming Services', 'Pharmaceuticals']
        })
    
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} stocks from {csv_file}")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total stocks loaded: {len(combined)}")
        return combined
    return pd.DataFrame()

stocks_data = load_csv_files()

# ============ IMPORT QUANT MODEL ============

ML_AVAILABLE = False
ml_model = None
market_analyzer = None
valuator = None

try:
    from quant_model import (
        QuantFinanceMLModel, 
        MarketConditionsAnalyzer, 
        EnhancedValuation
    )
    
    # Initialize the quant model
    ml_model = QuantFinanceMLModel()
    ml_model.master_df = stocks_data
    ml_model.process_gics_data()
    
    # Initialize analyzers
    market_analyzer = MarketConditionsAnalyzer()
    valuator = EnhancedValuation()
    
    ML_AVAILABLE = True
    logger.info("✅ Quant Model components loaded successfully")
    
except Exception as e:
    logger.error(f"❌ Error loading quant model: {e}")
    import traceback
    logger.error(traceback.format_exc())
    ML_AVAILABLE = False

# ============ PROFESSIONAL HEDGE FUND MODEL ============

class HedgeFundAnalyst:
    """Professional-grade fundamental analysis model mimicking hedge fund strategies"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.valuation_models = {}
        self.risk_models = {}
        
    def calculate_intrinsic_value(self, ticker_data, info, financials, cash_flow, balance_sheet):
        """Calculate intrinsic value using multiple valuation methods"""
        values = []
        weights = []
        
        # 1. DCF Model (Discounted Cash Flow)
        dcf_value = self.dcf_valuation(cash_flow, info)
        if dcf_value and dcf_value > 0:
            values.append(dcf_value)
            weights.append(0.30)  # 30% weight
            
        # 2. Earnings Power Value (EPV)
        epv_value = self.earnings_power_valuation(financials, info)
        if epv_value and epv_value > 0:
            values.append(epv_value)
            weights.append(0.25)  # 25% weight
            
        # 3. Asset-Based Valuation
        asset_value = self.asset_based_valuation(balance_sheet, info)
        if asset_value and asset_value > 0:
            values.append(asset_value)
            weights.append(0.15)  # 15% weight
            
        # 4. Comparable Company Analysis
        comp_value = self.comparable_analysis(info)
        if comp_value and comp_value > 0:
            values.append(comp_value)
            weights.append(0.20)  # 20% weight
            
        # 5. Dividend Discount Model (if applicable)
        if info.get('dividendYield', 0) > 0:
            ddm_value = self.dividend_discount_model(info)
            if ddm_value and ddm_value > 0:
                values.append(ddm_value)
                weights.append(0.10)  # 10% weight
                
        if not values:
            return None
            
        # Weighted average
        total_weight = sum(weights)
        weighted_value = sum(v * w for v, w in zip(values, weights)) / total_weight
        
        return weighted_value
    
    def dcf_valuation(self, cash_flow, info):
        """Professional DCF model with multiple scenarios"""
        try:
            if cash_flow is None or cash_flow.empty:
                return None
                
            # Get Free Cash Flow
            fcf = None
            if 'Free Cash Flow' in cash_flow.index:
                fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
            elif 'Operating Cash Flow' in cash_flow.index:
                fcf = cash_flow.loc['Operating Cash Flow'].iloc[0]
                if 'Capital Expenditure' in cash_flow.index:
                    fcf -= abs(cash_flow.loc['Capital Expenditure'].iloc[0])
            
            if not fcf or fcf <= 0:
                return None
                
            # Calculate WACC (Weighted Average Cost of Capital)
            beta = info.get('beta', 1.0)
            if beta <= 0:
                beta = 1.0
                
            risk_free_rate = 0.045  # Current 10-year Treasury
            market_premium = 0.08   # Historical equity risk premium
            cost_of_equity = risk_free_rate + beta * market_premium
            
            # Get debt metrics
            total_debt = info.get('totalDebt', 0)
            market_cap = info.get('marketCap', 0)
            
            if market_cap == 0:
                return None
                
            # Calculate WACC
            tax_rate = 0.21  # Corporate tax rate
            debt_ratio = total_debt / (total_debt + market_cap) if (total_debt + market_cap) > 0 else 0
            equity_ratio = market_cap / (total_debt + market_cap) if (total_debt + market_cap) > 0 else 1
            
            # Estimate cost of debt
            if info.get('interestExpense', 0) > 0 and total_debt > 0:
                cost_of_debt = info['interestExpense'] / total_debt
            else:
                cost_of_debt = 0.04  # Default
                
            wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))
            
            # Growth rate estimation (multi-stage model)
            revenue_growth = info.get('revenueGrowth', 0.05)
            earnings_growth = info.get('earningsGrowth', 0.05)
            
            # Stage 1: High growth (years 1-5)
            growth_rate_1 = min(max(np.mean([revenue_growth, earnings_growth]), 0), 0.25)
            
            # Stage 2: Declining growth (years 6-10)
            growth_rate_2 = growth_rate_1 * 0.5
            
            # Terminal growth rate
            terminal_growth = min(0.03, growth_rate_2 * 0.5)  # Conservative terminal rate
            
            # Project cash flows
            projected_cf = []
            
            # Stage 1: Years 1-5
            for year in range(1, 6):
                cf = fcf * ((1 + growth_rate_1) ** year)
                pv = cf / ((1 + wacc) ** year)
                projected_cf.append(pv)
                
            # Stage 2: Years 6-10
            last_cf = fcf * ((1 + growth_rate_1) ** 5)
            for year in range(6, 11):
                cf = last_cf * ((1 + growth_rate_2) ** (year - 5))
                pv = cf / ((1 + wacc) ** year)
                projected_cf.append(pv)
                
            # Terminal value
            terminal_cf = last_cf * ((1 + growth_rate_2) ** 5) * (1 + terminal_growth)
            if wacc > terminal_growth:
                terminal_value = terminal_cf / (wacc - terminal_growth)
                pv_terminal = terminal_value / ((1 + wacc) ** 10)
            else:
                pv_terminal = 0
            
            # Enterprise Value
            enterprise_value = sum(projected_cf) + pv_terminal
            
            # Add cash, subtract debt
            cash = info.get('totalCash', 0)
            equity_value = enterprise_value + cash - total_debt
            
            # Per share value
            shares = info.get('sharesOutstanding', 1)
            if shares <= 0:
                return None
                
            intrinsic_value = equity_value / shares
            
            # Apply margin of safety
            margin_of_safety = 0.85  # 15% margin of safety
            
            return intrinsic_value * margin_of_safety
            
        except Exception as e:
            logger.error(f"DCF error: {e}")
            return None
    
    def earnings_power_valuation(self, financials, info):
        """EPV based on normalized earnings"""
        try:
            if financials is None or financials.empty:
                return None
                
            # Get normalized earnings (average of last 3 years if available)
            earnings = None
            if 'Net Income' in financials.index:
                earnings = financials.loc['Net Income'].mean()
            elif 'EBIT' in financials.index:
                ebit = financials.loc['EBIT'].mean()
                tax_rate = 0.21
                earnings = ebit * (1 - tax_rate)
            
            if not earnings or earnings <= 0:
                return None
                
            # Normalize for one-time items
            if 'Total Unusual Items' in financials.index:
                unusual = financials.loc['Total Unusual Items'].mean()
                earnings -= unusual
                
            # Calculate cost of capital
            beta = info.get('beta', 1.0)
            if beta <= 0:
                beta = 1.0
                
            risk_free_rate = 0.045
            market_premium = 0.08
            cost_of_equity = risk_free_rate + beta * market_premium
            
            # EPV = Normalized Earnings / Cost of Capital
            epv = earnings / cost_of_equity
            
            # Add excess cash
            cash = info.get('totalCash', 0)
            debt = info.get('totalDebt', 0)
            net_cash = cash - debt
            
            equity_value = epv + net_cash
            
            # Per share
            shares = info.get('sharesOutstanding', 1)
            if shares <= 0:
                return None
                
            return equity_value / shares
            
        except Exception as e:
            logger.error(f"EPV error: {e}")
            return None
    
    def asset_based_valuation(self, balance_sheet, info):
        """Graham-style net asset value"""
        try:
            if balance_sheet is None or balance_sheet.empty:
                return None
                
            # Get current assets
            current_assets = 0
            if 'Total Current Assets' in balance_sheet.index:
                current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
            elif 'Cash' in balance_sheet.index:
                current_assets = balance_sheet.loc['Cash'].iloc[0]
                if 'Short Term Investments' in balance_sheet.index:
                    current_assets += balance_sheet.loc['Short Term Investments'].iloc[0]
                if 'Net Receivables' in balance_sheet.index:
                    current_assets += balance_sheet.loc['Net Receivables'].iloc[0] * 0.75  # Haircut
                if 'Inventory' in balance_sheet.index:
                    current_assets += balance_sheet.loc['Inventory'].iloc[0] * 0.5  # Haircut
                    
            # Get total liabilities
            total_liabilities = 0
            if 'Total Liab' in balance_sheet.index:
                total_liabilities = balance_sheet.loc['Total Liab'].iloc[0]
            elif 'Total Current Liabilities' in balance_sheet.index:
                total_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
                if 'Long Term Debt' in balance_sheet.index:
                    total_liabilities += balance_sheet.loc['Long Term Debt'].iloc[0]
                    
            # Net Current Asset Value (NCAV)
            ncav = current_assets - total_liabilities
            
            if ncav <= 0:
                return None
                
            # Per share
            shares = info.get('sharesOutstanding', 1)
            if shares <= 0:
                return None
                
            return ncav / shares
            
        except Exception as e:
            logger.error(f"Asset valuation error: {e}")
            return None
    
    def comparable_analysis(self, info):
        """Industry multiple-based valuation"""
        try:
            current_price = info.get('currentPrice', 0)
            if current_price <= 0:
                return None
                
            # Get industry averages (simplified - in production, use sector data)
            sector = info.get('sector', 'Unknown')
            
            # Default industry multiples by sector
            sector_multiples = {
                'Technology': {'pe': 25, 'ps': 4, 'pb': 5},
                'Healthcare': {'pe': 22, 'ps': 3, 'pb': 4},
                'Financial Services': {'pe': 15, 'ps': 2, 'pb': 1.5},
                'Consumer Cyclical': {'pe': 20, 'ps': 1.5, 'pb': 3},
                'Consumer Defensive': {'pe': 18, 'ps': 1.2, 'pb': 3},
                'Industrials': {'pe': 18, 'ps': 1.5, 'pb': 2.5},
                'Energy': {'pe': 15, 'ps': 1, 'pb': 1.2},
                'Utilities': {'pe': 16, 'ps': 1.5, 'pb': 1.5},
                'Real Estate': {'pe': 20, 'ps': 3, 'pb': 1.2},
                'Basic Materials': {'pe': 15, 'ps': 1.2, 'pb': 1.8},
                'Communication Services': {'pe': 20, 'ps': 2, 'pb': 3}
            }
            
            multiples = sector_multiples.get(sector, {'pe': 18, 'ps': 2, 'pb': 2})
            
            values = []
            
            # P/E based valuation
            eps = info.get('trailingEps', 0)
            if eps > 0:
                pe_value = eps * multiples['pe']
                values.append(pe_value)
                
            # P/S based valuation
            revenue_per_share = info.get('revenuePerShare', 0)
            if revenue_per_share > 0:
                ps_value = revenue_per_share * multiples['ps']
                values.append(ps_value)
                
            # P/B based valuation
            book_value = info.get('bookValue', 0)
            if book_value > 0:
                pb_value = book_value * multiples['pb']
                values.append(pb_value)
                
            if not values:
                return None
                
            return np.mean(values)
            
        except Exception as e:
            logger.error(f"Comparable analysis error: {e}")
            return None
    
    def dividend_discount_model(self, info):
        """Gordon Growth Model for dividend stocks"""
        try:
            dividend_rate = info.get('dividendRate', 0)
            if dividend_rate <= 0:
                return None
                
            # Get dividend growth rate
            dividend_yield = info.get('dividendYield', 0)
            payout_ratio = info.get('payoutRatio', 0.5)
            
            # Estimate growth from ROE and retention
            roe = info.get('returnOnEquity', 0.10)
            retention_ratio = 1 - payout_ratio
            growth_rate = roe * retention_ratio
            
            # Cap growth rate
            growth_rate = min(growth_rate, 0.08)  # Max 8% perpetual growth
            
            # Required return (CAPM)
            beta = info.get('beta', 1.0)
            if beta <= 0:
                beta = 1.0
                
            risk_free_rate = 0.045
            market_premium = 0.08
            required_return = risk_free_rate + beta * market_premium
            
            if required_return <= growth_rate:
                return None  # Model breaks down
                
            # DDM Value
            value = dividend_rate * (1 + growth_rate) / (required_return - growth_rate)
            
            return value
            
        except Exception as e:
            logger.error(f"DDM error: {e}")
            return None
    
    def calculate_quality_score(self, info, financials, balance_sheet):
        """Calculate Piotroski F-Score + additional quality metrics"""
        score = 0
        max_score = 0
        
        # Profitability signals (4 points)
        if info.get('returnOnAssets', 0) > 0:
            score += 1
        max_score += 1
        
        if financials is not None and not financials.empty:
            if 'Operating Cash Flow' in financials.index:
                if financials.loc['Operating Cash Flow'].iloc[0] > 0:
                    score += 1
            max_score += 1
            
            # Check if OCF > Net Income (quality of earnings)
            if 'Operating Cash Flow' in financials.index and 'Net Income' in financials.index:
                if financials.loc['Operating Cash Flow'].iloc[0] > financials.loc['Net Income'].iloc[0]:
                    score += 1
            max_score += 1
        
        # Check ROA trend
        if info.get('returnOnAssets', 0) > 0.05:  # ROA > 5%
            score += 1
        max_score += 1
        
        # Leverage signals (3 points)
        current_ratio = info.get('currentRatio', 1)
        if current_ratio > 1.5:
            score += 1
        max_score += 1
        
        debt_to_equity = info.get('debtToEquity', 100)
        if debt_to_equity < 50:  # Conservative leverage
            score += 1
        max_score += 1
        
        # Operating efficiency signals (2 points)
        gross_margin = info.get('grossMargins', 0)
        if gross_margin > 0.3:  # 30% gross margin
            score += 1
        max_score += 1
        
        asset_turnover = info.get('assetTurnover', 0)
        if asset_turnover > 0.5:
            score += 1
        max_score += 1
        
        # Additional quality metrics
        if info.get('profitMargins', 0) > 0.10:  # 10% net margin
            score += 1
        max_score += 1
        
        if info.get('returnOnEquity', 0) > 0.15:  # 15% ROE
            score += 1
        max_score += 1
        
        # Normalize to 0-1 scale
        return score / max_score if max_score > 0 else 0
    
    def calculate_risk_metrics(self, hist_data, info):
        """Calculate comprehensive risk metrics"""
        risk_metrics = {}
        
        try:
            # Price volatility
            returns = hist_data['Close'].pct_change().dropna()
            risk_metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # Downside deviation
            negative_returns = returns[returns < 0]
            risk_metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            risk_metrics['max_drawdown'] = drawdown.min()
            
            # Beta
            risk_metrics['beta'] = info.get('beta', 1.0)
            
            # Value at Risk (95% confidence)
            risk_metrics['var_95'] = np.percentile(returns, 5) if len(returns) > 0 else -0.02
            
            # Sharpe ratio (assuming risk-free rate of 4.5%)
            risk_free_rate = 0.045 / 252
            excess_returns = returns - risk_free_rate
            risk_metrics['sharpe_ratio'] = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
        except Exception as e:
            logger.error(f"Risk calculation error: {e}")
            risk_metrics = {
                'volatility': 0.25,
                'downside_deviation': 0.15,
                'max_drawdown': -0.20,
                'beta': 1.0,
                'var_95': -0.02,
                'sharpe_ratio': 0.5
            }
            
        return risk_metrics

# Initialize hedge fund model
hedge_fund_model = HedgeFundAnalyst()

# ============ ENHANCED FEATURE EXTRACTION ============

def get_financial_data_with_retry(symbol, max_retries=3):
    """Get financial data with retry logic to avoid 0.00 values"""
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got real data
            if info.get('marketCap', 0) > 0:
                # Get additional financial statements
                financials = ticker.financials
                balance_sheet = ticker.balance_sheet
                cashflow = ticker.cashflow
                
                # Fix common 0.00 issues by calculating if missing
                if not info.get('trailingPE') or info.get('trailingPE') == 0:
                    if info.get('trailingEps', 0) > 0 and info.get('currentPrice', 0) > 0:
                        info['trailingPE'] = info['currentPrice'] / info['trailingEps']
                
                if not info.get('pegRatio') or info.get('pegRatio') == 0:
                    if info.get('trailingPE', 0) > 0 and info.get('earningsGrowth', 0) > 0:
                        info['pegRatio'] = info['trailingPE'] / (info['earningsGrowth'] * 100)
                
                if not info.get('priceToBook') or info.get('priceToBook') == 0:
                    if info.get('bookValue', 0) > 0 and info.get('currentPrice', 0) > 0:
                        info['priceToBook'] = info['currentPrice'] / info['bookValue']
                
                return info, financials, balance_sheet, cashflow
            
            # If no market cap, try alternative API fields
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    # Return defaults if all attempts fail
    return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def extract_comprehensive_features(symbol, hist_data, info, financials, balance_sheet, cashflow):
    """Extract comprehensive features ensuring no 0.00 values"""
    features = {}
    
    try:
        # Price momentum features (dynamic based on current data)
        close_prices = hist_data['Close']
        current_price = close_prices.iloc[-1]
        
        # Calculate various momentum indicators
        for days in [5, 10, 20, 50, 100, 200]:
            if len(close_prices) > days:
                features[f'return_{days}d'] = (current_price / close_prices.iloc[-days] - 1)
            else:
                features[f'return_{days}d'] = 0
        
        # Volatility (changes daily)
        returns = close_prices.pct_change().dropna()
        features['volatility_20d'] = returns.tail(20).std() * np.sqrt(252) if len(returns) > 20 else 0.25
        features['volatility_60d'] = returns.tail(60).std() * np.sqrt(252) if len(returns) > 60 else 0.25
        features['volatility_120d'] = returns.tail(120).std() * np.sqrt(252) if len(returns) > 120 else 0.25
        
        # Technical indicators
        if len(close_prices) >= 14:
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = float(100 - (100 / (1 + rs.iloc[-1])))
        else:
            features['rsi'] = 50
        
        # MACD
        if len(close_prices) >= 26:
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            features['macd_signal'] = float((macd - signal).iloc[-1] / current_price) if current_price != 0 else 0
            features['macd_histogram'] = float((macd - signal).iloc[-1])
        else:
            features['macd_signal'] = 0
            features['macd_histogram'] = 0
        
        # Bollinger Bands
        if len(close_prices) >= 20:
            sma_20 = close_prices.rolling(20).mean()
            std_20 = close_prices.rolling(20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            bb_width = upper_band.iloc[-1] - lower_band.iloc[-1]
            if bb_width > 0:
                features['bb_position'] = float((current_price - lower_band.iloc[-1]) / bb_width)
            else:
                features['bb_position'] = 0.5
            features['bb_width'] = float(bb_width / current_price) if current_price > 0 else 0
        else:
            features['bb_position'] = 0.5
            features['bb_width'] = 0
        
        # Moving averages
        for period in [20, 50, 100, 200]:
            if len(close_prices) >= period:
                ma = close_prices.rolling(period).mean().iloc[-1]
                features[f'price_to_ma{period}'] = current_price / ma if ma > 0 else 1
                features[f'ma{period}_slope'] = (ma - close_prices.rolling(period).mean().iloc[-period//2]) / (period//2) if len(close_prices) > period else 0
            else:
                features[f'price_to_ma{period}'] = 1
                features[f'ma{period}_slope'] = 0
        
        # Volume features
        if 'Volume' in hist_data:
            volume = hist_data['Volume']
            features['volume_ratio_20_50'] = (volume.tail(20).mean() / volume.tail(50).mean() 
                                              if len(volume) > 50 and volume.tail(50).mean() > 0 else 1)
            features['volume_ratio_5_20'] = (volume.tail(5).mean() / volume.tail(20).mean() 
                                            if len(volume) > 20 and volume.tail(20).mean() > 0 else 1)
            features['dollar_volume'] = current_price * volume.iloc[-1]
            features['avg_dollar_volume_20d'] = current_price * volume.tail(20).mean() if len(volume) > 20 else features['dollar_volume']
            features['volume_trend'] = (volume.tail(5).mean() - volume.tail(20).mean()) / volume.tail(20).mean() if len(volume) > 20 and volume.tail(20).mean() > 0 else 0
        
        # Fundamental features with defaults to avoid 0.00
        features['pe_ratio'] = info.get('trailingPE', 0) or info.get('forwardPE', 20) or 20
        features['forward_pe'] = info.get('forwardPE', features['pe_ratio'])
        features['peg_ratio'] = info.get('pegRatio', 0) or 1.5
        features['price_to_book'] = info.get('priceToBook', 0) or 2
        features['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0) or 2
        features['ev_to_ebitda'] = info.get('enterpriseToEbitda', 0) or 12
        features['ev_to_revenue'] = info.get('enterpriseToRevenue', 0) or 3
        
        # Profitability metrics
        features['gross_margin'] = info.get('grossMargins', 0) or 0.25
        features['operating_margin'] = info.get('operatingMargins', 0) or 0.10
        features['profit_margin'] = info.get('profitMargins', 0) or 0.05
        features['ebitda_margin'] = info.get('ebitdaMargins', 0) or features['operating_margin']
        features['roe'] = info.get('returnOnEquity', 0) or 0.10
        features['roa'] = info.get('returnOnAssets', 0) or 0.05
        features['roic'] = features['roe'] * 0.8  # Approximation
        
        # Growth metrics
        features['revenue_growth'] = info.get('revenueGrowth', 0) or 0.05
        features['earnings_growth'] = info.get('earningsGrowth', 0) or 0.05
        features['earnings_quarterly_growth'] = info.get('earningsQuarterlyGrowth', 0) or 0.05
        features['revenue_quarterly_growth'] = info.get('revenueQuarterlyGrowth', 0) or features['revenue_growth']
        
        # Financial health
        features['current_ratio'] = info.get('currentRatio', 0) or 1.5
        features['quick_ratio'] = info.get('quickRatio', 0) or 1.0
        features['debt_to_equity'] = info.get('debtToEquity', 0) or 50
        features['total_debt_to_capital'] = features['debt_to_equity'] / (100 + features['debt_to_equity'])
        features['interest_coverage'] = info.get('interestCoverage', 0) or 5
        features['debt_to_ebitda'] = info.get('debtToEbitda', 0) or 3
        
        # Cash flow metrics
        features['operating_cash_flow'] = info.get('operatingCashflow', 0) / info.get('marketCap', 1) if info.get('marketCap', 0) > 0 else 0
        features['free_cash_flow'] = info.get('freeCashflow', 0) / info.get('marketCap', 1) if info.get('marketCap', 0) > 0 else 0
        features['fcf_yield'] = info.get('freeCashflow', 0) / info.get('marketCap', 1) if info.get('marketCap', 0) > 0 else 0
        features['capex_to_revenue'] = abs(info.get('capitalExpenditures', 0)) / info.get('totalRevenue', 1) if info.get('totalRevenue', 0) > 0 else 0.1
        
        # Efficiency metrics
        features['asset_turnover'] = info.get('assetTurnover', 0) or 0.5
        features['inventory_turnover'] = info.get('inventoryTurnover', 0) or 5
        features['receivables_turnover'] = info.get('receivablesTurnover', 0) or 8
        features['days_sales_outstanding'] = 365 / features['receivables_turnover'] if features['receivables_turnover'] > 0 else 45
        
        # Market data
        features['market_cap'] = info.get('marketCap', 1e9)
        features['market_cap_log'] = np.log(features['market_cap']) if features['market_cap'] > 0 else 20
        features['enterprise_value'] = info.get('enterpriseValue', features['market_cap'])
        features['shares_outstanding'] = info.get('sharesOutstanding', 1e8)
        features['shares_outstanding_log'] = np.log(features['shares_outstanding']) if features['shares_outstanding'] > 0 else 18
        features['float_shares'] = info.get('floatShares', features['shares_outstanding'])
        features['float_shares_ratio'] = features['float_shares'] / features['shares_outstanding'] if features['shares_outstanding'] > 0 else 0.8
        features['shares_short'] = info.get('sharesShort', 0)
        features['short_ratio'] = info.get('shortRatio', 0) or 1
        features['short_percent_float'] = info.get('shortPercentOfFloat', 0) or 0.02
        features['beta'] = info.get('beta', 1) or 1
        
        # Analyst data
        features['recommendation_score'] = info.get('recommendationMean', 3) or 3
        features['number_of_analysts'] = info.get('numberOfAnalystOpinions', 0) or 5
        features['target_price_ratio'] = (info.get('targetMeanPrice', current_price) / current_price 
                                         if current_price > 0 else 1.1)
        features['target_high_ratio'] = (info.get('targetHighPrice', current_price * 1.2) / current_price 
                                        if current_price > 0 else 1.2)
        features['target_low_ratio'] = (info.get('targetLowPrice', current_price * 0.9) / current_price 
                                       if current_price > 0 else 0.9)
        
        # Dividend data
        features['dividend_yield'] = info.get('dividendYield', 0) or 0
        features['dividend_rate'] = info.get('dividendRate', 0) or 0
        features['payout_ratio'] = info.get('payoutRatio', 0) or 0
        features['five_year_avg_dividend_yield'] = info.get('fiveYearAvgDividendYield', features['dividend_yield'])
        
        # Ownership
        features['institutional_ownership'] = info.get('heldPercentInstitutions', 0) or 0.5
        features['insider_ownership'] = info.get('heldPercentInsiders', 0) or 0.05
        
        # Add timestamp for dynamic predictions
        features['analysis_timestamp'] = time.time()
        features['day_of_week'] = datetime.now().weekday()
        features['hour_of_day'] = datetime.now().hour
        features['month_of_year'] = datetime.now().month
        
        # Sector and industry encoding (if available)
        features['sector'] = info.get('sector', 'Unknown')
        features['industry'] = info.get('industry', 'Unknown')
        
    except Exception as e:
        logger.error(f"Feature extraction error for {symbol}: {e}")
        # Return reasonable defaults for essential features
        return {
            'return_5d': 0, 'return_10d': 0, 'return_20d': 0,
            'volatility_20d': 0.25, 'rsi': 50,
            'pe_ratio': 20, 'peg_ratio': 1.5, 'roe': 0.10,
            'revenue_growth': 0.05, 'profit_margin': 0.05,
            'current_ratio': 1.5, 'debt_to_equity': 50,
            'market_cap_log': 20, 'beta': 1,
            'analysis_timestamp': time.time()
        }
    
    return features

# ============ INTEGRATED ML TRAINING ============

def train_integrated_ml_model():
    """Train ML model integrating quant_model.py with hedge fund analysis"""
    global ml_model, ml_model_trained, training_metrics, last_training_time, hedge_fund_model
    
    if not ML_AVAILABLE or not ml_model:
        logger.error("Quant model not available for training")
        return False
    
    try:
        start_time = time.time()
        logger.info("="*50)
        logger.info("Starting Integrated ML Model Training...")
        
        # Select diverse training symbols
        training_symbols = []
        
        # Get symbols from each sector using quant model's processed data
        if hasattr(ml_model, 'sectors') and ml_model.sectors:
            for sector in ml_model.sectors[:10]:  # Top 10 sectors
                sector_stocks = stocks_data[stocks_data['GICS Sector'] == sector]['Symbol'].tolist()
                if sector_stocks:
                    training_symbols.extend(sector_stocks[:8])  # 8 stocks per sector
        else:
            # Fallback if sectors not processed
            training_symbols = stocks_data['Symbol'].tolist()[:50]
        
        if len(training_symbols) < 20:
            logger.error(f"Insufficient training symbols: {len(training_symbols)}")
            return False
        
        logger.info(f"Training with {len(training_symbols)} symbols")
        
        # Prepare training data using quant model
        training_data = ml_model.prepare_training_data(training_symbols)
        
        if training_data.empty:
            logger.error("No training data from quant model")
            # Try alternative approach
            training_data = []
            
            for symbol in training_symbols[:30]:  # Limit for speed
                try:
                    # Get comprehensive data
                    info, financials, balance_sheet, cashflow = get_financial_data_with_retry(symbol)
                    
                    if not info:
                        continue
                    
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="6mo")
                    
                    if hist.empty or len(hist) < 60:
                        continue
                    
                    # Extract features
                    features = extract_comprehensive_features(
                        symbol, hist, info, financials, balance_sheet, cashflow
                    )
                    
                    # Calculate quality score
                    quality_score = hedge_fund_model.calculate_quality_score(
                        info, financials, balance_sheet
                    )
                    features['quality_score'] = quality_score
                    
                    # Calculate risk metrics
                    risk_metrics = hedge_fund_model.calculate_risk_metrics(hist, info)
                    features.update(risk_metrics)
                    
                    # Calculate intrinsic value
                    intrinsic_value = hedge_fund_model.calculate_intrinsic_value(
                        hist, info, financials, cashflow, balance_sheet
                    )
                    
                    if intrinsic_value and intrinsic_value > 0:
                        current_price = hist['Close'].iloc[-1]
                        features['value_ratio'] = intrinsic_value / current_price
                        
                        # Calculate target (30-day forward return)
                        if len(hist) > 30:
                            future_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-30] - 1)
                            features['target'] = future_return
                            training_data.append(features)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} for training: {e}")
                    continue
            
            if training_data:
                training_data = pd.DataFrame(training_data)
            else:
                logger.error("No training data generated")
                return False
        
        logger.info(f"Prepared {len(training_data)} training samples")
        
        # Add sentiment features using quant model
        training_data = ml_model.add_sentiment_features(training_data)
        
        # Train the prediction model using quant model
        cv_results = ml_model.train_prediction_model(training_data)
        
        # Also train ensemble models if available
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import RobustScaler
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            import xgboost as xgb
            
            # Prepare features for additional model training
            feature_cols = [col for col in training_data.columns 
                          if col not in ['target', 'symbol', 'sector', 'industry', 'analysis_timestamp']]
            
            X = training_data[feature_cols].fillna(0)
            y = training_data['target'] if 'target' in training_data else training_data['recent_returns']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost model
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_score = xgb_model.score(X_test_scaled, y_test)
            
            # Store the enhanced model
            ml_model.enhanced_model = xgb_model
            ml_model.enhanced_scaler = scaler
            ml_model.enhanced_features = feature_cols
            
            logger.info(f"Enhanced XGBoost model R² score: {xgb_score:.4f}")
            
        except Exception as e:
            logger.error(f"Enhanced model training error: {e}")
        
        # Store training metrics
        training_metrics = {
            'samples': len(training_data),
            'symbols_used': len(training_symbols),
            'features': len(ml_model.feature_cols) if hasattr(ml_model, 'feature_cols') else len(training_data.columns),
            'cv_results': cv_results if cv_results else {},
            'training_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        ml_model_trained = True
        last_training_time = time.time()
        
        logger.info(f"✅ Model training completed in {time.time() - start_time:.1f} seconds")
        logger.info(f"Training metrics: {json.dumps(training_metrics, indent=2, default=str)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        ml_model_trained = False
        return False

# ============ COMPREHENSIVE STOCK ANALYSIS ============

def analyze_stock_comprehensive(symbol, market_data):
    """Comprehensive analysis using all available models"""
    global ml_model, valuator, market_analyzer, hedge_fund_model
    
    try:
        logger.info(f"Starting comprehensive analysis for {symbol}...")
        
        # Get all financial data
        info, financials, balance_sheet, cashflow = get_financial_data_with_retry(symbol)
        
        if not info:
            logger.error(f"No financial data for {symbol}")
            return None
        
        # Get historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            logger.error(f"No historical data for {symbol}")
            return None
        
        current_price = float(hist['Close'].iloc[-1])
        
        # Extract comprehensive features
        features = extract_comprehensive_features(symbol, hist, info, financials, balance_sheet, cashflow)
        
        # Calculate quality score using hedge fund model
        quality_score = hedge_fund_model.calculate_quality_score(info, financials, balance_sheet)
        
        # Calculate risk metrics
        risk_metrics = hedge_fund_model.calculate_risk_metrics(hist, info)
        
        # Calculate intrinsic value using multiple methods
        intrinsic_value = hedge_fund_model.calculate_intrinsic_value(
            hist, info, financials, cashflow, balance_sheet
        )
        
        if not intrinsic_value:
            intrinsic_value = current_price * 1.1  # Default fallback
        
        # ML Prediction - try multiple approaches
        ml_prediction = 0.05  # Default 5%
        
        if ML_AVAILABLE and ml_model and ml_model_trained:
            try:
                # Approach 1: Use enhanced model if available
                if hasattr(ml_model, 'enhanced_model') and ml_model.enhanced_model is not None:
                    feature_df = pd.DataFrame([features])[ml_model.enhanced_features].fillna(0)
                    scaled_features = ml_model.enhanced_scaler.transform(feature_df)
                    ml_prediction = float(ml_model.enhanced_model.predict(scaled_features)[0])
                    logger.info(f"{symbol}: Enhanced model prediction = {ml_prediction*100:.2f}%")
                
                # Approach 2: Use quant model's trained model
                elif hasattr(ml_model, 'ml_model') and ml_model.ml_model is not None:
                    if hasattr(ml_model, 'feature_cols'):
                        feature_df = pd.DataFrame([features])
                        for col in ml_model.feature_cols:
                            if col not in feature_df.columns:
                                feature_df[col] = 0
                        feature_df = feature_df[ml_model.feature_cols]
                    else:
                        feature_df = pd.DataFrame([features])
                    
                    if hasattr(ml_model, 'scaler') and ml_model.scaler is not None:
                        scaled_features = ml_model.scaler.transform(feature_df)
                        raw_prediction = ml_model.ml_model.predict(scaled_features)[0]
                    else:
                        raw_prediction = ml_model.ml_model.predict(feature_df)[0]
                    
                    ml_prediction = float(raw_prediction)
                    logger.info(f"{symbol}: Quant model prediction = {ml_prediction*100:.2f}%")
                
                # Market regime adjustment
                if 'Bear' in market_data.get('market_regime', ''):
                    ml_prediction *= 0.85
                elif 'Bull' in market_data.get('market_regime', ''):
                    ml_prediction *= 1.15
                
                # VIX adjustment
                vix = market_data.get('vix', 20)
                if vix > 30:
                    ml_prediction *= 0.9
                elif vix < 15:
                    ml_prediction *= 1.1
                
                # Add dynamic variation
                market_noise = (random.random() - 0.5) * 0.01
                time_factor = np.sin(time.time() / 3600) * 0.005
                ml_prediction = ml_prediction + market_noise + time_factor
                
                # Ensure reasonable bounds
                ml_prediction = max(-0.30, min(0.50, ml_prediction))
                
            except Exception as e:
                logger.error(f"ML prediction error for {symbol}: {e}")
                # Fallback calculation
                ml_prediction = 0.05 + features.get('return_20d', 0) * 0.3 - features.get('volatility_20d', 0.25) * 0.1
        
        # Use Enhanced Valuation from quant model
        target_price = intrinsic_value
        
        if ML_AVAILABLE and valuator:
            try:
                # Calculate sentiment
                sentiment_score = 0
                if info.get('recommendationMean', 3) < 2.5:
                    sentiment_score = 1
                elif info.get('recommendationMean', 3) > 3.5:
                    sentiment_score = -1
                
                # Get market adjustment
                market_adjustment = 1.0
                if 'Bear' in market_data.get('market_regime', ''):
                    market_adjustment = 0.9
                elif 'Bull' in market_data.get('market_regime', ''):
                    market_adjustment = 1.1
                
                # Use valuator's comprehensive valuation
                valuation_result = valuator.calculate_comprehensive_valuation(
                    symbol,
                    ml_prediction,
                    sentiment_score,
                    market_adjustment
                )
                
                if valuation_result and 'target_price' in valuation_result:
                    target_price = valuation_result['target_price']
                    
            except Exception as e:
                logger.error(f"Valuation error for {symbol}: {e}")
        
        # Combine intrinsic value and ML-based target
        final_target = (intrinsic_value * 0.4 + target_price * 0.4 + current_price * (1 + ml_prediction) * 0.2)
        
        # Calculate confidence factors
        confidence_factors = []
        confidence_score = quality_score  # Start with quality score
        
        # Value factor
        value_discount = (intrinsic_value - current_price) / current_price
        if value_discount > 0.20:
            confidence_factors.append(f"Undervalued by {value_discount*100:.1f}%")
            confidence_score = min(confidence_score + 0.1, 1.0)
        
        # Quality factors
        if features.get('pe_ratio', 999) < 20 and features.get('pe_ratio', 0) > 0:
            confidence_factors.append(f"Attractive P/E: {features['pe_ratio']:.1f}")
        
        if features.get('roe', 0) > 0.15:
            confidence_factors.append(f"Strong ROE: {features['roe']*100:.1f}%")
        
        if features.get('revenue_growth', 0) > 0.10:
            confidence_factors.append(f"Good growth: {features['revenue_growth']*100:.1f}%")
        
        if risk_metrics['sharpe_ratio'] > 1:
            confidence_factors.append(f"Good risk-adjusted returns")
        
        if features.get('current_ratio', 0) > 2 and features.get('debt_to_equity', 999) < 50:
            confidence_factors.append("Strong balance sheet")
        
        return {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'intrinsic_value': intrinsic_value,
            'target_price': final_target,
            'ml_prediction': ml_prediction,
            'quality_score': quality_score,
            'confidence_score': confidence_score,
            'confidence_factors': confidence_factors,
            'market_cap': features.get('market_cap', 0),
            'fundamentals': {
                'pe_ratio': features.get('pe_ratio', 20),
                'peg_ratio': features.get('peg_ratio', 1.5),
                'roe': features.get('roe', 0.10),
                'profit_margin': features.get('profit_margin', 0.05),
                'revenue_growth': features.get('revenue_growth', 0.05),
                'debt_to_equity': features.get('debt_to_equity', 50),
                'current_ratio': features.get('current_ratio', 1.5),
                'price_to_book': features.get('price_to_book', 2),
                'ev_to_ebitda': features.get('ev_to_ebitda', 12),
                'free_cash_flow_yield': features.get('fcf_yield', 0.05)
            },
            'technicals': {
                'volatility': features.get('volatility_20d', 0.25),
                'momentum_20d': features.get('return_20d', 0),
                'rsi': features.get('rsi', 50),
                'macd_signal': features.get('macd_signal', 0),
                'bb_position': features.get('bb_position', 0.5)
            },
            'risk_metrics': risk_metrics,
            'all_features': features  # For debugging
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis error for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ============ BATCH ANALYSIS ============

def analyze_stocks_batch(symbols, market_data):
    """Analyze multiple stocks in parallel with comprehensive analysis"""
    results = []
    
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for symbol in symbols:
            future = pool.submit(analyze_stock_comprehensive, symbol, market_data)
            futures.append((symbol, future))
        
        for symbol, future in futures:
            try:
                result = future.result(timeout=30)
                if result:
                    # Calculate upside
                    upside = ((result['target_price'] / result['current_price']) - 1) * 100
                    
                    # Combined score
                    combined_score = (
                        upside * 0.35 +
                        result['quality_score'] * 100 * 0.25 +
                        result['confidence_score'] * 100 * 0.20 +
                        result['ml_prediction'] * 100 * 0.20
                    )
                    
                    result['upside'] = upside
                    result['combined_score'] = combined_score
                    
                    # Include all stocks with reasonable upside
                    if upside > -10:  # Even include slightly negative if high quality
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Error in batch analysis for {symbol}: {e}")
                continue
    
    return results

# ============ MARKET DATA ============

def fetch_vix():
    """Fetch VIX with caching"""
    cached_vix = get_cached(cache, 'vix_value', 300)
    if cached_vix:
        return cached_vix
    
    try:
        vix_data = yf.download("^VIX", period="1d", progress=False, timeout=10)
        if not vix_data.empty:
            vix_value = float(vix_data['Close'].iloc[-1])
            set_cached(cache, 'vix_value', vix_value)
            return vix_value
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")
    
    return 20.0 + random.random() * 2  # Default with variation

def get_market_data():
    """Get comprehensive market data"""
    global market_data_cache, market_data_timestamp
    
    if market_data_cache and (time.time() - market_data_timestamp) < 300:
        return market_data_cache
    
    market_data = {
        'vix': fetch_vix(),
        'spy_price': 500,
        'spy_trend': 1,
        'treasury_10y': 4.3,
        'dollar_index': 105,
        'market_regime': 'Neutral',
        'sector_rotation': {},
        'risk_metrics': {},
        'market_adjustment': 1.0
    }
    
    try:
        # Get SPY data
        spy_data = yf.download("SPY", period="5d", progress=False, timeout=10)
        if not spy_data.empty:
            market_data['spy_price'] = float(spy_data['Close'].iloc[-1])
            market_data['spy_trend'] = float(spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0])
    except Exception as e:
        logger.error(f"Error fetching SPY: {e}")
    
    # Use MarketConditionsAnalyzer if available
    if ML_AVAILABLE and market_analyzer:
        try:
            market_data['market_regime'] = market_analyzer.get_market_regime()
            market_data['sector_rotation'] = market_analyzer.analyze_sector_rotation()
            market_data['risk_metrics'] = market_analyzer.calculate_risk_metrics()
            
            # Calculate market adjustment
            if 'Bull' in market_data['market_regime']:
                market_data['market_adjustment'] = 1.1
            elif 'Bear' in market_data['market_regime']:
                market_data['market_adjustment'] = 0.9
            else:
                market_data['market_adjustment'] = 1.0
                
        except Exception as e:
            logger.error(f"Error getting market conditions: {e}")
    
    market_data_cache = market_data
    market_data_timestamp = time.time()
    
    return market_data

# ============ MARKET CAP FILTERING ============

def get_market_caps_batch(symbols):
    """Get market caps for multiple symbols efficiently"""
    result = {}
    
    for symbol in symbols[:50]:  # Limit to avoid too many requests
        try:
            cached = get_cached(market_cap_cache, symbol, MARKET_CAP_CACHE_DURATION)
            if cached is not None:
                result[symbol] = cached
            else:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    result[symbol] = market_cap
                    set_cached(market_cap_cache, symbol, market_cap)
        except Exception as e:
            logger.error(f"Error getting market cap for {symbol}: {e}")
            continue
    
    return result

# ============ BACKGROUND ANALYSIS JOB ============

def run_analysis_background(job_id, request_data):
    """Background analysis job with comprehensive analysis"""
    try:
        start_time = time.time()
        logger.info(f"Starting analysis job {job_id}")
        
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'processing',
                'progress': 'Initializing comprehensive analysis...',
                'started': start_time
            }
        
        analysis_type = request_data['analysis_type']
        target = request_data['target']
        market_cap_size = request_data.get('market_cap_size', 'all')
        
        logger.info(f"Analysis: type={analysis_type}, target={target}, cap={market_cap_size}")
        
        # Clear old cache for fresh analysis
        clear_old_cache()
        
        # Get market data
        with job_lock:
            analysis_jobs[job_id]['progress'] = 'Analyzing market conditions...'
        
        market_data = get_market_data()
        logger.info(f"Market: VIX={market_data['vix']:.2f}, Regime={market_data['market_regime']}")
        
        # Filter stocks
        with job_lock:
            analysis_jobs[job_id]['progress'] = 'Filtering stocks...'
        
        if analysis_type == "sector":
            filtered_stocks = stocks_data[stocks_data['GICS Sector'] == target]
        else:
            filtered_stocks = stocks_data[stocks_data['GICS Sub-Industry'] == target]
        
        if len(filtered_stocks) == 0:
            logger.error(f"No stocks found for {target}")
            with job_lock:
                analysis_jobs[job_id] = {
                    'status': 'completed',
                    'results': {
                        'top_stocks': [],
                        'market_conditions': {
                            'vix': market_data['vix'],
                            'regime': market_data['market_regime'],
                            'ml_model_trained': ml_model_trained
                        },
                        'total_analyzed': 0,
                        'analysis_time': time.time() - start_time
                    }
                }
            return
        
        all_symbols = filtered_stocks['Symbol'].tolist()
        logger.info(f"Found {len(all_symbols)} stocks in {target}")
        
        # Market cap filtering
        if market_cap_size != 'all':
            with job_lock:
                analysis_jobs[job_id]['progress'] = 'Filtering by market cap...'
            
            market_caps = get_market_caps_batch(all_symbols)
            cap_range = MARKET_CAP_RANGES[market_cap_size]
            
            filtered = [(s, c) for s, c in market_caps.items() 
                       if cap_range['min'] <= c < cap_range['max']]
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            analysis_symbols = [s[0] for s in filtered[:20]]
            logger.info(f"Filtered to {len(analysis_symbols)} stocks by market cap")
        else:
            analysis_symbols = all_symbols[:20]
        
        if not analysis_symbols:
            analysis_symbols = all_symbols[:15]
        
        # Perform comprehensive analysis
        with job_lock:
            analysis_jobs[job_id]['progress'] = f'Performing hedge fund analysis on {len(analysis_symbols)} stocks...'
        
        logger.info(f"Analyzing: {', '.join(analysis_symbols)}")
        results = analyze_stocks_batch(analysis_symbols, market_data)
        
        logger.info(f"Analysis returned {len(results)} results")
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Get top picks with different criteria
        top_picks = []
        
        # First, get high upside stocks
        for stock in results:
            if stock['upside'] > 10 and stock['quality_score'] > 0.5:
                top_picks.append(stock)
                if len(top_picks) >= 3:
                    break
        
        # If not enough, add high quality stocks
        if len(top_picks) < 3:
            for stock in results:
                if stock not in top_picks and stock['quality_score'] > 0.6:
                    top_picks.append(stock)
                    if len(top_picks) >= 3:
                        break
        
        # If still not enough, add best remaining
        if len(top_picks) < 3:
            for stock in results:
                if stock not in top_picks:
                    top_picks.append(stock)
                    if len(top_picks) >= 3:
                        break
        
        # Format results for frontend
        formatted = []
        for stock in top_picks:
            formatted.append({
                "symbol": stock['symbol'],
                "company_name": stock['company_name'],
                "market_cap": f"${stock['market_cap']/1e9:.2f}B" if stock['market_cap'] > 1e9 else f"${stock['market_cap']/1e6:.0f}M",
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside'], 1),
                    "confidence_score": int(stock['confidence_score'] * 100),
                    "ml_score": round(stock['ml_prediction'], 4)
                },
                "analysis_details": {
                    "fundamentals": stock['fundamentals'],
                    "technicals": stock['technicals'],
                    "ml_prediction": stock['ml_prediction'],
                    "quality_score": stock['quality_score'],
                    "risk_metrics": stock.get('risk_metrics', {})
                },
                "investment_thesis": stock['confidence_factors']
            })
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Analysis completed in {elapsed:.1f}s with {len(formatted)} picks")
        
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'completed',
                'results': {
                    'top_stocks': formatted,
                    'market_conditions': {
                        'vix': market_data['vix'],
                        'regime': market_data['market_regime'],
                        'ml_model_trained': ml_model_trained,
                        'training_metrics': training_metrics if ml_model_trained else {}
                    },
                    'total_analyzed': len(results),
                    'analysis_time': elapsed
                }
            }
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'error',
                'error': str(e)
            }

# ============ API ENDPOINTS (REST OF CODE) ============

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str
    market_cap_size: str = 'all'

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis job"""
    
    # Check if model needs training
    if not ml_model_trained or (time.time() - last_training_time > RETRAIN_INTERVAL):
        logger.info("Model needs training")
        background_tasks.add_task(train_integrated_ml_model)
    
    job_id = str(uuid.uuid4())
    logger.info(f"Starting job {job_id} for {request.analysis_type}: {request.target}")
    background_tasks.add_task(run_analysis_background, job_id, request.dict())
    return JSONResponse(content={"job_id": job_id, "status": "started"})

@app.get("/api/analysis/{job_id}")
async def get_analysis_status(job_id: str):
    """Get analysis job status"""
    with job_lock:
        if job_id not in analysis_jobs:
            return JSONResponse(content={"status": "not_found"}, status_code=404)
        
        job = analysis_jobs[job_id]
        if job['status'] == 'completed':
            result = {"status": "completed", "results": job['results'], "ml_powered": ML_AVAILABLE}
            del analysis_jobs[job_id]
            return JSONResponse(content=result)
        elif job['status'] == 'error':
            result = {"status": "error", "error": job.get('error')}
            del analysis_jobs[job_id]
            return JSONResponse(content=result, status_code=500)
        else:
            return JSONResponse(content={"status": "processing", "progress": job.get('progress')})

@app.get("/api/market-conditions")
async def get_market_conditions():
    """Get current market conditions"""
    market_data = get_market_data()
    vix = market_data['vix']
    
    # Determine regime
    regime = market_data.get('market_regime', 'Normal')
    
    # Calculate recession risk
    recession_risk = "Unknown"
    try:
        ten_year_data = yf.download("^TNX", period="1d", progress=False, timeout=5)
        two_year_data = yf.download("^FVX", period="1d", progress=False, timeout=5)
        
        if not ten_year_data.empty and not two_year_data.empty:
            ten_year = float(ten_year_data['Close'].iloc[-1])
            two_year = float(two_year_data['Close'].iloc[-1])
            yield_spread = ten_year - two_year
            
            if yield_spread < -0.5:
                recession_risk = "High"
            elif yield_spread < 0:
                recession_risk = "Medium-High"
            elif yield_spread < 0.5:
                recession_risk = "Medium"
            else:
                recession_risk = "Low"
        else:
            if vix > 30:
                recession_risk = "High"
            elif vix > 25:
                recession_risk = "Medium"
            else:
                recession_risk = "Low"
                
    except Exception as e:
        logger.error(f"Error calculating recession risk: {e}")
        if vix > 30:
            recession_risk = "High"
        elif vix > 25:
            recession_risk = "Medium"
        else:
            recession_risk = "Low"
    
    # Determine Fed stance
    fed_stance = "Neutral"
    try:
        fed_data = yf.download("^IRX", period="3mo", progress=False, timeout=5)
        if not fed_data.empty and len(fed_data) > 20:
            recent = float(fed_data['Close'].iloc[-1])
            past = float(fed_data['Close'].iloc[-20])
            
            if recent > past + 0.25:
                fed_stance = "Hawkish"
            elif recent < past - 0.25:
                fed_stance = "Dovish"
    except:
        pass
    
    return JSONResponse(content={
        "regime": regime,
        "vix": vix,
        "fed_stance": fed_stance,
        "recession_risk": recession_risk,
        "ml_trained": ml_model_trained,
        "training_metrics": training_metrics if ml_model_trained else {}
    })

@app.get("/api/stocks/list")
async def get_stocks_list():
    """Get list of sectors and sub-industries"""
    sectors = stocks_data['GICS Sector'].dropna().unique().tolist() if 'GICS Sector' in stocks_data.columns else []
    sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist() if 'GICS Sub-Industry' in stocks_data.columns else []
    total_stocks = len(stocks_data)
    
    return JSONResponse(content={
        "sectors": sectors,
        "sub_industries": sub_industries,
        "total_stocks": total_stocks,
        "ml_status": {
            "trained": ml_model_trained,
            "training_metrics": training_metrics
        }
    })

@app.get("/api/ml-status")
async def get_ml_status():
    """Get detailed ML model status"""
    status = {
        "ml_available": ML_AVAILABLE,
        "ml_model_trained": ml_model_trained,
        "training_metrics": training_metrics,
        "quant_model_loaded": ml_model is not None,
        "market_analyzer_loaded": market_analyzer is not None,
        "valuator_loaded": valuator is not None,
        "hedge_fund_model_loaded": hedge_fund_model is not None,
        "has_enhanced_model": hasattr(ml_model, 'enhanced_model') if ml_model else False,
        "last_training": datetime.fromtimestamp(last_training_time).isoformat() if last_training_time > 0 else None,
        "next_training": datetime.fromtimestamp(last_training_time + RETRAIN_INTERVAL).isoformat() if last_training_time > 0 else None,
        "cache_entries": {
            "general": len(cache),
            "market_cap": len(market_cap_cache),
            "stock_info": len(stock_info_cache),
            "financial": len(financial_cache)
        }
    }
    return JSONResponse(content=status)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "ml_available": ML_AVAILABLE,
        "ml_trained": ml_model_trained,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(content={
        "name": "AutoAnalyst",
        "version": "21.0",
        "ml_enabled": ML_AVAILABLE,
        "ml_trained": ml_model_trained,
        "training_metrics": training_metrics,
        "components": {
            "quant_model": ml_model is not None,
            "market_analyzer": market_analyzer is not None,
            "valuator": valuator is not None,
            "hedge_fund_model": hedge_fund_model is not None
        }
    })

# ============ STARTUP EVENTS ============

@app.on_event("startup")
async def startup_event():
    """Initialize and train model on startup"""
    logger.info("="*50)
    logger.info("Starting AutoAnalyst API v21.0...")
    logger.info(f"ML Available: {ML_AVAILABLE}")
    
    if ML_AVAILABLE:
        logger.info("Components loaded:")
        logger.info(f"  - QuantFinanceMLModel: {ml_model is not None}")
        logger.info(f"  - MarketConditionsAnalyzer: {market_analyzer is not None}")
        logger.info(f"  - EnhancedValuation: {valuator is not None}")
        logger.info(f"  - HedgeFundAnalyst: {hedge_fund_model is not None}")
    
    # Load market data
    get_market_data()
    
    # Start model training in background
    if ML_AVAILABLE:
        executor.submit(train_integrated_ml_model)
        logger.info("Integrated model training started in background...")
    
    logger.info("API ready!")
    logger.info("="*50)

async def periodic_retrain():
    """Retrain model periodically for fresh predictions"""
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL)
        if ML_AVAILABLE and ml_model:
            logger.info("Starting periodic model retraining...")
            executor.submit(train_integrated_ml_model)

async def periodic_cache_cleanup():
    """Clean up cache periodically"""
    while True:
        await asyncio.sleep(600)
        clear_old_cache()
        total_cache = len(cache) + len(market_cap_cache) + len(stock_info_cache) + len(financial_cache)
        logger.info(f"Cache cleanup completed. Total entries: {total_cache}")

@app.on_event("startup")
async def start_periodic_tasks():
    """Start background tasks"""
    asyncio.create_task(periodic_retrain())
    asyncio.create_task(periodic_cache_cleanup())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)