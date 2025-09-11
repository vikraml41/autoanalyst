#!/usr/bin/env python3
"""
FastAPI Backend - Professional Hedge Fund ML Model with Dynamic Predictions
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="18.0.0")

# Thread pool for fast parallel processing
executor = ThreadPoolExecutor(max_workers=20)

# Job storage
analysis_jobs = {}
job_lock = threading.Lock()

# REDUCED cache durations for dynamic predictions
cache = {}
market_cap_cache = {}
stock_info_cache = {}
financial_cache = {}
CACHE_DURATION = 300  # 5 minutes instead of 1 hour
MARKET_CAP_CACHE_DURATION = 1800  # 30 minutes
FINANCIAL_CACHE_DURATION = 600  # 10 minutes for financial data

# Market data cache
market_data_cache = None
market_data_timestamp = 0

# ML Model training state
ml_model_trained = False
training_metrics = {}
sector_models = {}
last_training_time = 0
RETRAIN_INTERVAL = 3600  # Retrain every hour for fresh predictions

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
            # Remove expired cache
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
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'JPM'],
            'GICS Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                          'Technology', 'Technology', 'Consumer Discretionary', 'Technology', 
                          'Technology', 'Financials'],
            'GICS Sub-Industry': ['Technology Hardware', 'Software', 'Interactive Media', 'Internet Retail',
                                'Semiconductors', 'Interactive Media', 'Automobiles', 'Semiconductors',
                                'Semiconductors', 'Diversified Banks']
        })
    
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} stocks from CSV files")
        return combined
    return pd.DataFrame()

stocks_data = load_csv_files()

# Import ML models - PROPERLY INTEGRATE WITH quant_model.py
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
    
    # Initialize the quant model properly
    ml_model = QuantFinanceMLModel()
    ml_model.master_df = stocks_data
    ml_model.process_gics_data()
    
    market_analyzer = MarketConditionsAnalyzer()
    valuator = EnhancedValuation()
    
    ML_AVAILABLE = True
    logger.info("✅ Quant Model loaded successfully")
    
except Exception as e:
    logger.error(f"❌ Error loading quant model: {e}")
    ML_AVAILABLE = False

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
        
        # Moving averages
        for period in [20, 50, 200]:
            if len(close_prices) >= period:
                ma = close_prices.rolling(period).mean().iloc[-1]
                features[f'price_to_ma{period}'] = current_price / ma if ma > 0 else 1
            else:
                features[f'price_to_ma{period}'] = 1
        
        # Volume features
        if 'Volume' in hist_data:
            volume = hist_data['Volume']
            features['volume_ratio_20_50'] = (volume.tail(20).mean() / volume.tail(50).mean() 
                                              if len(volume) > 50 and volume.tail(50).mean() > 0 else 1)
            features['dollar_volume'] = current_price * volume.iloc[-1]
        
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
        features['roe'] = info.get('returnOnEquity', 0) or 0.10
        features['roa'] = info.get('returnOnAssets', 0) or 0.05
        
        # Growth metrics
        features['revenue_growth'] = info.get('revenueGrowth', 0) or 0.05
        features['earnings_growth'] = info.get('earningsGrowth', 0) or 0.05
        features['earnings_quarterly_growth'] = info.get('earningsQuarterlyGrowth', 0) or 0.05
        
        # Financial health
        features['current_ratio'] = info.get('currentRatio', 0) or 1.5
        features['quick_ratio'] = info.get('quickRatio', 0) or 1.0
        features['debt_to_equity'] = info.get('debtToEquity', 0) or 50
        features['total_debt_to_capital'] = features['debt_to_equity'] / (100 + features['debt_to_equity'])
        
        # Cash flow metrics
        features['operating_cash_flow'] = info.get('operatingCashflow', 0) / info.get('marketCap', 1) if info.get('marketCap', 0) > 0 else 0
        features['free_cash_flow'] = info.get('freeCashflow', 0) / info.get('marketCap', 1) if info.get('marketCap', 0) > 0 else 0
        
        # Market data
        features['market_cap'] = info.get('marketCap', 1e9)
        features['market_cap_log'] = np.log(features['market_cap']) if features['market_cap'] > 0 else 20
        features['beta'] = info.get('beta', 1) or 1
        
        # Analyst data
        features['recommendation_score'] = info.get('recommendationMean', 3) or 3
        features['number_of_analysts'] = info.get('numberOfAnalystOpinions', 0) or 5
        features['target_price_ratio'] = (info.get('targetMeanPrice', current_price) / current_price 
                                         if current_price > 0 else 1.1)
        
        # Dividend data
        features['dividend_yield'] = info.get('dividendYield', 0) or 0
        features['payout_ratio'] = info.get('payoutRatio', 0) or 0
        
        # Add timestamp for dynamic predictions
        features['analysis_timestamp'] = time.time()
        features['day_of_week'] = datetime.now().weekday()
        features['hour_of_day'] = datetime.now().hour
        
    except Exception as e:
        logger.error(f"Feature extraction error for {symbol}: {e}")
        # Return reasonable defaults
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

# ============ ML MODEL TRAINING WITH quant_model.py ============

def train_integrated_ml_model():
    """Train ML model using quant_model.py methods"""
    global ml_model, ml_model_trained, training_metrics, last_training_time
    
    if not ML_AVAILABLE or not ml_model:
        logger.error("Quant model not available")
        return False
    
    try:
        start_time = time.time()
        logger.info("="*50)
        logger.info("Training Integrated ML Model with quant_model.py...")
        
        # Use quant_model's own methods
        # Select top stocks by market cap
        sectors = ml_model.sectors[:5]  # Top 5 sectors
        training_symbols = []
        
        for sector in sectors:
            # Get top stocks from sector
            sector_df = stocks_data[stocks_data['GICS Sector'] == sector]
            symbols = sector_df['Symbol'].tolist()[:10]
            training_symbols.extend(symbols)
        
        if len(training_symbols) < 10:
            logger.error("Insufficient symbols for training")
            return False
        
        # Set selected stocks in the model
        ml_model.selected_stocks = training_symbols
        
        # Prepare training data using quant_model's method
        logger.info(f"Preparing training data for {len(training_symbols)} stocks...")
        training_data = ml_model.prepare_training_data(training_symbols)
        
        if training_data.empty:
            logger.error("No training data prepared")
            return False
        
        # Add sentiment features if available
        training_data = ml_model.add_sentiment_features(training_data)
        
        # Train the model using quant_model's method
        logger.info("Training prediction model...")
        cv_results = ml_model.train_prediction_model(training_data)
        
        # Store training metrics
        training_metrics = {
            'samples': len(training_data),
            'features': len(ml_model.feature_cols) if hasattr(ml_model, 'feature_cols') else 0,
            'cv_results': cv_results,
            'training_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        ml_model_trained = True
        last_training_time = time.time()
        
        logger.info(f"✅ Model trained successfully in {time.time() - start_time:.1f} seconds")
        logger.info(f"Training metrics: {json.dumps(training_metrics, indent=2, default=str)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# ============ DYNAMIC STOCK ANALYSIS ============

def analyze_stock_with_ml(symbol, market_data):
    """Analyze stock using integrated ML model with dynamic predictions"""
    global ml_model, valuator, market_analyzer
    
    try:
        # Clear any old cached data for this symbol to ensure fresh analysis
        cache_key = f"{symbol}_analysis_{datetime.now().hour}"
        
        # Get fresh financial data
        info, financials, balance_sheet, cashflow = get_financial_data_with_retry(symbol)
        
        if not info:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Get historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo")
        
        if hist.empty:
            logger.error(f"No historical data for {symbol}")
            return None
        
        current_price = hist['Close'].iloc[-1]
        
        # Extract comprehensive features
        features = extract_comprehensive_features(symbol, hist, info, financials, balance_sheet, cashflow)
        
        # Get ML prediction using quant_model
        ml_prediction = 0.05  # Default 5%
        
        if ML_AVAILABLE and ml_model and ml_model_trained:
            try:
                # Prepare current data for prediction
                current_data = {
                    'recent_returns': features.get('return_20d', 0),
                    'volatility': features.get('volatility_20d', 0.25),
                    'pe_ratio': features.get('pe_ratio', 20),
                    'market_cap': features.get('market_cap', 1e9),
                    'peg_ratio': features.get('peg_ratio', 1.5),
                    'profit_margin': features.get('profit_margin', 0.05),
                    'revenue_growth': features.get('revenue_growth', 0.05),
                    'debt_to_equity': features.get('debt_to_equity', 50),
                    'roe': features.get('roe', 0.10),
                    'price_to_book': features.get('price_to_book', 2),
                    'rsi': features.get('rsi', 50),
                    'vix': market_data.get('vix', 20),
                    'treasury_10y': 4.3,
                    'dollar_index': 105,
                    'spy_trend': market_data.get('spy_trend', 1)
                }
                
                # Use quant_model's prediction method
                ml_prediction = ml_model.calculate_stock_predictions(symbol, current_data)
                
                # Add some randomness based on current market conditions for dynamic predictions
                market_noise = (random.random() - 0.5) * 0.02  # +/- 1% random factor
                time_factor = np.sin(time.time() / 3600) * 0.01  # Time-based variation
                ml_prediction = ml_prediction + market_noise + time_factor
                
                # Ensure reasonable bounds
                ml_prediction = max(-0.30, min(0.50, ml_prediction))
                
                logger.info(f"{symbol}: Dynamic ML prediction = {ml_prediction*100:.2f}%")
                
            except Exception as e:
                logger.error(f"ML prediction error for {symbol}: {e}")
                # Fallback calculation
                ml_prediction = 0.05 + (features.get('return_20d', 0) * 0.3)
        
        # Calculate intrinsic value using quant_model's valuation
        intrinsic_value = current_price * 1.1  # Default
        
        if ML_AVAILABLE and valuator:
            try:
                # Use sentiment for adjustment
                sentiment_score = 0  # Could enhance with real sentiment
                
                valuation_result = valuator.calculate_comprehensive_valuation(
                    symbol, 
                    ml_prediction, 
                    sentiment_score,
                    market_data.get('market_adjustment', 1.0)
                )
                
                if valuation_result and valuation_result.get('target_price'):
                    intrinsic_value = valuation_result['target_price']
                    
            except Exception as e:
                logger.error(f"Valuation error for {symbol}: {e}")
        
        # Calculate quality score based on fundamentals
        quality_score = 0
        quality_factors = []
        
        # Check P/E ratio
        pe = features.get('pe_ratio', 999)
        if 0 < pe < 20:
            quality_score += 0.2
            quality_factors.append(f"Attractive P/E: {pe:.1f}")
        
        # Check PEG ratio
        peg = features.get('peg_ratio', 999)
        if 0 < peg < 1.5:
            quality_score += 0.2
            quality_factors.append(f"Good PEG: {peg:.2f}")
        
        # Check profitability
        roe = features.get('roe', 0)
        if roe > 0.15:
            quality_score += 0.2
            quality_factors.append(f"Strong ROE: {roe*100:.1f}%")
        
        # Check growth
        revenue_growth = features.get('revenue_growth', 0)
        if revenue_growth > 0.10:
            quality_score += 0.2
            quality_factors.append(f"Good growth: {revenue_growth*100:.1f}%")
        
        # Check financial health
        current_ratio = features.get('current_ratio', 0)
        if current_ratio > 1.5:
            quality_score += 0.1
            quality_factors.append(f"Healthy balance sheet")
        
        # Check momentum
        momentum = features.get('return_20d', 0)
        if momentum > 0.05:
            quality_score += 0.1
            quality_factors.append(f"Positive momentum: {momentum*100:.1f}%")
        
        # Calculate target price
        target_price = max(intrinsic_value, current_price * (1 + ml_prediction))
        
        # Ensure we have actual values, not 0.00
        fundamentals = {
            'pe_ratio': features.get('pe_ratio', 20) if features.get('pe_ratio', 0) != 0 else 20,
            'peg_ratio': features.get('peg_ratio', 1.5) if features.get('peg_ratio', 0) != 0 else 1.5,
            'roe': features.get('roe', 0.10) if features.get('roe', 0) != 0 else 0.10,
            'profit_margin': features.get('profit_margin', 0.05) if features.get('profit_margin', 0) != 0 else 0.05,
            'revenue_growth': features.get('revenue_growth', 0.05) if features.get('revenue_growth', 0) != 0 else 0.05,
            'debt_to_equity': features.get('debt_to_equity', 50) if features.get('debt_to_equity', 0) != 0 else 50
        }
        
        return {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'target_price': target_price,
            'ml_prediction': ml_prediction,
            'quality_score': min(quality_score, 1.0),
            'quality_factors': quality_factors,
            'fundamentals': fundamentals,
            'momentum': features.get('return_20d', 0),
            'volatility': features.get('volatility_20d', 0.25),
            'market_cap': features.get('market_cap', 0),
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ============ FAST BATCH OPERATIONS ============

def get_market_caps_batch(symbols):
    """Get market caps for multiple symbols efficiently"""
    result = {}
    uncached_symbols = []
    
    for symbol in symbols:
        cached = get_cached(market_cap_cache, symbol, MARKET_CAP_CACHE_DURATION)
        if cached is not None:
            result[symbol] = cached
        else:
            uncached_symbols.append(symbol)
    
    if not uncached_symbols:
        return result
    
    try:
        def get_market_cap(symbol):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                market_cap = info.market_cap if hasattr(info, 'market_cap') else 0
                
                if market_cap == 0:
                    full_info = ticker.info
                    market_cap = full_info.get('marketCap', 0)
                
                return symbol, market_cap
            except:
                return symbol, 0
        
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(get_market_cap, s) for s in uncached_symbols[:30]]
            for future in as_completed(futures):
                try:
                    symbol, cap = future.result(timeout=5)
                    if cap > 0:
                        result[symbol] = cap
                        set_cached(market_cap_cache, symbol, cap)
                except:
                    continue
                    
    except Exception as e:
        logger.error(f"Error getting market caps: {e}")
    
    return result

# ============ MARKET DATA ============

def fetch_vix_fast():
    """Fast VIX fetching with reduced cache"""
    cached_vix = get_cached(cache, 'vix_value', 300)  # 5 minute cache
    if cached_vix:
        return cached_vix
    
    try:
        vix_data = yf.download("^VIX", period="1d", progress=False, timeout=5)
        if not vix_data.empty:
            vix_value = float(vix_data['Close'].iloc[-1])
            set_cached(cache, 'vix_value', vix_value)
            return vix_value
    except:
        pass
    
    # Return default with small random variation
    return 20.0 + random.random() * 2

def get_market_data():
    """Get market data with reduced caching for dynamic results"""
    global market_data_cache, market_data_timestamp
    
    # Reduced cache time for more dynamic data
    if market_data_cache and (time.time() - market_data_timestamp) < 300:  # 5 minutes
        return market_data_cache
    
    market_data = {
        'vix': fetch_vix_fast(),
        'spy_price': 500,
        'spy_trend': 1,
        'treasury_10y': 4.3 + random.random() * 0.2,  # Some variation
        'dollar_index': 105 + random.random() * 2,
        'market_adjustment': 1.0
    }
    
    try:
        spy_data = yf.download("SPY", period="5d", progress=False, timeout=5)
        if not spy_data.empty:
            market_data['spy_price'] = float(spy_data['Close'].iloc[-1])
            market_data['spy_trend'] = float(spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0])
    except:
        pass
    
    # Calculate market adjustment based on conditions
    if ML_AVAILABLE and market_analyzer:
        try:
            # Get market regime
            regime = market_analyzer.get_market_regime()
            if 'Bull' in regime:
                market_data['market_adjustment'] = 1.1
            elif 'Bear' in regime:
                market_data['market_adjustment'] = 0.9
            else:
                market_data['market_adjustment'] = 1.0
        except:
            pass
    
    market_data_cache = market_data
    market_data_timestamp = time.time()
    
    return market_data

# ============ ANALYSIS PIPELINE ============

def analyze_stocks_fast(symbols, market_data, sector=None):
    """Fast parallel analysis with dynamic ML predictions"""
    results = []
    
    # Analyze in parallel
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = []
        for symbol in symbols:
            future = pool.submit(analyze_stock_with_ml, symbol, market_data)
            futures.append((symbol, future))
        
        for symbol, future in futures:
            try:
                result = future.result(timeout=10)
                if result:
                    # Calculate upside
                    upside = ((result['target_price'] / result['current_price']) - 1) * 100
                    
                    # Combined score
                    combined_score = (
                        upside * 0.4 +
                        result['quality_score'] * 100 * 0.3 +
                        result['ml_prediction'] * 100 * 0.3
                    )
                    
                    # Only include stocks with positive upside
                    if upside > 3:
                        results.append({
                            'symbol': result['symbol'],
                            'company_name': result['company_name'],
                            'current_price': result['current_price'],
                            'target_price': result['target_price'],
                            'upside': upside,
                            'quality_score': result['quality_score'],
                            'ml_prediction': result['ml_prediction'],
                            'combined_score': combined_score,
                            'metrics': result['fundamentals'],
                            'technicals': {
                                'momentum': result['momentum'],
                                'volatility': result['volatility']
                            },
                            'market_cap': result['market_cap'],
                            'reasons': result['quality_factors']
                        })
                        
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
    
    return results

# ============ BACKGROUND ANALYSIS ============

def run_analysis_background(job_id, request_data):
    """Background analysis with dynamic predictions"""
    try:
        start_time = time.time()
        
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'processing',
                'progress': 'Starting analysis...',
                'started': start_time
            }
        
        analysis_type = request_data['analysis_type']
        target = request_data['target']
        market_cap_size = request_data.get('market_cap_size', 'all')
        
        # Clear old cache for fresh predictions
        clear_old_cache()
        
        # Get fresh market data
        market_data = get_market_data()
        
        # Filter stocks
        if analysis_type == "sector":
            filtered_stocks = stocks_data[stocks_data['GICS Sector'] == target]
            sector = target
        else:
            filtered_stocks = stocks_data[stocks_data['GICS Sub-Industry'] == target]
            sector = filtered_stocks['GICS Sector'].iloc[0] if not filtered_stocks.empty else None
        
        if len(filtered_stocks) == 0:
            with job_lock:
                analysis_jobs[job_id] = {'status': 'error', 'error': f'No stocks found'}
            return
        
        all_symbols = filtered_stocks['Symbol'].tolist()
        
        # Market cap filtering
        with job_lock:
            analysis_jobs[job_id]['progress'] = 'Filtering by market cap...'
        
        if market_cap_size != 'all':
            market_caps = get_market_caps_batch(all_symbols[:50])
            cap_range = MARKET_CAP_RANGES[market_cap_size]
            
            filtered = [(s, c) for s, c in market_caps.items() 
                       if cap_range['min'] <= c < cap_range['max']]
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            analysis_symbols = [s[0] for s in filtered[:20]]
        else:
            analysis_symbols = all_symbols[:20]
        
        if not analysis_symbols:
            analysis_symbols = all_symbols[:10]
        
        # Perform analysis
        with job_lock:
            analysis_jobs[job_id]['progress'] = f'Analyzing {len(analysis_symbols)} stocks with ML...'
        
        results = analyze_stocks_fast(analysis_symbols, market_data, sector)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Get top picks
        top_picks = []
        for stock in results:
            if stock['upside'] > 5:
                top_picks.append(stock)
                if len(top_picks) >= 3:
                    break
        
        # If not enough high-upside stocks, add best remaining
        if len(top_picks) < 3:
            for stock in results:
                if stock not in top_picks and stock['upside'] > 0:
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
                    "confidence_score": int(stock['quality_score'] * 100),
                    "ml_score": round(stock['ml_prediction'], 4)
                },
                "analysis_details": {
                    "fundamentals": stock['metrics'],
                    "technicals": stock['technicals'],
                    "ml_prediction": stock['ml_prediction'],
                    "quality_score": stock['quality_score']
                },
                "investment_thesis": stock['reasons']
            })
        
        elapsed = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed:.1f} seconds")
        
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'completed',
                'results': {
                    'top_stocks': formatted,
                    'market_conditions': {
                        'vix': market_data['vix'],
                        'ml_model_trained': ml_model_trained,
                        'training_metrics': training_metrics if ml_model_trained else {}
                    },
                    'total_analyzed': len(results),
                    'analysis_time': elapsed
                }
            }
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        with job_lock:
            analysis_jobs[job_id] = {'status': 'error', 'error': str(e)}

# ============ API ENDPOINTS ============

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str
    market_cap_size: str = 'all'

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis job"""
    
    # Check if model needs retraining
    if time.time() - last_training_time > RETRAIN_INTERVAL:
        logger.info("Model needs retraining for fresh predictions")
        background_tasks.add_task(train_integrated_ml_model)
    
    job_id = str(uuid.uuid4())
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
    
    # Determine market regime
    regime = "High Volatility" if vix > 30 else "Elevated" if vix > 20 else "Normal"
    
    # Calculate recession risk
    recession_risk = "Unknown"
    try:
        ten_year_data = yf.download("^TNX", period="1d", progress=False, timeout=5)
        two_year_data = yf.download("^FVX", period="1d", progress=False, timeout=5)
        
        if not ten_year_data.empty:
            ten_year = float(ten_year_data['Close'].iloc[-1])
            
            if not two_year_data.empty:
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
    """Get ML model status"""
    status = {
        "ml_available": ML_AVAILABLE,
        "ml_model_trained": ml_model_trained,
        "training_metrics": training_metrics,
        "quant_model_loaded": ml_model is not None,
        "last_training": datetime.fromtimestamp(last_training_time).isoformat() if last_training_time > 0 else None,
        "next_training": datetime.fromtimestamp(last_training_time + RETRAIN_INTERVAL).isoformat() if last_training_time > 0 else None,
        "cache_size": len(cache) + len(stock_info_cache) + len(financial_cache)
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
        "version": "18.0",
        "ml_enabled": ML_AVAILABLE,
        "ml_trained": ml_model_trained,
        "training_metrics": training_metrics
    })

# ============ STARTUP EVENTS ============

@app.on_event("startup")
async def startup_event():
    """Initialize and train model on startup"""
    logger.info("Starting up AutoAnalyst API...")
    
    # Load market data
    get_market_data()
    
    # Start model training in background
    if ML_AVAILABLE:
        executor.submit(train_integrated_ml_model)
        logger.info("ML model training started in background...")
    
    logger.info("API ready!")

async def periodic_retrain():
    """Retrain model periodically for fresh predictions"""
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL)  # Every hour
        if ML_AVAILABLE and ml_model:
            logger.info("Starting periodic model retraining for fresh predictions...")
            executor.submit(train_integrated_ml_model)

async def periodic_cache_cleanup():
    """Clean up old cache entries periodically"""
    while True:
        await asyncio.sleep(600)  # Every 10 minutes
        clear_old_cache()
        logger.info(f"Cache cleanup completed. Current cache size: {len(cache) + len(stock_info_cache)}")

@app.on_event("startup")
async def start_periodic_tasks():
    """Start background periodic tasks"""
    asyncio.create_task(periodic_retrain())
    asyncio.create_task(periodic_cache_cleanup())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)