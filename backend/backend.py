#!/usr/bin/env python3
"""
FastAPI Backend - Full ML Analysis for Best Stock Selection
"""

import os
import sys
import glob
import logging
import json
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="10.0.0")

# Thread pool
executor = ThreadPoolExecutor(max_workers=30)

# In-memory cache
cache = {}
CACHE_DURATION = 1800  # 30 minutes for fresher data

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ============ CACHING SYSTEM ============

def get_cache_key(prefix, *args):
    """Generate cache key"""
    key_str = f"{prefix}:{':'.join(str(arg) for arg in args)}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_from_cache(key):
    """Get from cache if not expired"""
    if key in cache:
        data, timestamp = cache[key]
        if time.time() - timestamp < CACHE_DURATION:
            logger.info(f"Cache hit for {key}")
            return data
        else:
            del cache[key]
    return None

def set_cache(key, data):
    """Set cache with timestamp"""
    cache[key] = (data, time.time())

# ============ DATA LOADING ============

def load_csv_files():
    """Load CSV files from data directory"""
    data_path = os.environ.get('DATA_PATH', '/app/data')
    if not os.path.exists(data_path):
        data_path = 'data'
    
    logger.info(f"Loading CSV files from: {data_path}")
    csv_pattern = os.path.join(data_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        logger.warning("No CSV files found, using sample data")
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            'GICS Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                           'Technology', 'Technology', 'Consumer Discretionary'],
            'GICS Sub-Industry': ['Hardware', 'Software', 'Internet', 'E-Commerce',
                                 'Semiconductors', 'Social Media', 'Automobiles']
        })
    
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df = df.replace([np.inf, -np.inf], np.nan)
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna('')
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].fillna(0)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {csv_file}")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total stocks loaded: {len(combined)}")
        return combined
    return pd.DataFrame()

# Load data
stocks_data = load_csv_files()
logger.info(f"Stocks data initialized: {len(stocks_data)} stocks")

# Import models
ML_AVAILABLE = False
ml_model = None
market_analyzer = None
valuator = None

try:
    logger.info("Importing quant_model.py...")
    from quant_model import (
        QuantFinanceMLModel, 
        MarketConditionsAnalyzer, 
        EnhancedValuation
    )
    
    ml_model = QuantFinanceMLModel()
    ml_model.master_df = stocks_data
    ml_model.process_gics_data()
    
    market_analyzer = MarketConditionsAnalyzer()
    valuator = EnhancedValuation()
    
    ML_AVAILABLE = True
    logger.info("✅ Models loaded successfully!")
    
except Exception as e:
    logger.error(f"❌ Error loading quant_model.py: {e}")
    ML_AVAILABLE = False

# ============ COMPREHENSIVE STOCK ANALYSIS ============

def comprehensive_stock_analysis(symbol, stock_data, ml_model, valuator, market_analyzer, sector):
    """
    Perform COMPLETE analysis using ALL ML features
    Returns detailed analysis with reasoning
    """
    try:
        info = stock_data.get('info', {})
        hist_dict = stock_data.get('history', {})
        
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not current_price and hist_dict.get('Close'):
            close_prices = list(hist_dict['Close'].values())
            if close_prices:
                current_price = close_prices[-1]
        
        if not current_price or current_price <= 0:
            return None
        
        # Convert history to DataFrame
        hist = pd.DataFrame(hist_dict) if hist_dict else pd.DataFrame()
        
        logger.info(f"Analyzing {symbol}: Price=${current_price:.2f}")
        
        # ========== 1. FUNDAMENTAL ANALYSIS ==========
        fundamentals = {
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'profit_margin': info.get('profitMargins', 0),
            'operating_margin': info.get('operatingMargins', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'earnings_growth': info.get('earningsGrowth', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0),
            'roa': info.get('returnOnAssets', 0),
            'price_to_book': info.get('priceToBook', 0),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
            'free_cash_flow': info.get('freeCashflow', 0),
            'analyst_rating': info.get('recommendationMean', 3),
            'target_mean_price': info.get('targetMeanPrice', current_price),
            'target_high_price': info.get('targetHighPrice', current_price),
            'target_low_price': info.get('targetLowPrice', current_price)
        }
        
        # Fundamental Score (0-1)
        fundamental_score = 0
        fundamental_reasons = []
        
        # P/E Analysis
        if 0 < fundamentals['pe_ratio'] < 25:
            fundamental_score += 0.15
            fundamental_reasons.append(f"Attractive P/E ratio of {fundamentals['pe_ratio']:.1f}")
        elif 25 <= fundamentals['pe_ratio'] < 35:
            fundamental_score += 0.08
        
        # PEG Analysis
        if 0 < fundamentals['peg_ratio'] < 1:
            fundamental_score += 0.15
            fundamental_reasons.append(f"Excellent PEG ratio of {fundamentals['peg_ratio']:.2f} (undervalued)")
        elif 1 <= fundamentals['peg_ratio'] < 1.5:
            fundamental_score += 0.08
        
        # Profitability
        if fundamentals['roe'] > 0.20:
            fundamental_score += 0.15
            fundamental_reasons.append(f"Strong ROE of {fundamentals['roe']*100:.1f}%")
        elif fundamentals['roe'] > 0.15:
            fundamental_score += 0.08
        
        if fundamentals['profit_margin'] > 0.15:
            fundamental_score += 0.1
            fundamental_reasons.append(f"Healthy profit margin of {fundamentals['profit_margin']*100:.1f}%")
        
        # Growth
        if fundamentals['revenue_growth'] > 0.15:
            fundamental_score += 0.15
            fundamental_reasons.append(f"Strong revenue growth of {fundamentals['revenue_growth']*100:.1f}%")
        elif fundamentals['revenue_growth'] > 0.08:
            fundamental_score += 0.08
        
        # Financial Health
        if fundamentals['debt_to_equity'] < 0.5:
            fundamental_score += 0.1
            fundamental_reasons.append("Strong balance sheet with low debt")
        elif fundamentals['debt_to_equity'] < 1:
            fundamental_score += 0.05
        
        # Analyst Sentiment
        if fundamentals['analyst_rating'] < 2:
            fundamental_score += 0.1
            fundamental_reasons.append("Strong buy rating from analysts")
        elif fundamentals['analyst_rating'] < 2.5:
            fundamental_score += 0.05
        
        # ========== 2. TECHNICAL ANALYSIS ==========
        technical_score = 0
        technical_reasons = []
        technicals = {}
        
        if len(hist) > 50:
            try:
                close_prices = hist['Close']
                
                # RSI
                close_delta = close_prices.diff()
                gain = (close_delta.where(close_delta > 0, 0)).rolling(window=14).mean()
                loss = (-close_delta.where(close_delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs.iloc[-1]))
                technicals['rsi'] = rsi
                
                # Moving Averages
                ma20 = close_prices.rolling(20).mean().iloc[-1]
                ma50 = close_prices.rolling(50).mean().iloc[-1]
                ma200 = close_prices.rolling(200).mean().iloc[-1] if len(hist) > 200 else ma50
                
                technicals['ma20'] = ma20
                technicals['ma50'] = ma50
                technicals['ma200'] = ma200
                
                # Momentum
                returns_20d = (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]
                returns_60d = (close_prices.iloc[-1] - close_prices.iloc[-60]) / close_prices.iloc[-60] if len(hist) > 60 else returns_20d
                
                technicals['momentum_20d'] = returns_20d
                technicals['momentum_60d'] = returns_60d
                technicals['volatility'] = close_prices.pct_change().std() * np.sqrt(252)
                
                # Technical Scoring
                if current_price > ma20 > ma50:
                    technical_score += 0.25
                    technical_reasons.append("Strong uptrend (price above moving averages)")
                
                if 30 < rsi < 70:
                    technical_score += 0.25
                    if rsi < 40:
                        technical_reasons.append("Oversold condition (potential bounce)")
                    elif rsi > 60:
                        technical_reasons.append("Strong momentum without being overbought")
                
                if returns_20d > 0.05:
                    technical_score += 0.25
                    technical_reasons.append(f"Positive momentum: {returns_20d*100:.1f}% gain in 20 days")
                
                if technicals['volatility'] < 0.4:
                    technical_score += 0.25
                    technical_reasons.append("Low volatility indicates stability")
                
            except Exception as e:
                logger.error(f"Technical analysis error for {symbol}: {e}")
                technicals = {'rsi': 50, 'momentum_20d': 0, 'momentum_60d': 0, 'volatility': 0.3}
        else:
            technicals = {'rsi': 50, 'momentum_20d': 0, 'momentum_60d': 0, 'volatility': 0.3}
        
        # ========== 3. ML PREDICTION (ALWAYS RUN) ==========
        ml_prediction = 0
        ml_confidence = 0
        
        if ml_model:
            try:
                logger.info(f"Running ML prediction for {symbol}")
                
                # Ensure model is trained
                if not hasattr(ml_model, 'ml_model') or ml_model.ml_model is None:
                    logger.info("Training ML model...")
                    training_data = ml_model.prepare_training_data([symbol])
                    if not training_data.empty:
                        training_data = ml_model.add_sentiment_features(training_data)
                        ml_model.train_prediction_model(training_data)
                
                # Prepare features
                current_features = {
                    'recent_returns': technicals.get('momentum_20d', 0),
                    'volatility': technicals.get('volatility', 0.3),
                    'pe_ratio': fundamentals['pe_ratio'],
                    'market_cap': fundamentals['market_cap'],
                    'peg_ratio': fundamentals['peg_ratio'],
                    'profit_margin': fundamentals['profit_margin'],
                    'revenue_growth': fundamentals['revenue_growth'],
                    'debt_to_equity': fundamentals['debt_to_equity'],
                    'roe': fundamentals['roe'],
                    'price_to_book': fundamentals['price_to_book'],
                    'rsi': technicals.get('rsi', 50),
                    'vix': 20,
                    'treasury_10y': 3.5,
                    'dollar_index': 100,
                    'spy_trend': 1
                }
                
                # Get prediction
                ml_prediction = ml_model.calculate_stock_predictions(symbol, current_features)
                ml_confidence = abs(ml_prediction)  # Confidence based on strength of prediction
                
                logger.info(f"ML Prediction for {symbol}: {ml_prediction*100:.2f}% expected return")
                
            except Exception as e:
                logger.error(f"ML prediction error for {symbol}: {e}")
                ml_prediction = 0
                ml_confidence = 0
        
        # ========== 4. SENTIMENT ANALYSIS (ALWAYS RUN) ==========
        sentiment_score = 0
        sentiment_reasons = []
        
        if ml_model and hasattr(ml_model, 'simple_sentiment'):
            try:
                company_name = info.get('longName', symbol)
                queries = [
                    f"{symbol} {company_name} stock outlook 2024",
                    f"{symbol} earnings growth potential",
                    f"Is {symbol} a good investment"
                ]
                
                sentiments = []
                for query in queries:
                    result = ml_model.simple_sentiment(query)
                    if result and len(result) > 0:
                        sent = result[0]
                        if sent['label'] == 'positive':
                            sentiments.append(sent['score'])
                        elif sent['label'] == 'negative':
                            sentiments.append(-sent['score'])
                
                if sentiments:
                    sentiment_score = np.mean(sentiments)
                    if sentiment_score > 0.5:
                        sentiment_reasons.append(f"Very positive market sentiment (score: {sentiment_score:.2f})")
                    elif sentiment_score > 0:
                        sentiment_reasons.append(f"Positive market sentiment (score: {sentiment_score:.2f})")
                
                logger.info(f"Sentiment for {symbol}: {sentiment_score:.3f}")
                
            except Exception as e:
                logger.error(f"Sentiment error for {symbol}: {e}")
                sentiment_score = 0
        
        # ========== 5. COMPREHENSIVE VALUATION (ALWAYS RUN) ==========
        target_price = fundamentals['target_mean_price']
        valuation_confidence = 0.5
        valuation_methods = {}
        
        if valuator:
            try:
                logger.info(f"Running comprehensive valuation for {symbol}")
                
                # Market adjustment based on sector
                market_adjustment = 1.0
                if market_analyzer:
                    sector_conditions = market_analyzer.analyze_sector_conditions(sector)
                    market_adjustment = market_analyzer.calculate_market_adjustment_factor(sector)
                    
                    # Adjust for sector momentum
                    if sector_conditions.get('momentum', 0) > 0:
                        market_adjustment *= 1.05
                
                # Get comprehensive valuation
                valuation_result = valuator.calculate_comprehensive_valuation(
                    symbol, 
                    ml_prediction, 
                    sentiment_score, 
                    market_adjustment
                )
                
                target_price = valuation_result.get('target_price', target_price)
                valuation_confidence = valuation_result.get('confidence', 0.5)
                valuation_methods = valuation_result.get('valuations', {})
                
                logger.info(f"Valuation for {symbol}: Target=${target_price:.2f}, Confidence={valuation_confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Valuation error for {symbol}: {e}")
        
        # ========== 6. CALCULATE FINAL SCORES ==========
        
        # Expected upside
        upside_potential = ((target_price / current_price) - 1) * 100 if current_price > 0 else 0
        
        # ONLY CONSIDER STOCKS WITH POSITIVE UPSIDE
        if upside_potential <= 0:
            logger.info(f"Skipping {symbol}: Negative upside {upside_potential:.1f}%")
            return None
        
        # ML-Adjusted upside (incorporate ML prediction)
        ml_adjusted_upside = upside_potential * (1 + ml_prediction)
        
        # Comprehensive Investment Score (0-100)
        investment_score = (
            fundamental_score * 30 +  # 30% weight on fundamentals
            technical_score * 20 +     # 20% weight on technicals
            (ml_prediction * 100) * 0.25 +  # 25% weight on ML prediction
            (sentiment_score * 50) * 0.1 +  # 10% weight on sentiment
            (valuation_confidence * 100) * 0.15  # 15% weight on valuation confidence
        )
        
        # Risk-adjusted score
        risk_adjusted_score = investment_score * (1 - technicals.get('volatility', 0.3)/2)
        
        # Generate investment thesis
        investment_thesis = {
            'fundamental_reasons': fundamental_reasons,
            'technical_reasons': technical_reasons,
            'sentiment_reasons': sentiment_reasons,
            'ml_confidence': ml_confidence,
            'overall_recommendation': self._generate_recommendation(
                investment_score, upside_potential, fundamental_score, technical_score
            )
        }
        
        return {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'target_price': target_price,
            'upside_potential': upside_potential,
            'ml_adjusted_upside': ml_adjusted_upside,
            'ml_prediction': ml_prediction,
            'sentiment_score': sentiment_score,
            'fundamental_score': fundamental_score,
            'technical_score': technical_score,
            'investment_score': investment_score,
            'risk_adjusted_score': risk_adjusted_score,
            'valuation_confidence': valuation_confidence,
            'fundamentals': fundamentals,
            'technicals': technicals,
            'valuation_methods': valuation_methods,
            'investment_thesis': investment_thesis
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis error for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def _generate_recommendation(investment_score, upside, fundamental_score, technical_score):
    """Generate investment recommendation text"""
    if investment_score > 80:
        return "STRONG BUY - Exceptional opportunity with multiple positive catalysts"
    elif investment_score > 65:
        return "BUY - Solid investment with good upside potential"
    elif investment_score > 50:
        return "MODERATE BUY - Decent opportunity with some positive factors"
    else:
        return "HOLD - Limited upside, consider other opportunities"

# ============ BATCH FETCHING ============

def batch_fetch_stock_info(symbols):
    """Batch fetch stock info with caching"""
    results = {}
    to_fetch = []
    
    # Check cache first
    for symbol in symbols:
        cache_key = get_cache_key('stock_info', symbol)
        cached_data = get_from_cache(cache_key)
        if cached_data:
            results[symbol] = cached_data
        else:
            to_fetch.append(symbol)
    
    if not to_fetch:
        return results
    
    logger.info(f"Fetching {len(to_fetch)} stocks from Yahoo Finance...")
    
    def fetch_single(symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="6mo")  # Get 6 months for better analysis
            
            data = {
                'info': info,
                'history': hist.to_dict() if not hist.empty else {}
            }
            
            # Cache the result
            cache_key = get_cache_key('stock_info', symbol)
            set_cache(cache_key, data)
            
            return symbol, data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return symbol, None
    
    # Parallel fetching
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(fetch_single, symbol) for symbol in to_fetch]
        
        for future in as_completed(futures):
            try:
                symbol, data = future.result(timeout=10)
                if data:
                    results[symbol] = data
            except:
                continue
    
    return results

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Full ML Analysis to find the BEST performers"""
    start_time = time.time()
    logger.info(f"="*60)
    logger.info(f"COMPREHENSIVE ANALYSIS: {request.analysis_type} - {request.target}")
    logger.info(f"="*60)
    
    # Check cache
    cache_key = get_cache_key('analysis', request.analysis_type, request.target)
    cached_result = get_from_cache(cache_key)
    if cached_result:
        logger.info("Returning cached result")
        return JSONResponse(content=cached_result, headers={"Access-Control-Allow-Origin": "*"})
    
    if not ML_AVAILABLE or ml_model is None:
        return JSONResponse(
            content={"status": "failed", "error": "ML Model not available"},
            status_code=500
        )
    
    try:
        # Filter stocks
        if request.analysis_type == "sector":
            filtered_stocks = stocks_data[stocks_data['GICS Sector'] == request.target]
        else:
            filtered_stocks = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
        
        if len(filtered_stocks) == 0:
            return JSONResponse(
                content={"status": "no_data", "error": f"No stocks found for {request.target}"},
                status_code=404
            )
        
        logger.info(f"Found {len(filtered_stocks)} stocks in {request.target}")
        
        # Get ALL symbols for comprehensive analysis
        all_symbols = filtered_stocks['Symbol'].tolist()
        
        # First pass: Get market caps to identify top stocks
        logger.info("Phase 1: Identifying top stocks by market cap...")
        market_caps = {}
        
        def get_market_cap(symbol):
            try:
                ticker = yf.Ticker(symbol)
                cap = ticker.info.get('marketCap', 0)
                return symbol, cap
            except:
                return symbol, 0
        
        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = [pool.submit(get_market_cap, sym) for sym in all_symbols[:100]]
            for future in as_completed(futures):
                try:
                    symbol, cap = future.result(timeout=3)
                    if cap > 0:
                        market_caps[symbol] = cap
                except:
                    continue
        
        # Get top 25 by market cap for detailed analysis
        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:25]
        analysis_symbols = [stock[0] for stock in sorted_stocks]
        
        logger.info(f"Phase 2: Fetching detailed data for {len(analysis_symbols)} stocks...")
        
        # Batch fetch all stock data
        stock_data_dict = batch_fetch_stock_info(analysis_symbols)
        
        # Setup ML model with all stocks
        logger.info("Phase 3: Training ML models...")
        ml_model.selected_stocks = analysis_symbols
        ml_model.master_df = stocks_data
        ml_model.process_gics_data()
        
        # ALWAYS train the model for best results
        try:
            training_symbols = list(stock_data_dict.keys())[:15]
            training_data = ml_model.prepare_training_data(training_symbols)
            
            if not training_data.empty:
                logger.info(f"Training on {len(training_data)} samples...")
                training_data = ml_model.add_sentiment_features(training_data)
                ml_model.training_data = training_data
                ml_model.train_prediction_model(training_data)
                logger.info("ML model training complete")
        except Exception as e:
            logger.error(f"Training error: {e}")
        
        # Phase 4: Comprehensive analysis of each stock
        logger.info("Phase 4: Running comprehensive analysis...")
        analysis_results = []
        
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = []
            for symbol in stock_data_dict.keys():
                future = pool.submit(
                    comprehensive_stock_analysis,
                    symbol,
                    stock_data_dict[symbol],
                    ml_model,
                    valuator,
                    market_analyzer,
                    request.target
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=15)
                    if result and result['upside_potential'] > 0:
                        analysis_results.append(result)
                        logger.info(f"✓ {result['symbol']}: Score={result['investment_score']:.1f}, Upside={result['upside_potential']:.1f}%")
                except Exception as e:
                    logger.error(f"Analysis error: {e}")
        
        # Phase 5: Select ONLY the BEST performers
        logger.info("Phase 5: Selecting best performers...")
        
        # Filter for quality and positive upside
        quality_stocks = [
            stock for stock in analysis_results
            if stock['upside_potential'] > 5  # At least 5% upside
            and stock['fundamental_score'] > 0.3  # Decent fundamentals
            and stock['investment_score'] > 40  # Overall good score
        ]
        
        if not quality_stocks:
            # Fallback to best available
            quality_stocks = [
                stock for stock in analysis_results
                if stock['upside_potential'] > 0
            ]
        
        # Sort by multiple criteria
        quality_stocks.sort(
            key=lambda x: (
                x['risk_adjusted_score'] * 0.4 +  # Risk-adjusted returns
                x['ml_adjusted_upside'] * 0.3 +   # ML-predicted upside
                x['investment_score'] * 0.3        # Overall score
            ),
            reverse=True
        )
        
        # Get top 3
        top_3_stocks = quality_stocks[:3]
        
        # Format results
        formatted_results = []
        for rank, stock in enumerate(top_3_stocks, 1):
            # Build recommendation narrative
            thesis_points = []
            if stock['investment_thesis']['fundamental_reasons']:
                thesis_points.extend(stock['investment_thesis']['fundamental_reasons'][:2])
            if stock['investment_thesis']['technical_reasons']:
                thesis_points.extend(stock['investment_thesis']['technical_reasons'][:1])
            if stock['ml_prediction'] > 0.1:
                thesis_points.append(f"ML models predict {stock['ml_prediction']*100:.1f}% returns")
            
            formatted_results.append({
                "rank": rank,
                "symbol": stock['symbol'],
                "company_name": stock['company_name'],
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside_potential'], 1),
                    "ml_adjusted_upside": round(stock['ml_adjusted_upside'], 1),
                    "confidence_score": int(stock['valuation_confidence'] * 100),
                    "sentiment_score": round(stock['sentiment_score'], 2),
                    "ml_score": round(stock['ml_prediction'], 3),
                    "investment_score": round(stock['investment_score'], 1),
                    "risk_adjusted_score": round(stock['risk_adjusted_score'], 1)
                },
                "analysis_details": {
                    "fundamentals": {
                        "pe_ratio": stock['fundamentals']['pe_ratio'],
                        "peg_ratio": stock['fundamentals']['peg_ratio'],
                        "roe": stock['fundamentals']['roe'],
                        "profit_margin": stock['fundamentals']['profit_margin'],
                        "revenue_growth": stock['fundamentals']['revenue_growth'],
                        "debt_to_equity": stock['fundamentals']['debt_to_equity']
                    },
                    "technicals": stock['technicals'],
                    "valuation_methods": stock['valuation_methods'],
                    "ml_prediction": stock['ml_prediction'],
                    "quality_scores": {
                        "fundamental_score": round(stock['fundamental_score'], 2),
                        "technical_score": round(stock['technical_score'], 2)
                    }
                },
                "investment_thesis": {
                    "recommendation": stock['investment_thesis']['overall_recommendation'],
                    "key_points": thesis_points
                }
            })
        
        elapsed = time.time() - start_time
        logger.info(f"="*60)
        logger.info(f"✅ Analysis complete in {elapsed:.1f} seconds")
        logger.info(f"Top picks: {', '.join([s['symbol'] for s in formatted_results])}")
        logger.info(f"="*60)
        
        result_data = {
            "status": "completed",
            "analysis_type": request.analysis_type,
            "target": request.target,
            "results": {
                "top_stocks": formatted_results,
                "analysis_summary": {
                    "total_analyzed": len(analysis_results),
                    "quality_filtered": len(quality_stocks),
                    "sector": request.target,
                    "methodology": "Full ML analysis with fundamental, technical, sentiment, and valuation models"
                }
            },
            "ml_powered": True,
            "execution_time": round(elapsed, 1)
        }
        
        # Cache result
        set_cache(cache_key, result_data)
        
        return JSONResponse(
            content=result_data,
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )

# Keep other endpoints the same...
@app.get("/")
async def root():
    return JSONResponse(
        content={
            "name": "AutoAnalyst API",
            "version": "10.0.0",
            "status": "running",
            "ml_enabled": ML_AVAILABLE,
            "stocks_loaded": len(stocks_data)
        }
    )

@app.get("/api/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "ml_available": ML_AVAILABLE,
            "stocks_count": len(stocks_data),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.get("/api/stocks/list")
async def get_stocks_list():
    sectors = []
    sub_industries = []
    
    if 'GICS Sector' in stocks_data.columns:
        sectors = stocks_data['GICS Sector'].dropna().unique().tolist()
        sectors = [s for s in sectors if isinstance(s, str) and s != '']
        sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist()
        sub_industries = [s for s in sub_industries if isinstance(s, str) and s != '']
    
    return JSONResponse(
        content={
            "sectors": sectors,
            "sub_industries": sub_industries,
            "total_stocks": len(stocks_data)
        }
    )

@app.get("/api/market-conditions")
async def get_market_conditions():
    try:
        if market_analyzer:
            regime = market_analyzer.get_market_regime()
            fed_data = market_analyzer.get_federal_reserve_stance()
            
            return JSONResponse(
                content={
                    "regime": regime,
                    "fed_stance": fed_data.get("stance", "Neutral"),
                    "vix": 20.0,
                    "recession_risk": "Low",
                    "ml_powered": ML_AVAILABLE
                }
            )
    except:
        pass
    
    return JSONResponse(
        content={
            "regime": "Normal",
            "fed_stance": "Neutral",
            "vix": 20.0,
            "recession_risk": "Low"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)