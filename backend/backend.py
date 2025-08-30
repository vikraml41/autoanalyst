#!/usr/bin/env python3
"""
FastAPI Backend - Enhanced with better stock filtering and comprehensive analysis
"""

import os
import sys
import glob
import logging
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="7.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add CORS headers manually
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

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

# Load data first
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

# ============ API ENDPOINTS ============

@app.get("/")
async def root():
    return JSONResponse(
        content={
            "name": "AutoAnalyst API",
            "version": "7.0.0",
            "status": "running",
            "ml_enabled": ML_AVAILABLE,
            "stocks_loaded": len(stocks_data)
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/api/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "ml_available": ML_AVAILABLE,
            "stocks_count": len(stocks_data),
            "timestamp": datetime.now().isoformat()
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/api/stocks/list")
async def get_stocks_list():
    sectors = []
    sub_industries = []
    
    if ML_AVAILABLE and ml_model and hasattr(ml_model, 'sectors'):
        sectors = ml_model.sectors if ml_model.sectors else []
        sub_industries = ml_model.sub_industries if ml_model.sub_industries else []
    elif 'GICS Sector' in stocks_data.columns:
        sectors = stocks_data['GICS Sector'].dropna().unique().tolist()
        sectors = [s for s in sectors if isinstance(s, str) and s != '']
        sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist()
        sub_industries = [s for s in sub_industries if isinstance(s, str) and s != '']
    
    return JSONResponse(
        content={
            "sectors": sectors,
            "sub_industries": sub_industries,
            "total_stocks": len(stocks_data)
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/api/market-conditions")
async def get_market_conditions():
    try:
        if not market_analyzer:
            return JSONResponse(
                content={
                    "regime": "Unknown",
                    "fed_stance": "Unknown",
                    "vix": 20.0,
                    "recession_risk": "Unknown"
                },
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        regime = market_analyzer.get_market_regime()
        fed_data = market_analyzer.get_federal_reserve_stance()
        yield_curve = market_analyzer.analyze_yield_curve()
        economic_data = market_analyzer.fetch_economic_data()
        
        vix_value = 20.0
        if economic_data and 'Volatility Index' in economic_data:
            vix_value = economic_data['Volatility Index'].get('current', 20.0)
        
        return JSONResponse(
            content={
                "regime": regime,
                "fed_stance": fed_data.get("stance", "Unknown"),
                "vix": vix_value,
                "recession_risk": yield_curve.get("recession_risk", "Unknown"),
                "ml_powered": ML_AVAILABLE
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Market conditions error: {e}")
        return JSONResponse(
            content={
                "regime": "Error",
                "fed_stance": "Error",
                "vix": 20.0,
                "recession_risk": "Error"
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Enhanced analysis with better stock filtering"""
    logger.info(f"=" * 50)
    logger.info(f"Analysis START: {request.analysis_type} - {request.target}")
    
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
        
        logger.info(f"Found {len(filtered_stocks)} stocks")
        
        # Get ALL stocks for better model training
        symbols = filtered_stocks['Symbol'].tolist()
        
        # Get market caps
        market_caps = {}
        for symbol in symbols[:100]:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    market_caps[symbol] = market_cap
            except:
                continue
        
        # Get top 20 by market cap for analysis
        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:20]
        analysis_symbols = [stock[0] for stock in sorted_stocks]
        
        logger.info(f"Analyzing {len(analysis_symbols)} stocks")
        
        # Set up model
        ml_model.selected_stocks = analysis_symbols
        ml_model.master_df = stocks_data
        ml_model.process_gics_data()
        
        # Get market conditions
        market_adjustment = 1.0
        sector_analysis = {}
        
        if market_analyzer:
            try:
                market_analyzer.fetch_economic_data()
                sector_conditions = market_analyzer.analyze_sector_conditions(request.target)
                market_adjustment = market_analyzer.calculate_market_adjustment_factor(request.target)
                ml_model.market_adjustment = market_adjustment
                
                sector_analysis = {
                    "market_regime": market_analyzer.get_market_regime(),
                    "fed_stance": market_analyzer.get_federal_reserve_stance(),
                    "yield_curve": market_analyzer.analyze_yield_curve(),
                    "sector_conditions": sector_conditions,
                    "market_adjustment": market_adjustment
                }
                
                logger.info(f"Market adjustment: {market_adjustment}, Sector: {sector_conditions.get('momentum')}")
            except Exception as e:
                logger.error(f"Market analysis error: {e}")
        
        # Train model on more data
        if len(analysis_symbols) > 5:
            try:
                logger.info("Training model...")
                training_symbols = analysis_symbols[:15]
                training_data = ml_model.prepare_training_data(training_symbols)
                
                if not training_data.empty:
                    training_data = ml_model.add_sentiment_features(training_data)
                    ml_model.training_data = training_data
                    ml_model.train_prediction_model(training_data)
            except Exception as e:
                logger.error(f"Training error: {e}")
        
        # Analyze each stock
        detailed_results = []
        
        for symbol in analysis_symbols:
            try:
                logger.info(f"Analyzing {symbol}...")
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="6mo")
                
                # Get price
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if not current_price or current_price <= 0:
                    if len(hist) > 0:
                        current_price = float(hist['Close'].iloc[-1])
                    else:
                        continue
                
                # Calculate metrics
                metrics = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'analyst_rating': info.get('recommendationMean', 3),
                    'target_mean_price': info.get('targetMeanPrice', current_price)
                }
                
                # Technical indicators
                if len(hist) > 50:
                    # RSI
                    close_delta = hist['Close'].diff()
                    gain = (close_delta.where(close_delta > 0, 0)).rolling(window=14).mean()
                    loss = (-close_delta.where(close_delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs.iloc[-1]))
                    
                    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                    ma50 = hist['Close'].rolling(50).mean().iloc[-1]
                    
                    returns_20d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20]
                    returns_60d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60] if len(hist) > 60 else returns_20d
                    
                    metrics['rsi'] = rsi
                    metrics['ma20_ratio'] = current_price / ma20
                    metrics['ma50_ratio'] = current_price / ma50
                    metrics['momentum_20d'] = returns_20d
                    metrics['momentum_60d'] = returns_60d
                    metrics['volatility'] = hist['Close'].pct_change().std() * np.sqrt(252)
                else:
                    metrics['rsi'] = 50
                    metrics['ma20_ratio'] = 1
                    metrics['ma50_ratio'] = 1
                    metrics['momentum_20d'] = 0
                    metrics['momentum_60d'] = 0
                    metrics['volatility'] = 0.2
                
                # Get ML prediction
                ml_prediction = 0
                if hasattr(ml_model, 'ml_model') and ml_model.ml_model is not None:
                    try:
                        current_data = {
                            'recent_returns': metrics['momentum_20d'],
                            'volatility': metrics['volatility'],
                            'pe_ratio': metrics['pe_ratio'],
                            'market_cap': metrics['market_cap'],
                            'peg_ratio': metrics['peg_ratio'],
                            'profit_margin': metrics['profit_margin'],
                            'revenue_growth': metrics['revenue_growth'],
                            'debt_to_equity': metrics['debt_to_equity'],
                            'roe': metrics['roe'],
                            'price_to_book': metrics['price_to_book'],
                            'rsi': metrics['rsi'],
                            'vix': 20,
                            'treasury_10y': 3.5,
                            'dollar_index': 100,
                            'spy_trend': 1
                        }
                        ml_prediction = ml_model.calculate_stock_predictions(symbol, current_data)
                    except:
                        ml_prediction = 0
                
                # Get sentiment
                sentiment_score = 0
                try:
                    company_name = info.get('longName', symbol)
                    sentiment_result = ml_model.simple_sentiment(f"{symbol} {company_name} stock outlook")
                    if sentiment_result and len(sentiment_result) > 0:
                        sent = sentiment_result[0]
                        if sent['label'] == 'positive':
                            sentiment_score = sent['score']
                        elif sent['label'] == 'negative':
                            sentiment_score = -sent['score']
                except:
                    sentiment_score = 0
                
                # Calculate valuation
                target_price = current_price
                confidence = 0.5
                
                if valuator and hasattr(valuator, 'calculate_comprehensive_valuation'):
                    try:
                        valuation_result = valuator.calculate_comprehensive_valuation(
                            symbol, 
                            ml_prediction, 
                            sentiment_score, 
                            market_adjustment
                        )
                        target_price = valuation_result.get('target_price', current_price)
                        confidence = valuation_result.get('confidence', 0.5)
                        metrics['valuation_details'] = valuation_result.get('valuations', {})
                    except:
                        if metrics['target_mean_price'] > 0:
                            target_price = metrics['target_mean_price'] * market_adjustment
                
                # Calculate scores
                upside = ((target_price / current_price) - 1) * 100 if current_price > 0 else 0
                
                # Quality score
                quality_score = 0
                if metrics['pe_ratio'] > 0 and metrics['pe_ratio'] < 30:
                    quality_score += 0.2
                if metrics['peg_ratio'] > 0 and metrics['peg_ratio'] < 1.5:
                    quality_score += 0.2
                if metrics['roe'] > 0.15:
                    quality_score += 0.2
                if metrics['revenue_growth'] > 0.1:
                    quality_score += 0.2
                if metrics['debt_to_equity'] < 1:
                    quality_score += 0.1
                if metrics['analyst_rating'] < 2.5:
                    quality_score += 0.1
                
                # Combined score
                combined_score = (
                    upside * 0.3 +
                    quality_score * 100 * 0.3 +
                    (ml_prediction * 100) * 0.2 +
                    (sentiment_score * 100) * 0.1 +
                    (confidence * 100) * 0.1
                )
                
                detailed_results.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'target_price': target_price,
                    'upside': upside,
                    'ml_prediction': ml_prediction,
                    'sentiment': sentiment_score,
                    'confidence': confidence,
                    'quality_score': quality_score,
                    'combined_score': combined_score,
                    'metrics': metrics
                })
                
                logger.info(f"✅ {symbol}: Upside={upside:.1f}%, Quality={quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Filter results - ONLY positive upside
        filtered_results = [
            r for r in detailed_results 
            if r['upside'] > 0 and r['quality_score'] > 0.3
        ]
        
        # Sort by combined score
        filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # If no good stocks, get least bad
        if len(filtered_results) == 0:
            filtered_results = sorted(detailed_results, key=lambda x: x['combined_score'], reverse=True)[:3]
        
        # Format top 3
        top_stocks = []
        for stock in filtered_results[:3]:
            top_stocks.append({
                "symbol": stock['symbol'],
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside'], 1),
                    "confidence_score": int(stock['confidence'] * 100),
                    "sentiment_score": round(abs(stock['sentiment']), 2),
                    "ml_score": round(stock['quality_score'], 3)
                },
                "analysis_details": {
                    "fundamentals": {
                        "pe_ratio": stock['metrics'].get('pe_ratio', 0),
                        "peg_ratio": stock['metrics'].get('peg_ratio', 0),
                        "roe": stock['metrics'].get('roe', 0),
                        "profit_margin": stock['metrics'].get('profit_margin', 0),
                        "revenue_growth": stock['metrics'].get('revenue_growth', 0),
                        "debt_to_equity": stock['metrics'].get('debt_to_equity', 0)
                    },
                    "technicals": {
                        "rsi": stock['metrics'].get('rsi', 50),
                        "momentum_20d": stock['metrics'].get('momentum_20d', 0),
                        "momentum_60d": stock['metrics'].get('momentum_60d', 0),
                        "volatility": stock['metrics'].get('volatility', 0)
                    },
                    "valuation_methods": stock['metrics'].get('valuation_details', {}),
                    "ml_prediction": stock['ml_prediction'],
                    "quality_score": stock['quality_score']
                }
            })
        
        return JSONResponse(
            content={
                "status": "completed",
                "analysis_type": request.analysis_type,
                "target": request.target,
                "results": {
                    "top_stocks": top_stocks,
                    "market_conditions": {
                        "regime": sector_analysis.get('market_regime', 'Unknown'),
                        "adjustment_factor": market_adjustment
                    },
                    "sector_analysis": sector_analysis,
                    "total_analyzed": len(detailed_results),
                    "total_qualified": len(filtered_results)
                },
                "ml_powered": True
            },
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)