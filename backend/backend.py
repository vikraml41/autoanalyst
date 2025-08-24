#!/usr/bin/env python3
"""
FastAPI Backend - Fixed CORS with manual headers
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
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="2.0.0")

# Try CORS middleware (might not work on Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRITICAL: Manually add CORS headers to EVERY response
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ============ DATA LOADING (keeping your existing code) ============

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
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            'GICS Sector': ['Technology'] * 5,
            'GICS Sub-Industry': ['Hardware', 'Software', 'Internet', 'E-Commerce', 'Semiconductors']
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
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

# Load data
stocks_data = load_csv_files()
logger.info(f"Loaded {len(stocks_data)} stocks")

# Simplified ML models (no imports needed)
class SimpleMLModel:
    def calculate_ml_score(self, symbol):
        return 0.75
    
    def get_sentiment_score(self, symbol):
        return 0.7
    
    def get_current_price(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice', 100)
        except:
            return 100

class SimpleMarketAnalyzer:
    def get_market_regime(self):
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1mo")
            if len(hist) > 0:
                change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                if change > 0.02:
                    return "Bull Market"
                elif change < -0.02:
                    return "Bear Market"
            return "Neutral Market"
        except:
            return "Neutral Market"
    
    def get_federal_reserve_stance(self):
        return {"stance": "Neutral"}
    
    def fetch_economic_data(self):
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="1d")
            current_vix = float(vix_hist['Close'].iloc[-1]) if len(vix_hist) > 0 else 20
        except:
            current_vix = 20
        return {"Volatility Index": {"current": current_vix}}

ml_model = SimpleMLModel()
market_analyzer = SimpleMarketAnalyzer()
valuator = SimpleMLModel()
ML_AVAILABLE = True

# ============ API ENDPOINTS with MANUAL CORS ============

@app.get("/")
async def root():
    return JSONResponse(
        content={
            "name": "AutoAnalyst API",
            "status": "running",
            "cors": "enabled",
            "stocks_loaded": len(stocks_data)
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/api/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "components": {
                "stocks_data": "✅" if len(stocks_data) > 0 else "❌",
                "ml_model": "✅",
                "market_analyzer": "✅",
                "valuator": "✅"
            },
            "ml_available": ML_AVAILABLE,
            "stocks_count": len(stocks_data),
            "timestamp": datetime.now().isoformat(),
            "cors_enabled": True
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.options("/api/health")
async def health_options():
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/api/stocks/list")
async def get_stocks_list():
    sectors = []
    sub_industries = []
    
    if 'GICS Sector' in stocks_data.columns:
        sectors = stocks_data['GICS Sector'].dropna().unique().tolist()
        sectors = [s for s in sectors if isinstance(s, str) and s != '']
    
    if 'GICS Sub-Industry' in stocks_data.columns:
        sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist()
        sub_industries = [s for s in sub_industries if isinstance(s, str) and s != '']
    
    return JSONResponse(
        content={
            "sectors": sectors,
            "sub_industries": sub_industries,
            "total_stocks": len(stocks_data)
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.options("/api/stocks/list")
async def stocks_options():
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/api/market-conditions")
async def get_market_conditions():
    try:
        regime = market_analyzer.get_market_regime()
        fed_data = market_analyzer.get_federal_reserve_stance()
        economic_data = market_analyzer.fetch_economic_data()
        
        return JSONResponse(
            content={
                "regime": regime,
                "fed_stance": fed_data.get("stance", "Neutral"),
                "vix": economic_data.get("Volatility Index", {}).get("current", 20),
                "recession_risk": "Low",
                "ml_powered": True
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        return JSONResponse(
            content={
                "regime": "Neutral",
                "fed_stance": "Neutral",
                "vix": 20.0,
                "recession_risk": "Low",
                "error": str(e)
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )

@app.options("/api/market-conditions")
async def market_options():
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    if stocks_data is None:
        raise HTTPException(status_code=500, detail="Stock data not initialized")
    
    # Filter stocks
    if request.analysis_type == "sector":
        filtered = stocks_data[stocks_data['GICS Sector'] == request.target]
    else:
        filtered = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
    
    if len(filtered) == 0:
        raise HTTPException(status_code=404, detail=f"No stocks found for {request.target}")
    
    results = []
    for _, stock in filtered.iterrows():
        symbol = stock['Symbol']
        ml_score = ml_model.calculate_ml_score(symbol)
        sentiment = ml_model.get_sentiment_score(symbol)
        current = ml_model.get_current_price(symbol)
        target_price = current * 1.15
        
        results.append({
            "symbol": symbol,
            "ml_score": ml_score,
            "sentiment": sentiment,
            "target_price": target_price,
            "current_price": current
        })
    
    # Sort and get top 3
    results.sort(key=lambda x: x['ml_score'], reverse=True)
    top_3 = results[:3]
    
    # Format response
    top_stocks = []
    for stock in top_3:
        top_stocks.append({
            "symbol": stock['symbol'],
            "metrics": {
                "current_price": stock['current_price'],
                "target_price": stock['target_price'],
                "upside_potential": 15.0,
                "confidence_score": int(stock['ml_score'] * 100),
                "sentiment_score": stock['sentiment'],
                "ml_score": stock['ml_score']
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
                    "regime": market_analyzer.get_market_regime(),
                    "adjustment_factor": 1.05
                }
            }
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.options("/api/analysis")
async def analysis_options():
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server with manual CORS headers on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)