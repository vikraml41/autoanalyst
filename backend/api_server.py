"""
FastAPI Server for Stock Analysis
Wraps the backend.py stock analyzer and provides REST API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from backend import StockAnalyzer, DataFetchError, FMP_API_KEY
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Analyzer API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StockAnalysisRequest(BaseModel):
    ticker: str


@app.post('/api/stock-analysis')
async def analyze_stock(request: StockAnalysisRequest):
    """
    Analyze a stock using DCF, Revenue Forecasting, and Comparable Companies

    Request JSON:
    {
        "ticker": "AAPL"
    }

    Response JSON:
    Complete analysis results including DCF, Revenue Forecast, Comps, and ML Synthesis
    """
    try:
        ticker = request.ticker.strip().upper()

        if not ticker:
            raise HTTPException(status_code=400, detail='Ticker symbol is required')

        # Check if FMP API key is configured
        if not FMP_API_KEY or FMP_API_KEY == 'demo':
            raise HTTPException(
                status_code=500,
                detail='FMP_API_KEY not configured. Please set the FMP_API_KEY environment variable on Render.'
            )

        logger.info(f"Starting analysis for {ticker} (FMP API key configured: {len(FMP_API_KEY)} chars)")

        # Create analyzer and run complete analysis
        analyzer = StockAnalyzer(ticker)
        results = analyzer.analyze()

        if not results:
            raise HTTPException(status_code=500, detail='Analysis failed - unable to retrieve data')

        logger.info(f"Analysis completed successfully for {ticker}")

        return results

    except HTTPException:
        raise
    except DataFetchError as e:
        logger.error(f"Data fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {str(e)}")
    except Exception as e:
        logger.error(f"Error analyzing stock: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get('/api/health')
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'stock-analyzer',
        'fmp_api_configured': bool(FMP_API_KEY and FMP_API_KEY != 'demo'),
        'fmp_api_key_length': len(FMP_API_KEY) if FMP_API_KEY else 0
    }


@app.get('/api/market-conditions')
async def market_conditions():
    """
    Get current market conditions (placeholder for compatibility)
    """
    return {
        'regime': 'Growth',
        'fed_stance': 'Neutral',
        'vix': 15.5,
        'recession_risk': 'Low'
    }


@app.get('/api/test-fmp/{ticker}')
async def test_fmp(ticker: str):
    """Debug endpoint to test FMP API - tries multiple endpoints"""
    import requests

    ticker = ticker.upper()
    base = "https://financialmodelingprep.com/api/v3"
    results = {}

    # Test multiple endpoints to see what works
    endpoints = [
        f"quote/{ticker}",
        f"quote-short/{ticker}",
        f"profile/{ticker}",
        f"stock-price-change/{ticker}",
        f"historical-price-full/{ticker}?serietype=line",
    ]

    for endpoint in endpoints:
        url = f"{base}/{endpoint}"
        if "?" in endpoint:
            url += f"&apikey={FMP_API_KEY}"
        else:
            url += f"?apikey={FMP_API_KEY}"

        try:
            response = requests.get(url, timeout=10)
            results[endpoint.split('/')[0]] = {
                'status': response.status_code,
                'response_preview': str(response.json())[:200]
            }
        except Exception as e:
            results[endpoint.split('/')[0]] = {'error': str(e)}

    return {
        'api_key_length': len(FMP_API_KEY) if FMP_API_KEY else 0,
        'api_key_preview': FMP_API_KEY[:8] + '...' if FMP_API_KEY else None,
        'endpoints_tested': results
    }


if __name__ == '__main__':
    import uvicorn
    import os

    # Get port from environment variable (for Render.com) or default to 8000
    port = int(os.environ.get('PORT', 8000))

    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run("api_server:app", host='0.0.0.0', port=port, log_level="info", reload=False)
