"""
FastAPI Server for Stock Analysis
Wraps the backend.py stock analyzer and provides REST API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from backend import StockAnalyzer
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

        logger.info(f"Starting analysis for {ticker}")

        # Create analyzer and run complete analysis
        analyzer = StockAnalyzer(ticker)
        results = analyzer.analyze()

        if not results:
            raise HTTPException(status_code=500, detail='Analysis failed - unable to retrieve data')

        logger.info(f"Analysis completed successfully for {ticker}")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing stock: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/health')
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'service': 'stock-analyzer'}


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


if __name__ == '__main__':
    import uvicorn
    import os

    # Get port from environment variable (for Render.com) or default to 8000
    port = int(os.environ.get('PORT', 8000))

    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run("api_server:app", host='0.0.0.0', port=port, log_level="info", reload=False)
