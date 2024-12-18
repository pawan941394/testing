
import warnings
warnings.filterwarnings("ignore")
import spacy
import requests
from fuzzywuzzy import process

import warnings
warnings.filterwarnings("ignore")

import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union
import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
import yfinance as yf

import requests
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re
import json
from typing import Dict, Any
from langchain.agents.output_parsers.react_single_input import OutputParserException
# Required libraries for historical employee count
import faiss  # For similarity search
import numpy as np  # For numerical operations
from sentence_transformers import SentenceTransformer  # For embeddings
import pickle  # For loading PKL file



"""# Validator"""

class ValidatorAgent:
    @staticmethod
    def validate_employee_count_response(response):
        """
        Validate employee count data response
        """
        try:
            # Check if response is an error message string
            if isinstance(response, str):
                return {
                    "status": False,
                    "message": response,
                    "data": None
                }

            # Check if data is empty
            if not response or not response.get('data'):
                return {
                    "status": False,
                    "message": "No employee count data available",
                    "data": None
                }

            data = response['data']

            # Define required fields and their types
            required_fields = {
                'employeeCount': dict,
                'periodOfReport': dict,
                'companyName': dict,
                'source': dict
            }

            # Check if all required fields exist and have correct types
            for field, field_type in required_fields.items():
                if field not in data:
                    return {
                        "status": False,
                        "message": f"Missing required field: {field}",
                        "data": None
                    }
                if not isinstance(data[field], field_type):
                    return {
                        "status": False,
                        "message": f"Invalid type for field {field}. Expected {field_type.__name__}",
                        "data": None
                    }

            # Validate employee count data
            try:
                latest_count = int(data['employeeCount']['0'])
                previous_count = int(data['employeeCount']['1'])

                # Basic sanity checks
                if latest_count <= 0 or previous_count <= 0:
                    return {
                        "status": False,
                        "message": "Invalid employee count values (must be positive)",
                        "data": None
                    }

                # Check for unreasonable changes (e.g., more than 500% change)
                change_pct = abs((latest_count - previous_count) / previous_count * 100)
                if change_pct > 500:
                    return {
                        "status": False,
                        "message": f"Suspicious change in employee count ({change_pct:.1f}% change)",
                        "data": None
                    }

            except (KeyError, ValueError, ZeroDivisionError) as e:
                return {
                    "status": False,
                    "message": f"Error processing employee count values: {str(e)}",
                    "data": None
                }

            # Validate dates
            try:
                latest_date = data['periodOfReport']['0']
                if not latest_date or not isinstance(latest_date, str):
                    return {
                        "status": False,
                        "message": "Invalid reporting date format",
                        "data": None
                    }
            except KeyError:
                return {
                    "status": False,
                    "message": "Missing reporting date",
                    "data": None
                }

            return {
                "status": True,
                "message": "Valid employee count data",
                "data": data
            }

        except Exception as e:
            return {
                "status": False,
                "message": f"Validation error: {str(e)}",
                "data": None
            }

    @staticmethod
    def format_employee_count_error(validation_result):
        """
        Format error messages for employee count data
        """
        return f"""
        Unable to retrieve employee count data:
        Error: {validation_result['message']}
        Please ensure the data meets the following requirements:
        - Valid company ticker symbol
        - Available employee count data
        - Reasonable employee count values
        - Valid reporting dates
        """




    @staticmethod
    def validate_company_notes_response(response):
        """
        Validate company notes data
        """
        try:
            # Check if response is an error message string
            if isinstance(response, str):
                return {
                    "status": False,
                    "message": response,
                    "data": None
                }

            # Check if data is empty or not a list
            if not response or not isinstance(response, list):
                return {
                    "status": False,
                    "message": "No company notes data available",
                    "data": None
                }

            # Define required fields for each note
            required_fields = ['cik', 'symbol', 'title', 'exchange']

            # Validate each note in the response
            validated_notes = []
            for note in response:
                # Check for required fields
                missing_fields = [field for field in required_fields if field not in note]
                if missing_fields:
                    continue

                # Additional validations
                if not note.get('title') or 'Notes due' not in note.get('title', ''):
                    continue

                validated_notes.append(note)

            if not validated_notes:
                return {
                    "status": False,
                    "message": "No valid company notes found after validation",
                    "data": None
                }

            return {
                "status": True,
                "message": "Valid company notes data",
                "data": validated_notes
            }

        except Exception as e:
            return {
                "status": False,
                "message": f"Validation error: {str(e)}",
                "data": None
            }
    @staticmethod
    def validate_stock_price_response(response: Dict[str, Any]):
        """
        Validate stock price data
        """
        try:
            # Check if response is an error message string
            if isinstance(response, str):
                if "Unable to" in response or "error" in response.lower():
                    return {
                        "status": False,
                        "message": response,
                        "data": None
                    }
                return {
                    "status": True,
                    "message": "Valid stock price data",
                    "data": response
                }

            return {
                "status": True,
                "message": "Valid stock price data",
                "data": response
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"Validation error: {str(e)}",
                "data": None
            }

    @staticmethod
    def format_error_message(error_data: Dict[str, Any]) -> str:
        """
        Format error messages for user display
        """
        return f"""
        Unable to retrieve requested data:
        Error: {error_data['message']}
        Please try again or check the ticker symbol.
        """


    @staticmethod
    def validate_key_metrics_response(response):
        """
        Validate key metrics data
        """
        try:
            if isinstance(response, str) and "Error" in response:
                return {
                    "status": False,
                    "message": response,
                    "data": None
                }

            # Check if data is empty
            if not response:
                return {
                    "status": False,
                    "message": "No key metrics data available",
                    "data": None
                }

            # Check if required metrics exist
            required_metrics = ['Market Cap', 'PE Ratio', 'ROE']
            if not all(metric in response[0] for metric in required_metrics):
                return {
                    "status": False,
                    "message": "Missing required metrics in data",
                    "data": None
                }

            return {
                "status": True,
                "message": "Valid key metrics data",
                "data": response
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"Validation error: {str(e)}",
                "data": None
            }


    @staticmethod
    def validate_cash_flow_statement_response(response):
        """
        Validate cash flow statement data
        """
        try:
            # Check if response is an error message string
            if isinstance(response, str):
                return {
                    "status": False,
                    "message": response,
                    "data": None
                }

            # Check if data is empty or not a list
            if not response or not isinstance(response, list):
                return {
                    "status": False,
                    "message": "No cash flow statement data available",
                    "data": None
                }

            # Get the latest data entry
            latest_data = response[0]

            # Define required fields
            required_fields = [
                'date', 'symbol', 'netIncome', 'netCashProvidedByOperatingActivities',
                'investmentsInPropertyPlantAndEquipment', 'netCashUsedForInvestingActivites',
                'netCashUsedProvidedByFinancingActivities', 'netChangeInCash',
                'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow',
                'freeCashFlow'
            ]

            # Check for required fields
            missing_fields = [field for field in required_fields if field not in latest_data]
            if missing_fields:
                return {
                    "status": False,
                    "message": f"Missing required fields: {', '.join(missing_fields)}",
                    "data": None
                }

            # Validate numeric fields
            numeric_fields = [
                'netIncome', 'netCashProvidedByOperatingActivities',
                'investmentsInPropertyPlantAndEquipment', 'netCashUsedForInvestingActivites',
                'netCashUsedProvidedByFinancingActivities', 'netChangeInCash',
                'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow',
                'freeCashFlow'
            ]

            for field in numeric_fields:
                if not isinstance(latest_data[field], (int, float)):
                    return {
                        "status": False,
                        "message": f"Invalid numeric value for {field}",
                        "data": None
                    }

            # Validate cash flow reconciliation
            if abs(latest_data['cashAtEndOfPeriod'] - latest_data['cashAtBeginningOfPeriod']
                  - latest_data['netChangeInCash']) > 1:  # Allow for rounding differences
                return {
                    "status": False,
                    "message": "Cash flow doesn't reconcile",
                    "data": None
                }

            return {
                "status": True,
                "message": "Valid cash flow statement data",
                "data": latest_data
            }

        except Exception as e:
            return {
                "status": False,
                "message": f"Validation error: {str(e)}",
                "data": None
            }


    @staticmethod
    def validate_executive_compensation_response(response):
        """
        Validate executive compensation data
        """
        try:
            if isinstance(response, str) and "Error" in response:
                return {
                    "status": False,
                    "message": response,
                    "data": None
                }

            # Check for required compensation fields
            required_fields = ['salary', 'total', 'nameAndPosition', 'year']
            if not all(field in response for field in required_fields):
                return {
                    "status": False,
                    "message": "Missing required compensation fields",
                    "data": None
                }

            return {
                "status": True,
                "message": "Valid compensation data",
                "data": response
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"Validation error: {str(e)}",
                "data": None
            }

    @staticmethod
    def format_error_message(error_data):
        """
        Format error messages for user display
        """
        return f"""
        Unable to retrieve requested data:
        Error: {error_data['message']}
        Please try again or contact support if the issue persists.
        """


    @staticmethod
    def validate_income_statement_response(response):
        """
        Validate income statement data
        """
        try:
            # Check if response is an error message string
            if isinstance(response, str):
                if "Error" in response or "Unable to" in response.lower():
                    return {
                        "status": False,
                        "message": response,
                        "data": None
                    }

            # Check if data is empty or not a list
            if not response or not isinstance(response, list):
                return {
                    "status": False,
                    "message": "No income statement data available",
                    "data": None
                }

            # Get the latest data entry
            latest_data = response[0]

            # Define required fields for income statement
            required_fields = [
                'date', 'symbol', 'reportedCurrency', 'revenue',
                'costOfRevenue', 'grossProfit', 'grossProfitRatio',
                'researchAndDevelopmentExpenses',
                'sellingGeneralAndAdministrativeExpenses',
                'operatingExpenses', 'operatingIncome', 'operatingIncomeRatio',
                'ebitda', 'ebitdaratio', 'netIncome', 'netIncomeRatio',
                'eps', 'epsdiluted'
            ]

            # Check for required fields
            missing_fields = [field for field in required_fields if field not in latest_data]
            if missing_fields:
                return {
                    "status": False,
                    "message": f"Missing required fields: {', '.join(missing_fields)}",
                    "data": None
                }

            # Validate numeric fields
            numeric_fields = [
                'revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio',
                'operatingExpenses', 'operatingIncome', 'operatingIncomeRatio',
                'ebitda', 'ebitdaratio', 'netIncome', 'netIncomeRatio',
                'eps', 'epsdiluted'
            ]

            for field in numeric_fields:
                if not isinstance(latest_data[field], (int, float)):
                    return {
                        "status": False,
                        "message": f"Invalid numeric value for {field}",
                        "data": None
                    }

            # Validate ratios
            ratio_fields = ['grossProfitRatio', 'operatingIncomeRatio', 'ebitdaratio', 'netIncomeRatio']
            for field in ratio_fields:
                if not (0 <= latest_data[field] <= 1):
                    return {
                        "status": False,
                        "message": f"Invalid ratio value for {field}",
                        "data": None
                    }

            return {
                "status": True,
                "message": "Valid income statement data",
                "data": latest_data
            }

        except Exception as e:
            return {
                "status": False,
                "message": f"Validation error: {str(e)}",
                "data": None
            }

    @staticmethod
    def validate_balance_sheet_response(response):
        """
        Validate balance sheet data with the complete set of fields
        """
        try:
            # Check if response is an error message string
            if isinstance(response, str):
                if "Error" in response or "Unable to" in response.lower():
                    return {
                        "status": False,
                        "message": response,
                        "data": None
                    }

            # Check if data is empty or not a list
            if not response or not isinstance(response, list):
                return {
                    "status": False,
                    "message": "No balance sheet data available",
                    "data": None
                }

            # Get the latest data entry (first item in the list)
            latest_data = response[0]

            # Define required fields for balance sheet
            required_fields = [
                'date', 'symbol', 'reportedCurrency',
                'cashAndCashEquivalents', 'shortTermInvestments',
                'cashAndShortTermInvestments', 'netReceivables',
                'inventory', 'otherCurrentAssets', 'totalCurrentAssets',
                'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets',
                'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets',
                'totalNonCurrentAssets', 'totalAssets', 'accountPayables',
                'shortTermDebt', 'taxPayables', 'deferredRevenue',
                'otherCurrentLiabilities', 'totalCurrentLiabilities',
                'longTermDebt', 'totalNonCurrentLiabilities', 'totalLiabilities',
                'commonStock', 'retainedEarnings', 'totalStockholdersEquity',
                'totalEquity', 'totalLiabilitiesAndStockholdersEquity'
            ]

            # Check for required fields
            missing_fields = [field for field in required_fields if field not in latest_data]
            if missing_fields:
                return {
                    "status": False,
                    "message": f"Missing required fields: {', '.join(missing_fields)}",
                    "data": None
                }

            # Validate numeric fields (all fields except date, symbol, reportedCurrency)
            numeric_fields = [f for f in required_fields if f not in ['date', 'symbol', 'reportedCurrency']]
            for field in numeric_fields:
                if not isinstance(latest_data[field], (int, float)):
                    return {
                        "status": False,
                        "message": f"Invalid numeric value for {field}",
                        "data": None
                    }

            # Validate basic accounting equation
            if abs(latest_data['totalAssets'] - (latest_data['totalLiabilities'] + latest_data['totalEquity'])) > 1:
                return {
                    "status": False,
                    "message": "Balance sheet equation doesn't balance",
                    "data": None
                }

            return {
                "status": True,
                "message": "Valid balance sheet data",
                "data": latest_data
            }

        except Exception as e:
            return {
                "status": False,
                "message": f"Validation error: {str(e)}",
                "data": None
            }

    @staticmethod
    def validate_ticker_input(input_str: str) -> Dict[str, Any]:
        """
        Validate the ticker input string
        """
        try:
            if not input_str:
                return {
                    "status": False,
                    "message": "Empty input provided",
                    "data": None
                }

            # Handle JSON-like string
            if '{' in input_str:
                try:
                    input_str = input_str.replace("'", '"')
                    input_dict = json.loads(input_str)
                    ticker = input_dict.get('ticker', '').strip('"')
                except json.JSONDecodeError:
                    # Fallback to regex if JSON parsing fails
                    match = re.search(r'ticker["\s:]+([A-Z]+)', input_str)
                    ticker = match.group(1) if match else None
            else:
                ticker = input_str.strip().upper()

            if not ticker:
                return {
                    "status": False,
                    "message": "No valid ticker symbol found in input",
                    "data": None
                }

            # Validate ticker format
            if not re.match(r'^[A-Z]{1,5}$', ticker):
                return {
                    "status": False,
                    "message": "Invalid ticker format",
                    "data": None
                }

            return {
                "status": True,
                "message": "Valid ticker input",
                "data": ticker
            }

        except Exception as e:
            return {
                "status": False,
                "message": f"Input validation error: {str(e)}",
                "data": None
            }

@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time
    """
    return datetime.now().strftime(format)

# Import the necessary library
import yfinance as yf
from datetime import datetime
from langchain_core.tools import tool

class CustomOutputParser(AgentOutputParser):
  def parse_input(input_str):
    """
    Simple parsing to extract ticker from input
    """
    try:
        # If input is a dictionary or dict-like
        if isinstance(input_str, dict):
            return input_str.get('ticker')

        # If input is a string
        if isinstance(input_str, str):
            # Try to extract ticker using simple regex
            match = re.search(r'[\'"]?ticker[\'"]?\s*[:=]\s*[\'"]?([A-Z]+)[\'"]?', input_str, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    except Exception as e:
        print(f"Input parsing error: {e}")
        return None

"""# Metrics , compansanation , price"""

@tool
def get_stock_price(ticker: str):
    """
    Get the live stock price with robust ticker extraction and validation
    """
    print(f"Raw ticker input: {repr(ticker)}")
    print(f"Type of ticker: {type(ticker)}")

    def extract_ticker(raw_input):
        """
        Extract clean ticker symbol with multiple strategies
        """
        if isinstance(raw_input, str):
            # Strategy 1: Extract between quotes after TICKER
            match = re.search(r'TICKER[=:]\s*["\'](.*?)["\']', raw_input)
            if match:
                return match.group(1).upper().strip()

            # Strategy 2: Extract first word matching stock ticker pattern
            match = re.search(r'\b([A-Z]{1,5})\b', raw_input.upper())
            if match:
                return match.group(1)

        # Fallback: convert to string and clean
        return str(raw_input).upper().strip()

    def get_price_with_fallbacks(clean_ticker):
        """
        Multiple methods to retrieve stock price
        """
        try:
            # Method 1: Direct yfinance history
            stock = yf.Ticker(clean_ticker)

            # Try history method
            history = stock.history(period="1d")
            if not history.empty:
                return history['Close'][-1]

            # Method 2: Stock info
            if stock.info and 'currentPrice' in stock.info:
                return stock.info['currentPrice']

            # Method 3: Quote type information
            quote_type = stock.info.get('quoteType')
            if quote_type == 'EQUITY':
                return None

        except Exception as e:
            print(f"Price retrieval error for {clean_ticker}: {e}")
            return None

        return None

    try:
        # Extract clean ticker
        clean_ticker = extract_ticker(ticker)

        if not clean_ticker:
            response = "Unable to extract valid ticker symbol"
        else:
            # Retrieve price
            price = get_price_with_fallbacks(clean_ticker)

            if price is not None:
                response = f"The current price of {clean_ticker} is {price:.2f} USD"
            else:
                response = f"Unable to retrieve price for {clean_ticker}"

        # Validate the response using ValidatorAgent
        validation_result = ValidatorAgent.validate_stock_price_response(response)

        if not validation_result["status"]:
            return ValidatorAgent.format_error_message(validation_result)

        return validation_result["data"]

    except Exception as e:
        error_response = f"Comprehensive error retrieving stock price: {str(e)}"
        validation_result = ValidatorAgent.validate_stock_price_response(error_response)
        return ValidatorAgent.format_error_message(validation_result)

import re
import json
import requests

def extract_ticker(input_str):
    """
    Robust ticker extraction with multiple strategies
    """
    # Print debug information
    print(f"DEBUG: Raw input received: {repr(input_str)}")

    # Strategy 1: Direct dictionary extraction
    if isinstance(input_str, dict):
        return input_str.get('ticker')

    # Strategy 2: String parsing with multiple regex patterns
    ticker_patterns = [
        r"'ticker':\s*['\"]*([A-Z]+)['\"]*",   # Matches 'ticker':'AAPL'
        r'"ticker":\s*["\']*([A-Z]+)["\']*',   # Matches "ticker":"AAPL"
        r'ticker[=:]\s*["\']*([A-Z]+)["\']*',  # Matches ticker="AAPL" or ticker=AAPL
    ]

    for pattern in ticker_patterns:
        match = re.search(pattern, str(input_str), re.IGNORECASE)
        if match:
            ticker = match.group(1)
            print(f"DEBUG: Ticker extracted: {ticker}")
            return ticker

    # Fallback: Manual extraction
    manual_extract = re.findall(r'[A-Z]{1,5}', str(input_str))
    if manual_extract:
        print(f"DEBUG: Manually extracted ticker: {manual_extract[0]}")
        return manual_extract[0]

    print(f"ERROR: Unable to extract ticker from input: {repr(input_str)}")
    return None

@tool
def get_key_metrics_and_analyze(ticker: str, query: str = None):
    """
    Get comprehensive key metrics with robust error handling
    """
    # Ensure ticker is correctly extracted
    if isinstance(ticker, str):
        ticker = extract_ticker(ticker)

    if not ticker:
        validation_result = ValidatorAgent.validate_key_metrics_response(None)
        return ValidatorAgent.format_error_message(validation_result)

    try:
        # Use Financial Modeling Prep API
        api_key = "uW7kG6fZagauUpoJowIRofnHJeCZbE7V"
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=annual&limit=1&apikey={api_key}"

        # Fetch data with error handling
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            error_response = f"Error fetching metrics for {ticker}. Status code: {response.status_code}"
            validation_result = ValidatorAgent.validate_key_metrics_response(error_response)
            return ValidatorAgent.format_error_message(validation_result)

        data = response.json()

        if not data:
            validation_result = ValidatorAgent.validate_key_metrics_response(None)
            return ValidatorAgent.format_error_message(validation_result)

        # Process and format metrics
        key_metrics = []
        for entry in data:
            metrics = {
                'Date': entry.get('date', 'N/A'),
                'Market Cap': f"${entry.get('marketCap', 'N/A'):,}",
                'PE Ratio': entry.get('peRatio', 'N/A'),
                'Price to Book': entry.get('pbRatio', 'N/A'),
                'ROE': f"{entry.get('roe', 'N/A'):.2f}%" if isinstance(entry.get('roe'), (int, float)) else 'N/A',
                'ROA': f"{entry.get('roa', 'N/A'):.2f}%" if isinstance(entry.get('roa'), (int, float)) else 'N/A',
                'Debt to Equity': entry.get('debtToEquity', 'N/A'),
                'Current Ratio': entry.get('currentRatio', 'N/A')
            }
            key_metrics.append(metrics)

        # Validate the formatted metrics
        validation_result = ValidatorAgent.validate_key_metrics_response(key_metrics)
        if not validation_result["status"]:
            return ValidatorAgent.format_error_message(validation_result)

        # Optional query-based filtering
        if query:
            query_terms = [q.strip().lower() for q in query.split(',')]
            filtered_metrics = []
            for metric_entry in key_metrics:
                filtered_entry = {k: v for k, v in metric_entry.items()
                                if any(term in k.lower() for term in query_terms)}
                filtered_metrics.append(filtered_entry)

            # Validate filtered results
            validation_result = ValidatorAgent.validate_key_metrics_response(filtered_metrics)
            if not validation_result["status"]:
                return ValidatorAgent.format_error_message(validation_result)

            return validation_result["data"]

        return validation_result["data"]

    except requests.exceptions.RequestException as req_error:
        error_response = f"Network error fetching metrics for {ticker}"
        validation_result = ValidatorAgent.validate_key_metrics_response(error_response)
        return ValidatorAgent.format_error_message(validation_result)

    except Exception as e:
        error_response = f"Error retrieving metrics for {ticker}: {e}"
        validation_result = ValidatorAgent.validate_key_metrics_response(error_response)
        return ValidatorAgent.format_error_message(validation_result)

@tool
def get_executive_compensation(ticker: str):
    """
    Retrieve and format the most recent executive compensation data for a given company
    """
    # Ensure ticker is correctly extracted
    if isinstance(ticker, str):
        ticker = extract_ticker(ticker)

    if not ticker:
        validation_result = ValidatorAgent.validate_executive_compensation_response({
            "error": "Unable to extract valid ticker symbol"
        })
        return ValidatorAgent.format_error_message(validation_result)

    try:
        # API Configuration
        api_key = "uW7kG6fZagauUpoJowIRofnHJeCZbE7V"
        compensation_url = f"https://financialmodelingprep.com/api/v4/governance/executive_compensation?symbol={ticker}&apikey={api_key}"

        # Make API Request
        response = requests.get(compensation_url, timeout=10)

        # Check response
        if response.status_code != 200:
            error_response = f"Error: Received status code {response.status_code}. Unable to fetch compensation data."
            validation_result = ValidatorAgent.validate_executive_compensation_response({
                "error": error_response
            })
            return ValidatorAgent.format_error_message(validation_result)

        data = response.json()

        # Validate data
        if not data:
            validation_result = ValidatorAgent.validate_executive_compensation_response({
                "error": f"No executive compensation data found for {ticker}."
            })
            return ValidatorAgent.format_error_message(validation_result)

        # Get the most recent entry (first in the list)
        latest_entry = data[0]

        # Validate the latest entry
        validation_result = ValidatorAgent.validate_executive_compensation_response(latest_entry)
        if not validation_result["status"]:
            return ValidatorAgent.format_error_message(validation_result)

        # Format the compensation details
        compensation_details = {
            "companyName": latest_entry['companyName'],
            "nameAndPosition": latest_entry['nameAndPosition'],
            "year": latest_entry['year'],
            "salary": latest_entry['salary'],
            "bonus": latest_entry['bonus'],
            "stock_award": latest_entry['stock_award'],
            "option_award": latest_entry['option_award'],
            "incentive_plan_compensation": latest_entry['incentive_plan_compensation'],
            "all_other_compensation": latest_entry['all_other_compensation'],
            "total": latest_entry['total'],
            "filingDate": latest_entry['filingDate'],
            "acceptedDate": latest_entry['acceptedDate'],
            "url": latest_entry['url']
        }

        # Validate formatted details
        validation_result = ValidatorAgent.validate_executive_compensation_response(compensation_details)
        if not validation_result["status"]:
            return ValidatorAgent.format_error_message(validation_result)

        # Format the final output
        formatted_output = f"""
**Executive Compensation Details for {compensation_details['companyName']}**

**Executive:** {compensation_details['nameAndPosition']}
**Year:** {compensation_details['year']}

**Compensation Breakdown:**
- **Salary:** ${compensation_details['salary']:,}
- **Bonus:** ${compensation_details['bonus']:,}
- **Stock Awards:** ${compensation_details['stock_award']:,}
- **Option Awards:** ${compensation_details['option_award']:,}
- **Incentive Plan Compensation:** ${compensation_details['incentive_plan_compensation']:,}
- **Other Compensation:** ${compensation_details['all_other_compensation']:,}

**Total Compensation:** ${compensation_details['total']:,}

**SEC Filing Details:**
- **Filing Date:** {compensation_details['filingDate']}
- **Accepted Date:** {compensation_details['acceptedDate']}

**SEC Filing Link:** [View Full Filing]({compensation_details['url']})
"""

        return formatted_output

    except requests.exceptions.RequestException as req_error:
        validation_result = ValidatorAgent.validate_executive_compensation_response({
            "error": f"Network error fetching compensation data: {req_error}"
        })
        return ValidatorAgent.format_error_message(validation_result)

    except Exception as e:
        validation_result = ValidatorAgent.validate_executive_compensation_response({
            "error": f"Error retrieving compensation data: {e}"
        })
        return ValidatorAgent.format_error_message(validation_result)



"""# Income statement tool"""

@tool
def get_income_statement(input_str: str):
    """
    Fetch the detailed income statement for a given ticker symbol
    """
    # Validate input
    validation_result = ValidatorAgent.validate_ticker_input(input_str)
    if not validation_result["status"]:
        return ValidatorAgent.format_error_message(validation_result)

    ticker = validation_result["data"]

    try:
        # API request
        api_key = "LDi3hXVABun1oUg77fZPdXU6UjiSsJ0c"
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&apikey={api_key}"

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return ValidatorAgent.format_error_message({
                "status": False,
                "message": f"API request failed with status code {response.status_code}"
            })

        data = response.json()

        # Validate response data
        validation_result = ValidatorAgent.validate_income_statement_response(data)
        if not validation_result["status"]:
            return ValidatorAgent.format_error_message(validation_result)

        latest_data = validation_result["data"]

        # Format the output (your existing formatting code)
        formatted_output = f"""Action: get_income_statement
Content: Income Statement Analysis for {ticker}
...
"""
        return formatted_output

    except requests.RequestException as e:
        return ValidatorAgent.format_error_message({
            "status": False,
            "message": f"Network error occurred: {str(e)}"
        })
    except Exception as e:
        return ValidatorAgent.format_error_message({
            "status": False,
            "message": f"Unexpected error: {str(e)}"
        })



"""# Balance Sheet"""

@tool
def get_balance_sheet(input_str: str):
    """
    Fetch and format the balance sheet for a given ticker symbol.
    Uses only the latest available data.
    """
    # Validate input
    validation_result = ValidatorAgent.validate_ticker_input(input_str)
    if not validation_result["status"]:
        return ValidatorAgent.format_error_message(validation_result)

    ticker = validation_result["data"]

    try:
        api_key = "LDi3hXVABun1oUg77fZPdXU6UjiSsJ0c"
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=annual&apikey={api_key}"

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return ValidatorAgent.format_error_message({
                "status": False,
                "message": f"API request failed with status code {response.status_code}"
            })

        data = response.json()

        # Validate response data
        validation_result = ValidatorAgent.validate_balance_sheet_response(data)
        if not validation_result["status"]:
            return ValidatorAgent.format_error_message(validation_result)

        latest_data = validation_result["data"]

        # Format the output with validated data
        formatted_output = f"""
**Balance Sheet for {latest_data['symbol']} ({latest_data['date']})**

**Assets**
Current Assets:
- Cash and Cash Equivalents: ${latest_data['cashAndCashEquivalents']:,}
- Short-term Investments: ${latest_data['shortTermInvestments']:,}
- Net Receivables: ${latest_data['netReceivables']:,}
- Inventory: ${latest_data['inventory']:,}
- Other Current Assets: ${latest_data['otherCurrentAssets']:,}
Total Current Assets: ${latest_data['totalCurrentAssets']:,}

Non-Current Assets:
- Property, Plant & Equipment: ${latest_data['propertyPlantEquipmentNet']:,}
- Long-term Investments: ${latest_data['longTermInvestments']:,}
- Tax Assets: ${latest_data['taxAssets']:,}
- Other Non-Current Assets: ${latest_data['otherNonCurrentAssets']:,}
Total Non-Current Assets: ${latest_data['totalNonCurrentAssets']:,}

Total Assets: ${latest_data['totalAssets']:,}

**Liabilities**
Current Liabilities:
- Accounts Payable: ${latest_data['accountPayables']:,}
- Short-term Debt: ${latest_data['shortTermDebt']:,}
- Tax Payables: ${latest_data['taxPayables']:,}
- Deferred Revenue: ${latest_data['deferredRevenue']:,}
- Other Current Liabilities: ${latest_data['otherCurrentLiabilities']:,}
Total Current Liabilities: ${latest_data['totalCurrentLiabilities']:,}

Non-Current Liabilities:
- Long-term Debt: ${latest_data['longTermDebt']:,}
- Other Non-Current Liabilities: ${latest_data['otherNonCurrentLiabilities']:,}
Total Non-Current Liabilities: ${latest_data['totalNonCurrentLiabilities']:,}

Total Liabilities: ${latest_data['totalLiabilities']:,}

**Shareholders' Equity**
- Common Stock: ${latest_data['commonStock']:,}
- Retained Earnings: ${latest_data['retainedEarnings']:,}
- Accumulated Other Comprehensive Income/Loss: ${latest_data['accumulatedOtherComprehensiveIncomeLoss']:,}
Total Shareholders' Equity: ${latest_data['totalStockholdersEquity']:,}

Total Liabilities and Equity: ${latest_data['totalLiabilitiesAndStockholdersEquity']:,}

**Key Financial Ratios**
- Current Ratio: {latest_data['totalCurrentAssets']/latest_data['totalCurrentLiabilities']:.2f}
- Debt to Equity Ratio: {latest_data['totalLiabilities']/latest_data['totalStockholdersEquity']:.2f}
- Working Capital: ${latest_data['totalCurrentAssets'] - latest_data['totalCurrentLiabilities']:,}

**Additional Information**
Filing Date: {latest_data['fillingDate']}
SEC Filing Link: {latest_data['finalLink']}
"""
        return formatted_output

    except requests.RequestException as e:
        return ValidatorAgent.format_error_message({
            "status": False,
            "message": f"Network error occurred: {str(e)}"
        })
    except Exception as e:
        return ValidatorAgent.format_error_message({
            "status": False,
            "message": f"Unexpected error: {str(e)}"
        })

@tool
def get_cash_flow_statement(input_str: str):
    """Get cash flow statement details."""
    try:
        # Extract ticker if provided in different formats
        ticker = input_str.upper().strip().split(',')[0].replace('"', '').replace("'", "")
        if ":" in ticker:
            ticker = ticker.split(':')[1].strip()

        # API Configuration
        api_key = "LDi3hXVABun1oUg77fZPdXU6UjiSsJ0c"  # Replace with your API key
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=annual&apikey={api_key}"

        # Make API request
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return ValidatorAgent.format_error_message({
                "status": False,
                "message": f"API request failed with status code {response.status_code}"
            })

        data = response.json()

        # Validate the response
        validation_result = ValidatorAgent.validate_cash_flow_statement_response(data)
        if not validation_result["status"]:
            return ValidatorAgent.format_error_message(validation_result)

        # Get validated data
        latest = validation_result["data"]

        return (
            f"Operating Activities:\n"
            f"Net Income: ${latest['netIncome']:,}\n"
            f"Operating Cash Flow: ${latest['netCashProvidedByOperatingActivities']:,}\n"
            f"Working Capital Change: ${latest['changeInWorkingCapital']:,}\n\n"
            f"Investing Activities:\n"
            f"Capital Expenditure: ${latest['investmentsInPropertyPlantAndEquipment']:,}\n"
            f"Net Investing Cash Flow: ${latest['netCashUsedForInvestingActivites']:,}\n\n"
            f"Financing Activities:\n"
            f"Stock Repurchases: ${latest['commonStockRepurchased']:,}\n"
            f"Dividends Paid: ${latest['dividendsPaid']:,}\n"
            f"Net Financing Cash Flow: ${latest['netCashUsedProvidedByFinancingActivities']:,}\n\n"
            f"Cash Position:\n"
            f"Beginning Balance: ${latest['cashAtBeginningOfPeriod']:,}\n"
            f"Ending Balance: ${latest['cashAtEndOfPeriod']:,}\n"
            f"Net Change: ${latest['netChangeInCash']:,}\n\n"
            f"Key Metrics:\n"
            f"Free Cash Flow: ${latest['freeCashFlow']:,}\n"
            f"Period: {latest['date']}"
        )

    except requests.RequestException as e:
        return ValidatorAgent.format_error_message({
            "status": False,
            "message": f"Network error: {str(e)}"
        })
    except Exception as e:
        return ValidatorAgent.format_error_message({
            "status": False,
            "message": f"Error processing data: {str(e)}"
        })

@tool
def get_company_notes(input_str: str):
    """Get company notes (debt instruments) details."""
    try:
        # Extract ticker
        ticker = input_str.upper().strip().split(',')[0].replace('"', '').replace("'", "")
        if ":" in ticker:
            ticker = ticker.split(':')[1].strip()

        # API Configuration
        api_key = "uW7kG6fZagauUpoJowIRofnHJeCZbE7V"
        notes_url = f"https://financialmodelingprep.com/api/v4/company-notes?symbol={ticker}&apikey={api_key}"

        # Make API request
        response = requests.get(notes_url, timeout=10)

        if response.status_code != 200:
            return ValidatorAgent.format_error_message({
                "status": False,
                "message": f"API request failed with status code {response.status_code}"
            })

        data = response.json()

        # Validate the response
        validation_result = ValidatorAgent.validate_company_notes_response(data)
        if not validation_result["status"]:
            return ValidatorAgent.format_error_message(validation_result)

        # Format validated notes
        validated_notes = validation_result["data"]

        # Sort notes by due date
        validated_notes.sort(key=lambda x: int(x['title'].split('due ')[1]))

        response_str = f"Company Notes for {ticker}:\n\n"

        for note in validated_notes[0]:
            # Extract interest rate and due date from title
            title_parts = note['title'].split('% Notes due ')
            interest_rate = title_parts[0]
            due_date = title_parts[1]

            response_str += (
                f"Note Details:\n"
                f"- Interest Rate: {interest_rate}%\n"
                f"- Due Date: {due_date}\n"
                f"- Exchange: {note['exchange']}\n"
                f"- CIK: {note['cik']}\n\n"
            )

        return response_str

    except requests.RequestException as e:
        return ValidatorAgent.format_error_message({
            "status": False,
            "message": f"Network error: {str(e)}"
        })
    except Exception as e:
        return ValidatorAgent.format_error_message({
            "status": False,
            "message": f"Error processing data: {str(e)}"
        })



"""# Historical Employment"""

class HistoricalEmployeeCount:
    _instance = None
    _data = None
    _faiss_index = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_data(self):
        if self._data is None:
            try:
                with open('employee_count_hist.pkl', 'rb') as f:
                    self._data = pickle.load(f)

                # Enhanced data verification
                required_keys = {'embeddings', 'symbols', 'details_dict'}
                if not all(key in self._data for key in required_keys):
                    missing_keys = required_keys - set(self._data.keys())
                    print(f"Missing required keys: {missing_keys}")
                    print(f"Available keys: {self._data.keys()}")
                    return None

                # Initialize FAISS index
                embeddings = np.array(self._data['embeddings']).astype('float32')
                dimension = embeddings.shape[1]
                self._faiss_index = faiss.IndexFlatL2(dimension)
                self._faiss_index.add(embeddings)

                print(f"Successfully loaded data with {len(self._data['symbols'])} companies")
                return self._data

            except FileNotFoundError:
                print("Employee count data file not found")
                return None
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                return None

    def search_ticker(self, ticker):
        """Enhanced search with validation and error handling"""
        if self._data is None:
            if self.load_data() is None:
                return {
                    'status': False,
                    'message': "Failed to load employee count data",
                    'data': None
                }

        try:
            clean_ticker = ticker.strip().upper()

            # Exact match check
            if clean_ticker in self._data['symbols']:
                details = self._data['details_dict'][clean_ticker]
                if self._validate_company_data(details):
                    return {
                        'status': True,
                        'data': details,
                        'match_type': 'exact'
                    }

            # Similarity search
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode([clean_ticker])[0].reshape(1, -1).astype('float32')

            k = 3  # Number of similar results to return
            distances, indices = self._faiss_index.search(query_embedding, k)

            similar_matches = [
                {
                    'ticker': self._data['symbols'][idx],
                    'similarity': float(1 / (1 + dist))  # Normalize similarity score
                }
                for idx, dist in zip(indices[0], distances[0])
            ]

            # Check if best match is close enough
            if distances[0][0] < 1.5:
                best_match_ticker = similar_matches[0]['ticker']
                details = self._data['details_dict'][best_match_ticker]
                if self._validate_company_data(details):
                    return {
                        'status': True,
                        'data': details,
                        'match_type': 'similar',
                        'similar_matches': similar_matches
                    }

            return {
                'status': False,
                'message': f"No close matches found for {clean_ticker}",
                'similar_matches': similar_matches
            }

        except Exception as e:
            print(f"Search error for {ticker}: {str(e)}")
            return {
                'status': False,
                'message': f"Error during search: {str(e)}",
                'data': None
            }

    def _validate_company_data(self, details):
        """Validate company data structure and content"""
        required_fields = {'employeeCount', 'periodOfReport', 'companyName', 'source'}

        if not isinstance(details, dict):
            return False

        if not all(field in details for field in required_fields):
            return False

        if not all(isinstance(details[field], dict) for field in required_fields):
            return False

        return True

@tool
def historical_employee_count(input_str: str):
    """Get latest employee count with year-over-year comparison and additional metrics."""
    try:
        print(f"DEBUG: Raw input received: {repr(input_str)}")

        # Extract ticker from JSON input
        ticker = None
        try:
            import json
            if isinstance(input_str, str):
                input_dict = json.loads(input_str)
                ticker = input_dict.get('ticker', '')
                print(f"DEBUG: Ticker extracted: {ticker}")
            elif isinstance(input_str, dict):
                ticker = input_str.get('ticker', '')
        except json.JSONDecodeError:
            match = re.search(r'"ticker":\s*"([^"]+)"', input_str)
            if match:
                ticker = match.group(1)
            else:
                ticker = input_str

        if not ticker:
            return "Could not extract ticker symbol from input"

        # Clean the ticker
        ticker = ticker.strip().upper()
        print(f"DEBUG: Final ticker: {ticker}")

        # Get data
        data_manager = HistoricalEmployeeCount.get_instance()
        search_result = data_manager.search_ticker(ticker)

        # Validate the response
        validation_result = ValidatorAgent.validate_employee_count_response(search_result)
        if not validation_result["status"]:
            return ValidatorAgent.format_employee_count_error(validation_result)

        data = validation_result["data"]

        try:
            latest_count = int(data['employeeCount']['0'])
            previous_count = int(data['employeeCount']['1'])
            latest_date = data['periodOfReport']['0']
            company_name = data['companyName']['0']
            source = data['source']['0']

            # Calculate metrics
            change = latest_count - previous_count
            change_pct = (change / previous_count) * 100

            # Format response
            response = f"""**Employee Information for {company_name} ({ticker})**

**Current Status:**
- Reporting Period: {latest_date}
- Current Employee Count: {latest_count:,}

**Year-over-Year Analysis:**
- Previous Count: {previous_count:,}
- Change: {change:+,d} employees
- Percentage Change: {change_pct:+.2f}%

**Source:** [SEC Filing]({source})

_Data sourced from official SEC filings_"""

            return response

        except (KeyError, ValueError) as e:
            print(f"DEBUG: Data processing error: {str(e)}")
            return f"Error processing data for {ticker}: {str(e)}"

    except Exception as e:
        print(f"DEBUG: General error: {str(e)}")
        return f"Unable to process employee count data. Error: {str(e)}"



"""# Tool Registeration"""

class DynamicCompanyInformationAgent:
    def __init__(self):
        # Dynamic information types with advanced keyword matching
        self.info_types = {
            'price': [
                'price', 'stock price', 'current price',
                'trading value', 'market value', 'share cost'
            ],
            'key_metrics': [
                'key metrics', 'metrics', 'financial metrics',
                'performance indicators', 'financial health'
            ],
            'income_statement': [
                'income statement', 'earnings', 'revenue',
                'profit', 'financial performance', 'monetary results'
            ],
            'balance_sheet': [
                'balance sheet', 'financial position',
                'assets', 'liabilities', 'net worth'
            ],
            # 'company_profile': [
            #     'company profile', 'about company',
            #     'corporate information', 'business overview'
            # ],
            'historical_employee_count': [
                'employee count', 'workforce size', 'employee history',
                'historical employees', 'staff count', 'employment history'
            ],
            'cash_flow': [
                'cash flow', 'cash flow statement', 'operating cash',
                'cash position', 'cash management', 'cash activities', 'cash flow details'
            ],
            'executive_compensation': [
                'executive pay', 'compensation', 'leadership salary',
                'executive earnings', 'management compensation'
            ],
            'company_notes': [
                'notes', 'debt instruments', 'company notes',
                'debt notes', 'bonds', 'debt securities'
            ]
        }

    def determine_requested_info(self, query):
        """
        Advanced information type detection with fuzzy matching
        """
        requested_info = []

        for info_type, keywords in self.info_types.items():
            # Use fuzzy matching for more flexible keyword detection
            matches = [
                keyword for keyword in keywords
                if process.extractOne(keyword, [query.lower()])[1] > 80
            ]

            if matches:
                requested_info.append(info_type)

        return requested_info

    def retrieve_company_information(self, ticker, query=None):
        """
        Main method to retrieve company information dynamically
        """
        try:
            # Clean and extract ticker
            def clean_ticker(raw_ticker):
                if isinstance(raw_ticker, str) and raw_ticker.startswith('{'):
                    import ast
                    try:
                        ticker_dict = ast.literal_eval(raw_ticker)
                        return ticker_dict.get('TICKER', '').strip('"')
                    except:
                        import re
                        match = re.search(r'TICKER:\s*"?(\w+)"?', raw_ticker)
                        return match.group(1) if match else raw_ticker
                return raw_ticker

            # Clean the ticker
            clean_ticker_symbol = clean_ticker(ticker)
            # If query is provided, determine requested information
            requested_info = self.determine_requested_info(query) if query else None

            # If no specific info requested, return all available information
            if not requested_info:
                requested_info = list(self.info_types.keys())

            # Retrieve requested information dynamically
            results = {}
            for info_type in requested_info:
                method = getattr(self, f'get_{info_type}', None)
                if method:
                    results[info_type] = method(ticker)

            # Format response
            if len(results) == 1:
                return list(results.values())[0]
            else:
                response = f"Information for {ticker}:\n"
                for info_type, info in results.items():
                    response += f"\n{info_type.replace('_', ' ').title()}:\n{info}\n"
                return response

        except Exception as e:
            return f"Error retrieving company information: {str(e)}"

    # Placeholder methods - replace with actual implementation
    def get_price(self, ticker):
        return get_stock_price(ticker)

    def get_key_metrics(self, ticker):
        return get_key_metrics_and_analyze(ticker)

    def get_income_statement(self, ticker):
        return get_income_statement(ticker)

    def get_balance_sheet(self, ticker):
        return get_balance_sheet(ticker)

    # def get_company_profile(self, ticker):
    #     return get_company_profile(ticker)
    def get_cash_flow(self, ticker):
        return get_cash_flow_statement(ticker)

    def get_executive_compensation(self, ticker):
        return get_executive_compensation(ticker)

    def get_company_notes(self, ticker):
        return get_company_notes(ticker)
    def get_historical_employee_count(self, ticker):
        return historical_employee_count(ticker)

#Tool definition
@tool
def company_comprehensive_agent(ticker: str, query: str = None):
    """
    Comprehensive company information retrieval tool

    Args:
    - ticker: Stock ticker symbol (required)
    - query: Optional natural language query to specify information needed
    """
    agent = DynamicCompanyInformationAgent()
    return agent.retrieve_company_information(ticker, query)

tools = [get_now,company_comprehensive_agent]

tool_executor = ToolExecutor(tools)

"""# Other code"""

def execute_tools(state):
    print("Called `execute_tools`")
    messages = [state["agent_outcome"]]
    last_message = messages[-1]

    tool_name = last_message.tool

    print(f"Calling tool: {tool_name}")

    action = ToolInvocation(
        tool=tool_name,
        tool_input=last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    return {"intermediate_steps": [(state["agent_outcome"], response)]}

def run_agent(state):
    """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    """
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


def should_continue(state):
    messages = [state["agent_outcome"]]
    last_message = messages[-1]
    if "Action" not in last_message.log:
        return "end"
    else:
        return "continue"

from langchain_core.prompts import PromptTemplate
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)


workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)


workflow.add_edge("action", "agent")
app = workflow.compile()

model = ChatOllama(base_url = "https://3pi0j4xr6a957r-11434.proxy.runpod.net/",
                  model="athene-v2:latest")

# prompt = hub.pull("hwchase17/react")
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You are a **Trader with 100 years of experience** in financial markets. You have seen various market cycles and know the intricacies of trading strategies, market analysis, and risk management.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")


agent_runnable = create_react_agent(model, tools, prompt)

chat_history={}
inputs = {"input": """what is price of apple""",}
username = 'Pawan'
if username not in chat_history.keys():
    chat_history[username] = {'1': []}


chat_history[username]['1']

inputs['chat_history'] = chat_history

def clean_response(response: str) -> str:
    """Clean the response by removing the information header."""
    try:
        # If response starts with "Information for"
        if 'Information for {' in response:
            # Find the end of the JSON-like header
            header_end = response.find('}:')
            if header_end != -1:
                # Skip the "}: " part as well
                response = response[header_end + 2:].strip()
        return response
    except Exception as e:
        print(f"Debug - Error cleaning response: {str(e)}")
        return response

# Then in your main code:
# results = []
# temp_chat = {'User': inputs['input'], 'AI': ''}  # Initialize with default values
# result =  None
# try:
#     for s in app.stream(inputs):
#         try:
#             result = list(s.values())[0]
#             results.append(result)

#             # Try to get the final answer from the standard output format
#             if 'agent_outcome' in result and hasattr(result['agent_outcome'], 'log'):
#                 if 'Final Answer:' in result['agent_outcome'].log:
#                     response = result['agent_outcome'].log.split('Final Answer:')[1].strip()
#                     temp_chat['AI'] = clean_response(response)

#         except (IndexError, KeyError) as e:
#             print(f"Debug - Stream processing error: {str(e)}")
#             continue
#         except Exception as e:
#             print(f"Debug - Unexpected error in stream processing: {str(e)}")
#             continue

# except OutputParserException as e:
#     print(f"Debug - Output parsing error: {str(e)}")
#     try:
#         if result and 'intermediate_steps' in result and result['intermediate_steps']:
#             response = list(result['intermediate_steps'][0])[1]
#             temp_chat['AI'] = clean_response(response)
#         else:
#             temp_chat['AI'] = "I apologize, but I encountered an error while processing the information. Please try asking your question again, perhaps with different wording."
#     except Exception as inner_e:
#         print(f"Debug - Error handling intermediate steps: {str(inner_e)}")
#         temp_chat['AI'] = "I apologize, but I encountered an error while processing the information. Please try asking your question again."

# except Exception as e:
#     print(f"Debug - Stream error: {str(e)}")
#     try:
#         if result and 'intermediate_steps' in result and result['intermediate_steps']:
#             response = list(result['intermediate_steps'][0])[1]
#             temp_chat['AI'] = clean_response(response)
#         else:
#             temp_chat['AI'] = "An error occurred while processing your request. Please try again."
#     except Exception as inner_e:
#         print(f"Debug - Error handling intermediate steps: {str(inner_e)}")
#         temp_chat['AI'] = "An error occurred while processing your request. Please try again."

# print(temp_chat['AI'])
def process_user_input(user_input: str, chat_history: dict, username: str) -> str:
    # Initialize chat history for the user if not already present
    if username not in chat_history:
        chat_history[username] = {'1': []}

    chat_history[username]['1'].append({'User': user_input, 'AI': ''})

    inputs = {"input": user_input, "chat_history": chat_history}

    results = []
    temp_chat = {'User': user_input, 'AI': ''}  # Initialize with default values
    result = None
    try:
        for s in app.stream(inputs):
            try:
                result = list(s.values())[0]
                results.append(result)

                # Try to get the final answer from the standard output format
                if 'agent_outcome' in result and hasattr(result['agent_outcome'], 'log'):
                    if 'Final Answer:' in result['agent_outcome'].log:
                        response = result['agent_outcome'].log.split('Final Answer:')[1].strip()
                        temp_chat['AI'] = clean_response(response)

            except (IndexError, KeyError) as e:
                print(f"Debug - Stream processing error: {str(e)}")
                continue
            except Exception as e:
                print(f"Debug - Unexpected error in stream processing: {str(e)}")
                continue

    except OutputParserException as e:
        print(f"Debug - Output parsing error: {str(e)}")
        try:
            if result and 'intermediate_steps' in result and result['intermediate_steps']:
                response = list(result['intermediate_steps'][0])[1]
                temp_chat['AI'] = clean_response(response)
            else:
                temp_chat['AI'] = "I apologize, but I encountered an error while processing the information. Please try asking your question again, perhaps with different wording."
        except Exception as inner_e:
            print(f"Debug - Error handling intermediate steps: {str(inner_e)}")
            temp_chat['AI'] = "I apologize, but I encountered an error while processing the information. Please try asking your question again."

    except Exception as e:
        print(f"Debug - Stream error: {str(e)}")
        try:
            if result and 'intermediate_steps' in result and result['intermediate_steps']:
                response = list(result['intermediate_steps'][0])[1]
                temp_chat['AI'] = clean_response(response)
            else:
                temp_chat['AI'] = "An error occurred while processing your request. Please try again."
        except Exception as inner_e:
            print(f"Debug - Error handling intermediate steps: {str(inner_e)}")
            temp_chat['AI'] = "An error occurred while processing your request. Please try again."

    chat_history[username]['1'][-1]['AI'] = temp_chat['AI']
    return temp_chat['AI']