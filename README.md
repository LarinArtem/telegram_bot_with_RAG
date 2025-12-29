Telegram-Based Financial Analytics Platform with Retrieval-Augmented Generation (RAG)
This project is a modular, production-oriented Telegram financial analytics bot that integrates real-time market data, quantitative screening models, options analytics, and large-language-model–driven document analysis into a single conversational interface. The system is designed to provide actionable investment insights by combining traditional financial data pipelines with retrieval-augmented generation (RAG) techniques for regulatory filings.
System Architecture and Design
The application is built in Python and follows a clean, modular architecture. Core analytical functionality is separated into a dedicated modules/ package, while the Telegram interaction layer orchestrates user input, asynchronous execution, and formatted output. The bot leverages the python-telegram-bot framework with callback handlers and inline keyboards to deliver a responsive, menu-driven user experience.
The architecture emphasizes:
•	Separation of concerns between data retrieval, analytics, and user interaction
•	Extensibility for additional financial tools
•	Asynchronous execution for I/O-heavy tasks such as API calls and LLM inference
Quantitative Market Analytics Modules
Several independent analytics engines are implemented and exposed through the Telegram interface:
•	Momentum Screening
A momentum engine queries the EOD HD market screener API to identify mid- to large-cap equities exhibiting strong five-day price momentum. Results are filtered by market capitalization and exchange, normalized into structured Pandas DataFrames, and returned as ranked summaries.
•	Undervalued Stock Identification
This module screens equities flagged as undervalued by Wall Street consensus signals, again using EOD HD data. The logic prioritizes analyst-derived valuation indicators and presents results in a concise, human-readable format.
•	Reddit Sentiment Analysis
A sentiment module integrates with the ApeWisdom API to retrieve Reddit ticker rankings. It computes day-over-day changes in mentions and rank position, enabling users to gauge retail investor momentum and sentiment shifts in near real time.
Options Strategy and Volatility Modeling
The options analytics engine is a quantitatively rigorous component designed for derivatives screening:
•	Implements the Yang–Zhang volatility estimator to compute realized volatility from historical OHLC price data
•	Retrieves live options chains via yfinance and extracts at-the-money implied volatility across maturities
•	Constructs an implied-volatility term structure using linear spline interpolation (scipy.interpolate)
•	Evaluates trading candidates based on liquidity thresholds, implied-to-realized volatility ratios, and term-structure slope dynamics
Each ticker is classified into Recommended, Consider, or Avoid categories, with transparent diagnostic metrics returned to the user.
Retrieval-Augmented Generation for SEC Filings
The most advanced component of the project is its RAG-based regulatory filing analysis engine:
•	Automatically fetches the latest 10-K or 10-Q filings using the EDGAR API
•	Cleans and tokenizes large documents, then chunks them to respect LLM context limits
•	Generates vector embeddings using OpenAI’s embedding models, with token accounting handled via tiktoken
•	Applies LLM-based analysis to core filing sections:
o	Risk Factors
o	Financial Statements
o	Management’s Discussion & Analysis (MD&A)
The system assigns structured grades to each section, produces executive-level summaries, synthesizes an overall assessment, and generates a forward-looking investment perspective. Both 10-K and 10-Q workflows are handled through a unified interface, demonstrating abstraction and code reuse.
Key Technologies and Tools
•	Languages & Frameworks: Python, python-telegram-bot
•	Data & Analytics: Pandas, NumPy, SciPy, yfinance
•	APIs: EOD HD, ApeWisdom, EDGAR
•	LLM & RAG Stack: OpenAI API, embeddings, tiktoken
•	Design Patterns: Modular services, asynchronous handlers, API abstraction
Professional Significance
This project demonstrates the ability to:
•	Design and deploy a multi-service financial analytics platform
•	Integrate LLMs with structured financial data in a production context
•	Apply quantitative finance concepts (momentum, volatility modeling, options term structure)
•	Build scalable, user-facing tools that translate complex analytics into accessible insights

