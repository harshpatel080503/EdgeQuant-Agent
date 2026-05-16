# EdgeQuant Agent: High-Conviction Hedge Fund PM

![EdgeQuant Dashboard](edgequant_dashboard_mockup_1778827556427.png)

## Description
**EdgeQuant Agent** is an autonomous AI trading system architected as a **High-Conviction Hedge Fund Portfolio Manager**. Built for the **CLEF 2026 Financial AI Challenge**, it transcends standard "HOLD-biased" agents by identifying dominant structural drivers through a combination of **FinMA-7B** (Finance-specialized LLM) and a sophisticated **Retrieval-Augmented Generation (RAG)** memory system.

The agent focuses on capturing **Alpha** by quantifying **Catalyst Magnitude** and **Expectation Variance** across volatile assets like Bitcoin (BTC) and Tesla (TSLA). It is designed to navigate market noise and execute high-conviction trades when structural catalysts are identified.

---

## Features
- **High-Conviction Alpha Engine**: Optimized to ignore retail noise and focus on institutional-grade market catalysts.
- **Three-Tiered Cognitive Memory**:
    - **Short-term**: Real-time news ingestion and immediate price action analysis.
    - **Mid-term**: Multi-day contextual patterns and recent self-reflections.
    - **Long-term**: Fundamental regime detection and historical market correlation.
- **RAG-Enhanced Reasoning**: Integration with **Qdrant/ChromaDB** for semantic retrieval of historical market "lessons" before every trade.
- **Risk-Aware Portfolio Optimization**: Uses **CVXPY** for convex optimization-based position sizing and asset allocation.
- **Multi-Phase Lifecycle**: Integrated pipeline for **Warmup** (Memory Building), **Simulation** (Backtesting), and **Performance Evaluation**.
- **Production-Ready API**: FastAPI-based infrastructure designed for sub-second responses in live trading competitions.

---

## Tech Stack
| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **LLM Engine** | Hugging Face Transformers (`FinMA-7B`, `Llama-3.1-8B`) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (Sentence-Transformers) |
| **Vector Database** | Qdrant / ChromaDB |
| **Optimization** | CVXPY (Convex Optimization) |
| **API Framework** | FastAPI & Uvicorn |
| **Data Handling** | Pandas, NumPy, Scipy, Pydantic |
| **Observability** | Loguru, Rich, Guardrails AI |
| **Deployment** | Docker & Docker Compose |

---

## Architecture
The agent follows a modular architecture where the **Memory Engine** acts as the bridge between raw market data and the **LLM Reasoner**.

```mermaid
graph TD
    subgraph Market Environment
        A[News / Filings / Prices]
    end

    subgraph EdgeQuant Agent
        B[Ingestion Module] --> C{Memory Engine}
        C -->|Semantic Search| D[(Vector Store)]
        D -->|Contextual Recall| E[LLM Reasoner: FinMA]
        E -->|Catalyst Analysis| F[Decision Controller]
        F --> G[Portfolio Manager]
    end

    G -->|CVXPY Optimization| H[Trade Execution]
    H -->|BUY/SELL/HOLD| Market Environment
    H -->|Post-Trade Reflection| D
```

---

## Installation

### 1. Prerequisites
- **Python 3.10+**
- **CUDA-compatible GPU** (Recommended 24GB+ VRAM for local inference)
- **Hugging Face Access Token** (for model weights)

### 2. Standard Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/edgequant-agent.git
cd edgequant-agent

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install GPU-specialized PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install accelerate bitsandbytes
```

---

## Environment Variables
Create a `.env` file in the root directory to configure the agent:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `HF_TOKEN` | Hugging Face API Token for model access | `Required` |
| `PORT` | API Server Port | `7860` |
| `CHAT_ENDPOINT` | URL for the LLM Inference Engine (e.g., Ollama) | `http://localhost:11434` |
| `CHAT_MODEL` | Specific model identifier | `gpt-oss:120b` |
| `CONFIG_PATH` | Path to the agent configuration JSON | `configs/main.json` |

---

## 📈 Usage

### Phase 1: Warmup (Memory Ingestion)
Builds the initial vector store by processing historical news and generating reflections.
```bash
python run.py warmup --config-path configs/main.json
```

### Phase 2: Simulation (Trading)
Executes the trading strategy over the test period using the built memory.
```bash
python run.py test
```

### Phase 3: Evaluation
Generates a comprehensive performance report (Sharpe, MDD, Returns).
```bash
python run.py eval
```

### Hosting the API
To run the agent as a live competition service:
```bash
python app.py
```

---

## API Endpoints
The agent exposes a RESTful API for real-time interaction.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root health check and welcome. |
| `GET` | `/health` | Detailed status (LLM connectivity, Agent readiness). |
| `POST` | `/trading_action/` | **Core Endpoint**: Submit market data to receive a decision. |

### Sample `trading_action` Payloads:

#### Bitcoin (BTC)
```json
{
  "date": "2026-05-15",
  "symbol": ["BTC"],
  "price": {"BTC": 65000.0},
  "news": {"BTC": ["Breaking: Institutional ETF inflow spikes..."]},
  "momentum": {"BTC": "bullish"}
}
```

#### Tesla (TSLA)
```json
{
  "date": "2026-05-15",
  "symbol": ["TSLA"],
  "price": {"TSLA": 175.0},
  "news": {"TSLA": ["Tesla quarterly delivery numbers beat estimates..."]},
  "momentum": {"TSLA": "bullish"}
}
```

---

## Folder Structure
```text
.
├── configs/            # JSON configurations for agent logic & portfolio params
├── src/                # Core Source Code
│   ├── agent.py        # Central reasoning logic and LLM orchestration
│   ├── portfolio.py    # Portfolio management and risk optimization
│   ├── memory_db.py    # RAG, Vector Store, and Reflection logic
│   ├── market_env.py   # Simulation environment for backtesting
│   └── competition_api.py # FastAPI implementation for CLEF Task 3
├── data/               # Local storage for BTC & TSLA datasets
├── checkpoints/        # Serialized states for warmup/test phases
├── metrics/            # Auto-generated performance reports & logs
├── outputs/            # Phase-specific execution logs and trade results
├── Dockerfile          # Production container definition
├── app.py              # API server entry point
└── run.py              # CLI utility for multi-phase execution
```

---

## Docker Setup
Run the EdgeQuant Agent in a containerized environment:

```bash
# Build the image
docker build -t edgequant-agent .

# Run the container (injecting GPU access)
docker run --gpus all -p 7860:7860 --env-file .env edgequant-agent
```

---

## Dataset Details
- **Source**: `TheFinAI/CLEF_Task3_Trading` (HuggingFace)
- **Symbols**: `BTC` (Digital Asset), `TSLA` (Equity)
- **Granularity**: Daily Close / Sentiment / 10-K & 10-Q Filings
- **Features**: Includes news headlines, price history, and momentum indicators.

---

## Results & Metrics
Based on benchmark evaluations using the `gpt-oss:120b-cloud` backplane:

```text
MODEL INFO
----------------------
Model Used: gpt-oss:120b-cloud

INDIVIDUAL ASSET METRICS

BTC
----------------------
Cumulative Return: -0.0444
Sharpe Ratio: -0.3740
Max Drawdown: 0.1910
Volatility: 0.3871

TSLA
----------------------
Cumulative Return: 0.4061
Sharpe Ratio: 5.1522
Max Drawdown: 0.0949
Volatility: 0.3323


PORTFOLIO METRICS

               Metric  Equal Weight Portfolio  Agent Portfolio
    Cumulative Return                0.188627         0.180871
         Sharpe Ratio                2.387286         3.164874
         Max Drawdown                0.117480         0.082248
Annualized Volatility                0.381591         0.266215
```

---

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
