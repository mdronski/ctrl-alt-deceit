# ctrl-alt-deceit

## Install uv 
Follow official docs to install uv - https://docs.astral.sh/uv/getting-started/installation/

## Install dependencies

```bash
uv sync
```

## Add dependencies

```bash
uv add <library_name>
```

## Configure env variables
Create local *.env* file and define API keys env variables there

```bash
TAVILY_API_KEY="YOUR_API_KEY"
GOOGLE_API_KEY="YOUR_API_KEY"
FINNHUB_API_KEY="YOUR_API_KEY"
GNEWS_API_KEY="YOUR_API_KEY"
NEWSAPI_API_KEY="YOUR_API_KEY"
```

## Run application

```bash
source .env
uv run app
```