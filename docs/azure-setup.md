# Azure Setup Guide

## Required Azure Resources

1. **Azure OpenAI** — for LLM and embeddings
   - Deploy: `gpt-4o` (chat)
   - Deploy: `text-embedding-ada-002` (embeddings)

2. **Azure SQL Database** (Project 3) — for patient data
   - Free tier: 32GB, sufficient for Synthea data

3. **Azure Cosmos DB** (Project 4) — for long-term memory
   - Free tier: 1000 RU/s, sufficient for dev

## Environment Variables

Copy `.env.example` to `.env` in each project folder and fill in:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

## Loading Synthea Data (Project 3)

```bash
curl -L https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar -o synthea.jar
java -jar synthea.jar -p 1000
python projects/project3-data-agent/db/seed_data.py
```
