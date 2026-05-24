# How to Run the Application

## Prerequisites

1. **Python environment** with all dependencies installed
2. **Node.js/Bun** for the frontend
3. **Neo4j database** running (check your `.env` file for connection details)

## Step 1: Start the Backend API

Open a terminal in the project root directory and run:

```bash
# Activate your Python virtual environment (if using one)
GraphRAG\Scripts\activate

# Start the FastAPI server
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at: `http://localhost:8000`

You can verify it's running by visiting: `http://localhost:8000/docs` (FastAPI Swagger UI)

## Step 2: Start the Frontend

Open a **new terminal** window, navigate to the frontend folder, and run:

```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start the development server
npm run dev
```

The frontend will be available at: `http://localhost:5173` (or another port if 5173 is busy)

## Step 3: Use the Application

1. Open your browser and go to `http://localhost:5173`
2. You should see the **Verdant KYC Intelligence** interface
3. Upload a filing (HTML or PDF format)
4. Enter the **Fiscal Year** (e.g., "2024")
5. Click **"Run Diligence"** to start the pipeline

## What Happens Next

The pipeline will run through 6 steps:
1. **Parse & convert document** - Extracts structured sections
2. **Extract target company entities & build KG** - Identifies metrics, risks, and industries
3. **Identify SIC code & find peers** - Finds comparable companies
4. **Retrieve peer metrics via XBRL** - Gets financial data for peers
5. **Retrieve peer risk factors** - Extracts risk information from peer filings
6. **Build & populate peer knowledge graph** - Completes the Neo4j graph

You'll see real-time progress updates in the UI!

## Troubleshooting

### Backend Issues

- **Port already in use**: Change the port with `--port 8001`
- **Module not found**: Make sure you're in the project root and virtual environment is activated
- **Neo4j connection error**: Check your `.env` file has correct Neo4j credentials

### Frontend Issues

- **Port already in use**: The dev server will automatically try another port
- **Backend connection failed**: Check the backend URL in the settings (gear icon in top right)
- **Dependencies error**: Run `bun install` again

### API Connection

If the frontend can't connect to the backend:
1. Click the **Settings icon** (gear) in the top right
2. Verify the Backend URL is set to `http://localhost:8000`
3. Check that the backend terminal shows the server is running

## Environment Variables

Make sure your `.env` file contains:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# OpenAI or other LLM API keys
OPENAI_API_KEY=your_key_here
```

## Notes

- The fiscal year you enter will be used to fetch peer data from the correct reporting period
- Both file and fiscal year are **required** fields
- Supported file formats: `.html`, `.htm`, `.pdf`
- The pipeline can take several minutes depending on the number of peers found
