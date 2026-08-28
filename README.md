# MLflow Evidence Promotion Gate

A deterministic `POST /promote` JSON API for promoting an MLflow model from verifiable evaluation evidence.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoint:

```text
POST http://localhost:8000/promote
```

## Deploy on Render

1. Create a GitHub repository and upload these files.
2. On Render, create a new **Web Service** from the repository.
3. Build command:
   `pip install -r requirements.txt`
4. Start command:
   `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy.
6. Submit the base URL, for example:
   `https://your-service.onrender.com`

The grader will call:

```text
POST https://your-service.onrender.com/promote
```

Do not put `/promote` in the public base URL field.
