# React Frontend Migration - Deployment Guide

## Overview

This guide covers deploying the React frontend alongside the FastAPI backend.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                     │
├─────────────────────────────────────────────────────────┤
│  Frontend (React)           │      Backend (FastAPI)     │
│  • vite build → dist/       │      • /api/* routes       │
│  • npm run build            │      • /ws/* WebSocket     │
│  • Static file serving      │      • Authentication      │
│  • Port: 3000 (dev)         │      • Port: 8000          │
│                              │                            │
│  ← Served by FastAPI ────────────→ Flask/Uvicorn        │
└─────────────────────────────────────────────────────────┘
```

## Development Workflow

### Local Development (Two Terminals)

**Terminal 1: Backend**
```bash
cd /path/to/My_Prediction
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2: Frontend**
```bash
cd /path/to/My_Prediction/frontend-react
npm run dev  # Starts on http://localhost:5173
```

The Vite dev server proxies API calls to `http://localhost:8000`, so no CORS issues.

## Production Deployment

### Step 1: Build React Frontend

```bash
cd frontend-react
npm run build
```

This creates `frontend-react/dist/` with optimized production files.

### Step 2: Update Dockerfile

Modify the existing Dockerfile to include React build:

```dockerfile
# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend-react
COPY frontend-react/package*.json ./
RUN npm install
COPY frontend-react .
RUN npm run build

# Stage 2: Build Python backend + serve frontend
FROM python:3.11-slim
WORKDIR /app

# Copy backend files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy built React app into static folder
COPY --from=frontend-builder /app/frontend-react/dist ./static

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 3: Update main.py for Static File Serving

Add to `main.py`:

```python
from fastapi.staticfiles import StaticFiles
import os

# ... existing code ...

# Mount React static files (serve on root path)
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

This serves React's `dist/` folder at the root, handling:
- HTML, JS, CSS static assets
- SPA routing (any unmatched route returns `index.html`)

### Step 4: Build & Run Docker Image

```bash
docker build -t quantai:latest .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e DATABASE_URL=postgresql://... \
  quantai:latest
```

Now visit `http://localhost:8000` to access the React frontend.

## Deployment Platforms

### Heroku

```bash
# 1. Install Heroku CLI
# 2. Login
heroku login

# 3. Create app
heroku create quantai-app

# 4. Set environment variables
heroku config:set OPENAI_API_KEY=sk-...
heroku config:set DATABASE_URL=postgresql://...

# 5. Deploy
git push heroku main

# 6. View logs
heroku logs --tail
```

### AWS

**Option A: ECS (Docker)**
1. Build and push image to ECR
2. Create ECS service/task
3. Configure ALB for routing

**Option B: Elastic Beanstalk**
```bash
eb create quantai-app
eb deploy
```

### Railway / Render

Simple alternatives to Heroku:

1. Connect GitHub repo
2. Set environment variables
3. Auto-deploy on push

### Local Production Testing

```bash
# Build frontend
cd frontend-react && npm run build && cd ..

# Start backend with static file serving
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000`.

## Environment Variables

For production, ensure these are set:

```env
# Authentication
OPENAI_API_KEY=sk-...
JWT_SECRET=your-secure-secret

# Database
DATABASE_URL=postgresql://user:password@host:5432/quantai

# Optional
REDIS_URL=redis://localhost:6379
APP_ENV=production
DEBUG=false

# CORS
FRONTEND_URL=https://yourdomain.com
```

FastAPI CORS middleware should allow the frontend domain:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # dev
        "https://yourdomain.com",  # prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Database Migrations

Before running, ensure PostgreSQL is set up:

```bash
# Inside container or locally:
python -c "from core.database import init_db; asyncio.run(init_db())"
```

## Monitoring & Logs

### Local
```bash
# Backend logs
uvicorn main:app --log-level debug

# Frontend (if dev server running)
npm run dev  # Shows compilation + HMR
```

### Production (Docker)
```bash
docker logs -f <container_id>
```

### Via Logging Service (DataDog, Sentry, etc.)
```python
import logging
logging.getLogger().addHandler(SentryHandler())
```

## Performance Optimization

### Frontend
- Bundle is ~150KB gzipped
- Uses code splitting for routes
- Static files cached by CDN
- Images optimized (WEBP)

### Backend
- Redis caching for API responses
- Database connection pooling
- Rate limiting enabled
- GZIP compression

### Deployment
- Use CDN (Cloudflare, AWS CloudFront)
- Enable HTTP/2 push for critical assets
- Set appropriate cache headers

## Rollback Procedure

If deployment fails:

**Local Development:**
```bash
cd frontend-react && npm run build && cd ..
# Test locally then re-deploy
```

**Docker:**
```bash
# Revert to previous image
docker run -p 8000:8000 quantai:v1.2.0
```

**Git:**
```bash
git revert <bad-commit>
git push
```

## Continuous Integration

Add GitHub Actions workflow (`.github/workflows/deploy.yml`):

```yaml
name: Deploy
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build frontend
        run: cd frontend-react && npm install && npm run build
      - name: Build Docker image
        run: docker build -t quantai:latest .
      - name: Deploy
        run: docker push my-registry/quantai:latest
```

## Troubleshooting

### Frontend Route Not Found
- Ensure `app.mount("/", StaticFiles(...), html=True)` is in `main.py`
- This rewrites `/something` → `index.html` for React routing

### WebSocket Connection Fails
- Backend must expose `/ws/*` endpoints (already does)
- Frontend must use correct protocol (`ws://` or `wss://`)

### 404 on API Calls
- Check API routes exist in `main.py`
- Verify backend is running
- Check CORS configuration

### Slow Page Loads
- Enable production build: `npm run build`
- Check backend response times
- Use CDN for static assets

---

**Last Updated:** March 10, 2026
