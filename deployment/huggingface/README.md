---
title: Hybrid Search Studio API
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

FastAPI backend for Hybrid Search Studio.

This Space runs the API only. Qdrant remains external, and the React frontend is
intended for a separate Vercel deployment.

Health check:

`GET /api/v1/health/live`
