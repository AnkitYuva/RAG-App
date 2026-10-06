# Docker Deployment

The Docker image runs the exact same Gradio RAG application as the local project.

## Run with Docker Compose

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it.
2. In the project folder, create or update `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
```

3. Start the application:

```powershell
docker compose up --build -d
```

4. Open http://localhost:7860.

View logs with:

```powershell
docker compose logs -f
```

Stop the application with:

```powershell
docker compose down
```

## Run with Docker Only

```powershell
docker build -t sem1-rag-assistant .
docker run --rm -p 7860:7860 --env-file .env --name sem1-rag-assistant sem1-rag-assistant
```

Then open http://localhost:7860.

## Deploy a Container

Push the repository to GitHub, then create a Docker web service on Render, Railway, or any container host. Set this environment variable in that host's secret manager:

```text
OPENROUTER_API_KEY
```

The platform should expose port `7860`, or provide a `PORT` environment variable. The application automatically uses `PORT` when it is set.

Never commit `.env`; it is excluded from Git and Docker image builds.
