# stage 1, react app
FROM node:20-slim AS frontend
WORKDIR /ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ ./
RUN npm run build

# stage 2, vector store
FROM python:3.13-slim
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY . /app

# copy frontend build
COPY --from=frontend /ui/dist /app/ui/dist

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]