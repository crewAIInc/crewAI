FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CrewAI
RUN pip install --no-cache-dir crewai

# Copy application code
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "server.py"]
