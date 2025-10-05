# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and
# to ensure stdout and stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create application directory
WORKDIR /app

# Install system dependencies needed for Python packages (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set a default location for the API key inside the container.  This path
# can be overridden at runtime using the ``OPENAI_KEY_PATH`` environment
# variable.  Persisting data in /data allows Docker volumes to retain
# configuration across container restarts or removals.
ENV OPENAI_KEY_PATH=/data/.openai_api_key

# Declare a volume for persistent data.  Mount a host directory or
# named volume to /data when running the container to persist the API
# key across runs (see README for instructions).
VOLUME ["/data"]

# Expose the port Flask will run on.  The application uses a high
# numbered port (5500) by default, which can be overridden at runtime via
# the PORT environment variable.
EXPOSE 5500

# Set the entrypoint. When running under Docker, gunicorn or another
# production server could be used instead of ``python app.py``. For
# simplicity we use the development server here.
CMD ["python", "app.py"]