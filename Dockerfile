# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all other files
COPY . .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the port Flask will run on
EXPOSE 8000

# Run gunicorn to serve the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
