FROM python:3.10-slim

# Set working directory
WORKDIR /stealing_lab

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
