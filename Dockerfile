FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the app with Gunicorn
CMD ["gunicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]