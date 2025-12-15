# Use Python 3.9
FROM python:3.9

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file first (better for caching)
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
COPY . .

# Create a writable cache directory for Hugging Face (Important!)
RUN mkdir -p /tmp/cache
ENV TRANSFORMERS_CACHE=/tmp/cache

# Start the app using Gunicorn on port 7860 (Hugging Face default)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]