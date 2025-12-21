# Use Python 3.9
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy requirements first
COPY ./requirements.txt /code/requirements.txt

# 1. Install dependencies (CPU-only torch to save 2GB+ space)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 2. Pre-download models during the build phase
# [CRITICAL] I changed 'base' to 'small' here to match your python code.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    from transformers import pipeline; \
    print('Downloading SBERT...'); \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Downloading Flan-T5-Small...'); \
    pipeline('text2text-generation', model='google/flan-t5-small');"

# Copy the rest of the application code
COPY . .

# Set cache env var (models are already at /root/.cache/huggingface from the step above)
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Start the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]