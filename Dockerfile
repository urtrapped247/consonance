# Dockerfile for FastAPI backend
FROM python:3.10
# FROM python:3.10-slim-buster

# Set the working directory
# WORKDIR /app

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-glx \
    ffmpeg \
    fluidsynth \
    && apt-get clean

# Copy the requirements.txt file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip install --no-cache-dir tensorflow && \
    pip install --no-cache-dir git+https://github.com/tensorflow/addons.git


# Copy the rest of the application code
COPY consonance consonance
COPY setup.py setup.py
RUN pip install .

# COPY Makefile Makefile
# RUN make reset_local_files

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "consonance.api.fast:app", "--host", "0.0.0.0", "--port", $PORT]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
