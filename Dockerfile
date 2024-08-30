# Dockerfile for FastAPI backend
FROM python:3.10
# FROM python:3.10-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-glx \
    ffmpeg \
    fluidsynth \
    libasound2 \
    libasound2-dev \
    libsndfile1-dev \
    && apt-get clean

# # Set the working directory
# WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install additional Python dependencies
RUN pip install --no-cache-dir tensorflow && \
    pip install --no-cache-dir git+https://github.com/tensorflow/addons.git

# # Copy the SoundFont file into the container
# COPY FluidR3_GM.sf2 /usr/share/sounds/sf2/default.sf2
# # Set FluidSynth to use the SoundFont
# ENV SOUND_FONT /usr/share/sounds/sf2/default.sf2

# Copy the application code
COPY consonance consonance
COPY setup.py setup.py
RUN pip install .

# COPY Makefile Makefile
# RUN make reset_local_files

# Expose the port the app runs on
# EXPOSE 8080 --> avoid exposing for now for gcp

# Command to run the FastAPI app
CMD uvicorn consonance.api.fast:app --host 0.0.0.0 --port $PORT
# CMD ["uvicorn", "consonance.api.fast:app", "--host", "0.0.0.0", "--port", $PORT] --> old command
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
