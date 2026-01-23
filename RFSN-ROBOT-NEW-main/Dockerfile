FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir mujoco numpy scipy pandas

CMD ["python", "run_demo.py"]
