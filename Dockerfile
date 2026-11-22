# One image for every node (coordinator, teacher workers, student trainer).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src/ src/
# Default job: data preparation. Override the command to run other scripts, e.g.
#   docker run protein-loc python src/baseline_student.py
CMD ["python", "src/prepare_data.py"]
