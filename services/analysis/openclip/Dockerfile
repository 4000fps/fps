FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

ENV TRANSFORMERS_CACHE=/cache/huggingface

COPY services/analysis/openclip/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/analysis/openclip/serve.py /app/services/analysis/openclip/serve.py
COPY services/common/utils.py /app/services/common/utils.py

EXPOSE 5000
CMD ["python", "-u", "-m", "services.analysis.openclip.serve"]
