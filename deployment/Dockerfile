FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

WORKDIR /app

COPY "pyproject.toml" "uv.lock" "./"

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-cache

COPY "predict.py" "pipeline_v1.bin" ./

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

