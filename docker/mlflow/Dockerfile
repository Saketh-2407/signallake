FROM python:3.11-slim

# Match the host user's UID/GID so ./mlartifacts (bind-mounted) stays owned by you, not root.
ARG UID=1000
ARG GID=1000

RUN pip install --no-cache-dir mlflow \
    && groupadd --gid ${GID} mlflow \
    && useradd --uid ${UID} --gid ${GID} --create-home mlflow \
    && mkdir -p /mlflow /mlartifacts \
    && chown mlflow:mlflow /mlflow /mlartifacts

USER mlflow
WORKDIR /mlflow
EXPOSE 5000

# sqlite backend store lives in the `mlflow-db` named volume (/mlflow);
# artifacts are proxied by the server and written to ./mlartifacts on the host (/mlartifacts).
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlartifacts", \
     "--serve-artifacts"]
