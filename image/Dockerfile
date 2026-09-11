# GLM-5.3-Flash GB10/SM121 image scaffold for 2× DGX Spark (TP=2).
# Prefer upstream GB10/SM121 support; strip 4-node-only deps after analysis.
# Do not bake model weights or HF_TOKEN into the image.

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# TODO: align CUDA / vLLM / PyTorch / FlashInfer / Triton with upstream recipe
LABEL org.opencontainers.image.description="GLM-5.3-Flash 2x DGX Spark scaffold"

WORKDIR /opt/glm53

# Copy patches and verification (implementation fills real patch logic)
COPY patches/ /opt/glm53/patches/
COPY verify.py /opt/glm53/verify.py
COPY entrypoint.sh /opt/glm53/entrypoint.sh

RUN chmod +x /opt/glm53/entrypoint.sh /opt/glm53/patches/link_cuda_headers.sh \
    && python3 /opt/glm53/verify.py || true

# Standard HF cache path inside container (host mount expected)
ENV HF_HOME=/root/.cache/huggingface
# Do not set MODEL_REVISION / --revision anywhere

ENTRYPOINT ["/opt/glm53/entrypoint.sh"]
