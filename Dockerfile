FROM ubuntu:24.04

# ROCm 10 は公式イメージが -full（約 8GB）しかないため、ubuntu に OpenCL ランタイムだけを入れる
ARG ROCM_VERSION=10.0
ENV ROCM_PATH=/opt/rocm/core-${ROCM_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates wget gpg python3 \
    && mkdir -p /etc/apt/keyrings \
    && wget -qO- https://stable.repo.amd.com/rocm/gpg/packages.gpg \
        | gpg --dearmor > /etc/apt/keyrings/amdrocm.gpg \
    && printf '%s\n' \
        'Types: deb' \
        'URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/' \
        'Suites: stable' \
        'Components: main' \
        'Architectures: amd64' \
        'Signed-By: /etc/apt/keyrings/amdrocm.gpg' \
        > /etc/apt/sources.list.d/amdrocm.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends amdrocm-opencl${ROCM_VERSION} \
    && apt-get purge -y wget gpg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ROCm 10 は OpenCL を ${ROCM_PATH} 配下に置き、ICD も標準の場所に登録しない。
# ICD を /etc/OpenCL/vendors に登録し、OpenCV には ROCm 付属の ICD ローダーを直接読ませる
RUN mkdir -p /etc/OpenCL/vendors \
    && echo "${ROCM_PATH}/lib/opencl/libamdocl64.so.2" > /etc/OpenCL/vendors/amdocl64.icd
ENV OPENCV_OPENCL_RUNTIME=${ROCM_PATH}/lib/libOpenCL.so.1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
