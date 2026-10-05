# HIP / PyTorch などへ広げる前提で、ROCm の SDK 一式が入った -full イメージを使う
ARG ROCM_IMAGE_TAG=10.0.0-full
FROM rocm/dev-ubuntu-24.04:${ROCM_IMAGE_TAG}

# ROCm 10 は OpenCL の ICD を /etc/OpenCL/vendors に登録しないため、自前で登録し、
# OpenCV には ROCm 付属の ICD ローダーを直接読ませる
RUN mkdir -p /etc/OpenCL/vendors \
    && echo "${ROCM_PATH}/lib/opencl/libamdocl64.so.2" > /etc/OpenCL/vendors/amdocl64.icd
ENV OPENCV_OPENCL_RUNTIME=/opt/rocm/lib/libOpenCL.so.1

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
