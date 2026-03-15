FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 先复制依赖文件，利用 Docker 缓存
COPY pyproject.toml uv.lock* ./

# 安装依赖
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# 复制项目代码
COPY src/ src/

# 安装项目本身
RUN uv sync --no-dev

# 默认命令（被 docker-compose 覆盖）
CMD ["uv", "run", "cryptocurrency_collaboration"]
