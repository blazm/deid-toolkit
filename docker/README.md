# Docker Environment

This directory contains Docker-specific configuration for `deid-toolkit`.

## Files

- **`conda-env.yml`** — Frozen conda export for reproducible Docker builds. Pins every transitive dependency to an exact version. Used by `Dockerfile.dev`.
  - For local development, use `environment.yml` (declarative, not frozen).
  - To regenerate: run `conda env export > docker/conda-env.yml` inside a clean container.

## Usage

```bash
docker compose -f docker-compose-dev.yml up -d --build
docker exec -it deidtoolkit /bin/bash
```
