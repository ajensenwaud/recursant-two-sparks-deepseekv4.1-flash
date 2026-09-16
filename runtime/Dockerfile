# Source-build candidate; see README release gates before spending build time.
FROM vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
ENV MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 NVCC_THREADS=1 \
    TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a
# Container-only packages. Exact versions fail rather than silently upgrading.
# Repository package SHA256/snapshot locking is a documented release blocker.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git=1:2.43.0-1ubuntu7.3 \
    libcusparse-dev-13-0=12.6.3.3-1 libcusolver-dev-13-0=12.0.4.66-1 \
    python3.12-dev=3.12.3-1ubuntu0.16 libpython3.12-dev=3.12.3-1ubuntu0.16 \
    && rm -rf /var/lib/apt/lists/*
COPY runtime/rootfs/opt/dsv41-patch/patch_exllamav3_aarch64.py /tmp/patch_exllama.py
RUN git clone https://github.com/turboderp-org/exllamav3.git /opt/src/exllamav3 \
    && git -C /opt/src/exllamav3 checkout --detach 5be886578ec80324c2c715269387be2058724b6e \
    && test "$(git -C /opt/src/exllamav3 rev-parse HEAD)" = 5be886578ec80324c2c715269387be2058724b6e \
    && python3 -S /tmp/patch_exllama.py /opt/src/exllamav3/exllamav3/exllamav3_ext \
    && python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir /opt/recipe-wheels /opt/src/exllamav3 \
    && python3 -m pip install --no-index --no-deps /opt/recipe-wheels/exllamav3-*.whl
RUN git clone https://github.com/vcruz305/vllm-exl3.git /opt/src/vllm-exl3 \
    && git -C /opt/src/vllm-exl3 checkout --detach d3cfd394920360d69f820d2dc96f8292a9e10283 \
    && test "$(git -C /opt/src/vllm-exl3 rev-parse HEAD)" = d3cfd394920360d69f820d2dc96f8292a9e10283
COPY runtime/native-setup.py /opt/src/vllm-exl3/setup.py
ENV EXL3_EXT_INCLUDE=/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext
WORKDIR /opt/src/vllm-exl3
# Same no-sitecustomize build invocation as the retained native extension.
RUN python3 -S -c 'import sys,runpy;sys.path.extend(["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages"]);sys.argv=["setup.py","bdist_wheel"];runpy.run_path("setup.py",run_name="__main__")' \
    && python3 -m pip install --no-index --no-deps --no-compile dist/vllm_exl3-*.whl \
    && cp dist/vllm_exl3-*.whl /opt/recipe-wheels/
# All retained writable-layer overlays are now part of the built image.
COPY runtime/rootfs/ /
COPY licenses/ /opt/recipe-licenses/
COPY runtime/source-manifest.json /opt/recipe/source-manifest.json
COPY pins.json /opt/recipe/pins.json
COPY scripts/artifacts.py scripts/runtime_check.py scripts/check_model.py scripts/verify.py /opt/recipe/
COPY tools/ /opt/recipe/tools/
COPY configs/source-model.json /opt/recipe/configs/source-model.json
COPY scripts/prepare.py /opt/recipe/prepare.py
# cuobjdump delegates SASS extraction to nvdisasm; the runtime base omits it.
RUN apt-get update && apt-get install -y --no-install-recommends cuda-nvdisasm-13-0=13.0.85-1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -S /opt/recipe/runtime_check.py --root / \
    && sha256sum /opt/recipe-wheels/*.whl > /opt/recipe/built-wheels.sha256
ENV VLLM_PLUGINS=vllm_exl3 DSV41_ENGRAM_DISK=1 \
    VLLM_EXL3_MOE_KERNEL=native DSV41_MXFP8_MODE=off
WORKDIR /opt/recipe
ENTRYPOINT ["vllm"]
CMD ["--help"]
