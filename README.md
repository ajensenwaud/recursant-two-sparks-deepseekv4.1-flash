# DeepSeek V4.1 Flash with vision on two Sparks
It is not fast, but it works. It is my first quantisation so bear with me.
## Install

On the head node or a Linux coordinator with SSH access to **both** nodes:

```bash
git clone https://github.com/ajensenwaud/recursant-two-sparks-deepseekv4.1-flash.git
cd recursant-two-sparks-deepseekv4.1-flash
./start.sh
```

The first run asks for a deployment name and, for each node:

- Existing SSH host alias or `user@host` (including the head, even when running there).
- Fabric IPv4 address, network interface and RDMA HCA. These are separate settings; the installer does not guess them.
- Absolute model directory and a separate, dedicated writable cache directory, with no symlink aliases or overlapping paths.

It saves `configs/cluster.local.json`, checks both nodes before installing, downloads the pinned runtime and model **directly on each node**, verifies them once, then starts worker and head through the coordinated launcher. The model is mounted read-only. Nothing is compiled or converted by this installer, and no host packages, drivers, credentials, firewall rules or kernel modules are installed or changed.

The first model download is **358,128,021,424 bytes per node** and may take hours. Keep the coordinator connected. Interrupted files resume when you rerun the same command. Each file is SHA256-checked before finalization; the complete model structure is checked before an installation receipt is written to the cache. Existing model files are checked without creating lock/receipt files in the model directory. Disk checks include missing model bytes, runtime archive/storage requirements and a 10 GiB reserve; the image-load budget is conservative. The runtime archive, if used, remains in the cache for recovery.

Normal `./start.sh` calls do **not** rehash or scan weights. A healthy owned deployment is left untouched after ownership and health checks. After `./stop.sh`, the same owned containers can restart without downloads or payload scans. Partial, changed, OOM-failed or foreign-owned deployments fail closed; unrelated GPU work and existing container names are never stopped, removed or replaced. Keep the ignored `.state/` directory: it records the exact container IDs and ownership needed for recovery. Do not edit model files after installation; installation receipts are not a continuous tamper monitor.

### Prerequisites

- Python **3.11+** and OpenSSH on the coordinator; Python 3.11+, Linux ARM64, Docker access without sudo, compatible NVIDIA drivers, NVIDIA Container Toolkit, `nvidia-smi` and `ip` on both Sparks.
- Working RDMA devices/fabric and SSH public-key access. Verify each server's host key yourself first. The recipe enforces `StrictHostKeyChecking=yes` and batch authentication; it never accepts unknown keys or collects credentials.
- Local NVMe space on both nodes for the exact model plus the image and cache. Dedicated model/cache directories must be writable for a fresh download; a complete existing model may be read-only. GPU workloads must be idle for first launch or restart.
- Trusted nodes and fabric. The retained configuration uses host networking/IPC, RDMA access, unlimited memlock, `IPC_LOCK` and a SELinux label override.

If a prerequisite is missing, the installer reports it and exits. Install/configure it yourself and rerun; no sudo fallback is attempted.

### Configuration and access

```bash
./start.sh --configure-only                   # interactive, save without SSH/install
./start.sh --config /path/to/cluster.json     # unattended using an existing configuration
./status.sh
./stop.sh
python3 -B scripts/dsv41.py logs
python3 -B scripts/dsv41.py verify            # explicit text/reasoning/two-PNG requests
```

`--config` also applies to stop/status and the Python frontend. Configuration is validated before network access. Noninteractive first runs without a config fail with instructions, rather than guessing. Help and cloning have no deployment side effects. `--release PATH` selects a reviewed release metadata file, not an unpinned URL.

API access defaults to **head-node `127.0.0.1:8000`**. Use your own SSH tunnel or authenticated TLS proxy for remote clients. Only to preserve an **already approved** LAN endpoint, first configure with `./start.sh --configure-only --preserve-existing-public-bind`; this narrowly records `0.0.0.0:8000`. No exposure is enabled silently and no firewall is changed. Never put credentials in configuration, Git, build arguments or logs. `RECIPE_API_KEY` is an optional client-only variable. API clients reject redirects; logs may contain user requests and must not be published.

## Pinned artifacts

The [exact retained model](https://huggingface.co/ajensenwaud/recursant-two-sparks-deepseekv4.1-flash/tree/d032e578f9ed3724e24239a79d408772620b1d9d) is pinned at revision `d032e578f9ed3724e24239a79d408772620b1d9d`: 53 retained files, including 48 shards and metadata, with exact sizes/SHA256 in `configs/release.json`. Anonymous Hub readback verified the full inventory and shard LFS hashes; bounded downloads verified all five retained metadata files. The entire model download was not repeated for that publication check.

Model publication is independent of image publication. You can download the model now on either node without cluster configuration:

```bash
python3 -B scripts/dsv41.py download-model --destination /srv/models/dsv41-exl3-mcg2
```

The complete release supports a registry image pinned by manifest digest, or a Docker archive pinned by immutable Hub revision, byte count and SHA256. Archive installation verifies the loaded ARM64/Linux image's complete Config, ordered RootFS layer digests and retained-source label. Docker storage backends can report different image IDs for identical content; launch pins the inspected immutable ID on each node. A local image-config ID is never presented as a registry digest. Model/image publication status and actual fresh-install verification are distinct claims.

The release is `published`: the [runtime archive and source/notices companion](https://huggingface.co/ajensenwaud/recursant-two-sparks-deepseekv4.1-flash-runtime/tree/e1cd8203d398cde316bf98a7ac50a043adc869e2) are pinned at `e1cd8203d398cde316bf98a7ac50a043adc869e2`. The Docker archive is 10,695,250,306 bytes with SHA256 `38402f4e4d1033eff2f7f2883afca292566fe2222edb91670d8c9e8d43bccf08`. Anonymous immutable readback verified publication; actual public-download/load/start acceptance is still pending. Runtime source fingerprints are unchanged by installer-only edits.

## Retained serving controls

TP2; native MCG decode with ExLlama fallback; dense MXFP8 with the retained narrow FlashInfer autotuner; original BF16 output head; K5 probabilistic drafting with ordinary target verification; vision enabled; HIGH thinking; 1,048,576 configured context; eight slots; 4 GiB KV per rank; 2,048 batched tokens.

Startup retains coordinated locks, exact ownership/configuration and mount checks, image-content parity, bounded readiness, a 6 GiB available-memory reserve and no-new-host-OOM guard. On failure it stops only the exact newly started owned IDs. These startup guards are not a continuous watchdog or an OOM-proof guarantee. Full 1M occupancy, broad quality and every host configuration are not certified. Explicit `verify` sends the smoke requests; ordinary start does not benchmark or send generation requests.

## Advanced local prebuilt use and reproduction

These paths are not needed for the normal installer. To use a clean image already installed on both nodes and an existing retained model:

```bash
python3 -B scripts/dsv41.py configure --local-image dsv41-vision:local-prebuilt
# Advanced path only: edit the generated local config for your existing nodes/paths.
python3 -B scripts/dsv41.py plan
python3 -B scripts/dsv41.py start
python3 -B scripts/dsv41.py verify
```

Explicit local-prebuilt mode is separate from a published release and does not pull or convert anything. Transfer only a clean image with `docker save IMAGE_TAG | ssh OTHER_NODE docker load`; never export or commit a running container. Existing locally approved public binds require the explicit configure flag described above.

CPU checks and advanced dry-runs:

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/build.py --dry-run
python3 -B scripts/download.py --dry-run
python3 -B scripts/prepare.py --source /source --destination /output --dry-run
```

`runtime/Dockerfile` pins the public base digest and plugin/ExLlama commits, retains modified source and licenses, and records wheel hashes. `scripts/build.py --execute --acknowledge-build-gaps --tag dsv41-vision:local-review` explicitly builds; compiler parallelism of two does not itself enforce a memory limit. Use a resource-enforcing builder under approved scope, never an uncapped build alongside serving.

`download.py` obtains the official source checkpoint, not the retained pack. `prepare.py --execute --acknowledge-unvalidated-conversion` requires a GPU runtime and separate read-only source/writable output mounts. Budget source (~510 GB), full output, at least 40 GiB reserve and image/cache storage. Conversion uses 2-bit MCG, one worker, batch one and greedy beam 16, preserving draft/non-routed tensors. A new conversion is not byte-equivalent or quality-certified by structure alone. No source deletion/replacement is automatic. `check_model.py --model PATH --checksums` checks conversion receipts when available; `benchmark.py --help` documents explicit usage-based measurements.

## License and release evidence

Original integration code is **AGPL-3.0-only**; see `LICENSE` and `NOTICE`. Upstream licenses/notices, including Mia's AI Lab and the AGPL plugin attribution, are preserved. Model weights remain under their separate MIT terms. This choice does not waive corresponding-source or CUDA/base redistribution obligations.

The public base digest resolves, but its metadata reports build commit `unknown`; installed vLLM identifies `179dd0fa9` and a local wheel path rather than obtainable full corresponding source. That provenance limitation remains disclosed. APT artifact snapshots are also a source-rebuild limitation; neither is disguised as a fabricated checksum or a successful rebuild.

Historical local evidence includes a clean ARM64 build, in-image CPU tests, and the local-prebuilt configure → plan → start → verify path on both GPUs. Text, reasoning and two PNG smokes passed before/after five measured concurrency cells (1, 2, 3, 4 and 8), using 60-second windows, 8,192-token prompts and substantial prefix-cache reuse. Those are bounded local measurements, not proof of a clean external download or general answer quality. CPU installer tests use synthetic artifacts and mocked Docker/SSH; they do not prove a real public image load or GPU startup.

Private audits, host identities, research archives, credentials, model payloads and image archives are excluded from this repository.
