# DeepSeek V4.1 vision on two GB10 systems

Prebuilt-first deployment of the retained **EXL3 MCG 2-bit / native decode / DSpark K5 / vision-enabled** configuration on two 128 GB ARM64 GB10 nodes.

**Source public; local GPU deployment verified; model upload pending; registry image unpublished.** No downloadable release image or verified Hub model revision is recorded in `configs/release.json` yet. Its null values are deliberate: artifact commands fail before networking or starting services, rather than inventing artifacts or substituting a different pack. Cloning/importing does nothing to the system. Local validation is not public artifact availability or broad quality certification.

## Normal installation: configure → obtain → start → verify

Once an independently validated release has real immutable pins:

```bash
python3 -B scripts/dsv41.py configure
# Edit configs/cluster.local.json: both SSH hosts, fabric IPs,
# interfaces, HCAs, model paths and dedicated cache paths.
python3 -B scripts/dsv41.py pull
# On EACH node, using the same release checkout:
python3 -B scripts/dsv41.py download-model --destination /srv/models/dsv41-exl3-mcg2
# On the coordinator:
python3 -B scripts/dsv41.py plan
python3 -B scripts/dsv41.py start
python3 -B scripts/dsv41.py status
python3 -B scripts/dsv41.py logs
python3 -B scripts/dsv41.py verify
python3 -B scripts/dsv41.py stop
```

Today `configure` writes `UNPUBLISHED`; obtaining/starting through the registry-release path fails with an actionable explanation. Do not change the release status or manufacture pins to bypass this gate. After image publication, regenerate configuration or update its image to the exact published digest. `--config` and `--release` accept explicit files. There is **no implicit source build or conversion**. Pull verifies ARM64/Linux, the requested digest and retained-source fingerprint on both nodes. Model retrieval uses one exact revision and SHA256 allowlist, resumes interrupted files and validates model structure. Run it on each node; it does not download through the coordinator.

Model publication is independent of image publication. Once the uploader has
verified the remote files and immutable Hub commit, release metadata can record
`model.status: "published"`, its exact repository/revision/file manifest and the
matching `source_manifest_sha256`, while leaving overall `status: "unpublished"`
and `image: null`. Then `download-model --destination ...` works without a cluster
configuration, SSH or a published image. A fully published release remains
supported. This code path is CPU-tested with synthetic pins; the real model
download is **not yet verified**. Until the verified metadata is recorded, it
continues to fail closed. Model-only publication does not authorize registry
`pull`, `plan` or `start`; the explicit local-prebuilt path remains separate.

Start retains coordinated locks, ownership checks, strict remote mount validation, image-content checks, bounded startup, a 6 GiB available-memory reserve and no-new-host-OOM guard. It refuses occupied GPUs or existing names, creates worker before head, and waits for readiness. Run explicit verify for text/reasoning/two-PNG verification. Failure stops only newly owned IDs while locks remain held. Stop still works if the model directory has disappeared; it does not require published release metadata. Stopped containers/state remain for inspection: no automatic replacement or deletion.

## Local prebuilt installation (available without a published release)

Use a locally built ARM64 image's full `sha256:...` Docker image ID or an explicit
local tag, already installed on both nodes. Transfer only that clean image with
`docker save IMAGE_TAG | ssh OTHER_NODE docker load`. Docker storage backends may
report different IDs for the same saved/loaded image; in that case give both local
images the same tag. Never export or commit a running container.

```bash
python3 -B scripts/dsv41.py configure --local-image dsv41-vision:local-prebuilt
# Edit configs/cluster.local.json with the existing exact retained model_path
# on each node, and distinct dedicated cache_path directories.
python3 -B scripts/dsv41.py plan
python3 -B scripts/dsv41.py start
python3 -B scripts/dsv41.py verify
```

`--local-image` is only for configure; subsequent commands use the saved config.
It records `local_prebuilt: true`; plan/start need no published release metadata.
It does not invent a registry release, pull an image, download/convert a model,
or certify GPU behavior. `pull` rejects local prebuilt images. The optional
`local_prebuilt` field must be a boolean; existing configurations, including
legacy SHA-ID local configurations, remain supported. Other fields are deployment, image, api_bind, api_port, master_port,
startup_timeout_seconds, and two nodes, each with ssh_host, fabric_ip,
socket_interface, rdma_hca, model_path, cache_path. Use absolute model/cache paths.

Installation/build validates imports and source overlays once. Normal start does
not rehash model payloads, disassemble vendor libraries, run import probes, or
send inference requests. It accepts identical image IDs, or (for differing IDs)
requires identical inspected Config, nonempty RootFS layer digests and platform
(OS, architecture and variant). Missing content metadata fails closed. This uses
only the existing image inspections, not extra remote checks. Each container is
created with its own node's inspected immutable ID, never the mutable tag, retaining ownership,
mount, lock, available-memory/OOM and readiness guards. Explicit `verify` sends
the four text, reasoning, red-square-CAT and blue-circle-DOG smoke requests.
Existing retained pack hashes are one-time provenance evidence, not a launch hook.

## Requirements and retained controls

- Existing Linux ARM64, compatible NVIDIA driver, Docker/NVIDIA Container Toolkit, Python 3.11+, host-key-verified SSH and working RDMA on both nodes. No host installation, firewall changes, sudo cache drops or credential setup is automated.
- Local NVMe holding the **same exact** 48-shard, 188,245-tensor pack on each node: 358,107,269,776 shard bytes, plus metadata and runtime/cache space. No weights are bundled. A similarly named public quantization is not a substitute.
- Dedicated trusted nodes and fabric; initial validation requires exclusive GPUs. Host networking/IPC, RDMA device access, unlimited memlock, IPC_LOCK and the retained SELinux-label override remain necessary parts of this recipe.
- TP2; native MCG decode with ExLlama fallback; dense MXFP8 with the retained narrow FlashInfer autotuner; original BF16 head; K5 probabilistic drafting and ordinary target verification; vision enabled; HIGH thinking; 1,048,576 configured context, eight slots, 4 GiB KV/rank, 2,048 batched tokens.

The local GPU deployment exercised read-only model mounts, offline Hub and no remote-code trust/profiler flags, while explicitly preserving an already approved public bind; that does not validate every new host or default-loopback deployment. The startup reserve is not a continuous watchdog or an OOM-proof guarantee. Full 1M occupancy and broad vision quality are not certified. Measured throughput is bounded workload evidence, not a hardware guarantee.

API access is head-node loopback; use a user-managed SSH tunnel or explicitly authenticated TLS proxy for remote clients. `RECIPE_API_KEY` is an optional client-only environment variable. Never put credentials in configuration, Git, build arguments or logs. API clients reject redirects. Logs can contain user requests: do not publish them.

## Advanced: local build, conversion and measurement

These are reproduction tools, **not the default install**:

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/build.py --dry-run
python3 -B scripts/download.py --dry-run
python3 -B scripts/prepare.py --source /source --destination /output --dry-run
```

`runtime/Dockerfile` pins the public base digest and plugin/ExLlama commits. It retains modified source, licenses and built wheel hashes. The compiler uses two jobs; this alone is not a memory limit. Use a resource-enforcing builder on an approved machine, verify its actual cgroups, and do not run an uncapped build alongside serving. `scripts/build.py --execute --acknowledge-build-gaps --tag dsv41-vision:local-review` explicitly builds but does not enforce host-memory limits or start the model. Local validation used a separate sterile context and checked 8 GiB/2-CPU cgroups before dependency installation; build success and GPU validation must be reported separately.

`download.py --destination /srv/models/dsv41-source` retrieves pinned official **source** assets, not the retained converted pack. `prepare.py --execute --acknowledge-unvalidated-conversion` requires the built GPU container with read-only source and separate writable output mounts. Budget source (~510 GB) + complete output + at least 40 GiB reserve, as well as image/cache storage. It uses 2-bit MCG, one worker, batch 1, greedy beam 16; draft/non-routed tensors are preserved and receipts checked. A new conversion is not byte-equivalent or quality-certified merely because structure passes. No automatic deletion or source replacement occurs.

Advanced local builds use `scripts/cluster.py configure|plan|start|status|stop`; start requires `--execute --acknowledge-unreleased`. The same frontend supports local prebuilt IDs and tags as documented above. Both paths share the same safety implementation. Do not use an occupied production pair. `scripts/check_model.py --model PATH --checksums` validates conversion receipts when available; header checks alone are not full-payload checks. `scripts/benchmark.py --help` documents explicit single-stream usage-based measurement, not a quality test.

## Release gates

The original integration license is **not selected**. Preserve `LICENSE`, `NOTICE` and all component notices, including Mia's AI Lab and AGPL plugin attribution. Owner license choice does not waive corresponding-source or CUDA/base redistribution obligations. The public base digest resolves, but its metadata says build commit `unknown`; installed vLLM identifies `179dd0fa9` and a local build-wheel path, not obtainable full corresponding source. Source provenance remains unresolved.

Completed local evidence: a clean ARM64 image build and 64 in-image CPU tests;
later frontend suites discovered 67 tests (66 passed, one skipped because Torch
was absent locally). The actual local-prebuilt configure → plan → start → verify
path ran on both GPUs, with text, reasoning and two PNG smokes passing before and
after five measured concurrency cells (1, 2, 3, 4 and 8). Those were 60-second
sustained windows with 8,192-token prompts and substantial prefix-cache reuse;
they do not certify full-context occupancy, general answer quality or every
kernel path. These are separate historical observations, not claims that this
frontend-only change rebuilt or reran the GPU deployment.

CPU tests and mocked registry/SSH fixtures do not establish real pulls or model
publication. Remaining image-release gates include source/license clearance,
distribution review and authenticated registry publication with anonymous
digest readback. Model publication separately needs the uploader's verified
remote allowlist and immutable revision; no Hub metadata is invented here.
The unresolved original integration license is not a blocker to the separate
MIT-licensed model upload. APT artifact snapshots remain a disclosed
source-rebuild limitation, not a fabricated blocker to a separately validated
prebuilt image. Further numerical/graph and broad quality claims require their
own evidence.

Private audits, host identities, research archives, credentials, model payloads and image archives are excluded from this tree.
