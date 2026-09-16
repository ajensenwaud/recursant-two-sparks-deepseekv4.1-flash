#!/usr/bin/env python3
"""Disk-backed Engram tables for DGX Spark unified memory.  + host-RAM row cache.

Original reader (unchanged behaviour when the cache is off): preads only the
hashed rows from NVMe safetensors shards, dequant CPU fp8 e4m3 x ue8m0 -> bf16.

MEASURED PROBLEM (2026-09-13, 2x DGX Spark, TP=2):
  49 NVMe reads/token, ~4.1 KB each, 627 IOPS sustained, 2.65 MB/s.
  At ~1.6 ms per read that is ~78 ms/token -- i.e. essentially the whole decode
  step time (13.7 tok/s). The GPU idles waiting on small serialised reads.
  Bandwidth is a non-issue (0.245 MB/token): this path is LATENCY-bound.

FIX: per-table LRU of raw row bytes held in host RAM.  UMA has ~100 GB unused
(KV cache usage measured at 0.1%), so caching rows is nearly free here.

Env:
  DSV41_ENGRAM_DISK=1            enable the disk reader (as before)
  DSV41_ENGRAM_CACHE_MB=<int>    RAM budget PER TABLE, MB. Default 256. 0 = off.
                                 49 Engram layers x 256 MB ~= 12.5 GB of the ~100 GB
                                 of unused UMA. Byte-budgeted so it is independent of dim.
  DSV41_ENGRAM_CACHE_DEBUG=1     log hit rate periodically
"""
from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

_DSV41_ENGRAM_DISK = os.environ.get("DSV41_ENGRAM_DISK", "0") == "1"


def _torch():
    import torch

    return torch


def engram_disk_enabled() -> bool:
    return os.environ.get("DSV41_ENGRAM_DISK", "0") == "1"


class DiskEngramTable:
    """Pread rows from safetensors shards. Never materializes the full table."""

    def __init__(self, model_dir: str, layer_id: int, dim: int, block_size: int):
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        wname = f"layers.{layer_id}.engram.embed.weight"
        sname = f"layers.{layer_id}.engram.embed.scale"
        if wname not in weight_map or sname not in weight_map:
            raise FileNotFoundError(
                f"Engram tensors missing from index: {wname} / {sname}"
            )
        self.w_fd, self.w_off, self.w_shape = self._open(
            model_dir, weight_map[wname], wname
        )
        self.s_fd, self.s_off, self.s_shape = self._open(
            model_dir, weight_map[sname], sname
        )
        self.dim = dim
        self.sb = dim // block_size
        if self.w_shape[1] != dim:
            raise ValueError(f"engram weight width {self.w_shape} != dim {dim}")
        if self.s_shape[1] != self.sb:
            raise ValueError(f"engram scale width {self.s_shape} != {self.sb}")
        self.threads = int(os.environ.get("DSV41_ENGRAM_DISK_THREADS", "32"))
        self.chunk = int(os.environ.get("DSV41_ENGRAM_DISK_CHUNK", "16"))
        self.pool = ThreadPoolExecutor(max_workers=self.threads)

        # ---- host-RAM row cache -------------------------------------------
        self._cache_max_bytes = int(
            float(os.environ.get("DSV41_ENGRAM_CACHE_MB", "256")) * 1024 * 1024
        )
        self._cache = OrderedDict() if self._cache_max_bytes > 0 else None
        self._cache_bytes = 0
        self._debug = os.environ.get("DSV41_ENGRAM_CACHE_DEBUG", "1") == "1"
        self._hits = 0
        self._misses = 0
        self._calls = 0

    @staticmethod
    def _open(model_dir: str, fname: str, tname: str):
        path = os.path.join(model_dir, fname)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except OSError:
            pass
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        meta = hdr[tname]
        start = meta["data_offsets"][0]
        return fd, 8 + n + start, tuple(meta["shape"])

    def _read_rows_direct(self, fd: int, base: int, rel: list[int], row_bytes: int, buf) -> None:
        """Original path: one preadv per row, fanned out over the thread pool."""
        def work(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                off = base + rel[i] * row_bytes
                view = buf[i * row_bytes : (i + 1) * row_bytes]
                got = 0
                while got < row_bytes:
                    n = os.preadv(fd, [view[got:]], off + got)
                    if n <= 0:
                        raise OSError("engram disk table: short read")
                    got += n

        n = len(rel)
        if n <= self.chunk:
            work(0, n)
            return
        futs = [
            self.pool.submit(work, lo, min(lo + self.chunk, n))
            for lo in range(0, n, self.chunk)
        ]
        for fut in futs:
            fut.result()

    def _read_rows(self, fd: int, base: int, rel: list[int], row_bytes: int, buf) -> None:
        cache = self._cache
        if cache is None:
            self._read_rows_direct(fd, base, rel, row_bytes, buf)
            return

        self._calls += 1
        n = len(rel)
        miss: list[int] = []
        for i in range(n):
            b = cache.get((fd, rel[i]))
            if b is None:
                miss.append(i)
            else:
                buf[i * row_bytes : (i + 1) * row_bytes] = b
                self._hits += 1
                cache.move_to_end((fd, rel[i]))

        if miss:
            self._misses += len(miss)
            tmp = bytearray(len(miss) * row_bytes)
            self._read_rows_direct(
                fd, base, [rel[i] for i in miss], row_bytes, memoryview(tmp)
            )
            for j, i in enumerate(miss):
                chunk = bytes(tmp[j * row_bytes : (j + 1) * row_bytes])
                buf[i * row_bytes : (i + 1) * row_bytes] = chunk
                key = (fd, rel[i])
                cache[key] = chunk
                self._cache_bytes += len(chunk)
                while self._cache_bytes > self._cache_max_bytes and len(cache) > 1:
                    _, evicted = cache.popitem(last=False)
                    self._cache_bytes -= len(evicted)

        if self._debug and self._calls % 200 == 0:
            tot = self._hits + self._misses
            print(
                f"[engram-cache] calls={self._calls} hit_rate={self._hits / max(tot, 1):.3f} "
                f"entries={len(cache)} bytes={self._cache_bytes} "
                f"({self._cache_bytes / 1048576:.1f} MB)",
                flush=True,
            )

    def gather_dequant(self, rel, owned):
        """rel: [R] int64 CPU local row ids; owned: [R] bool. Returns [R, dim] bf16 CPU."""
        torch = _torch()
        r = int(rel.numel())
        w = torch.empty((r, self.dim), dtype=torch.uint8)
        s = torch.empty((r, self.sb), dtype=torch.uint8)
        rel_l = rel.tolist()
        self._read_rows(
            self.w_fd, self.w_off, rel_l, self.dim, memoryview(w.numpy()).cast("B")
        )
        self._read_rows(
            self.s_fd, self.s_off, rel_l, self.sb, memoryview(s.numpy()).cast("B")
        )
        vals = w.view(torch.float8_e4m3fn).to(torch.float32).view(r, self.sb, -1)
        scale = (s.to(torch.int32) << 23).view(torch.float32)
        out = (vals * scale[:, :, None]).reshape(r, self.dim)
        out[~owned] = 0
        return out.to(torch.bfloat16)


def is_engram_embed_tensor(name: str) -> bool:
    """Checkpoint names the disk reader (and the loader skip) must ignore."""
    return name.endswith((".engram.embed.weight", ".engram.embed.scale"))
