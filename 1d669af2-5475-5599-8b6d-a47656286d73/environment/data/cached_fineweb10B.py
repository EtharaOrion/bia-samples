#!/usr/bin/env python3
"""The pinned upstream FineWeb10B loader, vendored into this bundle as a build input.

`environment/bootstrap.py` names this file at `data/cached_fineweb10B.py`, refuses to
reimplement it, and refuses to run without it. It was never shipped, so both image
builds died on `the pinned upstream loader data/cached_fineweb10B.py is absent from
this image`. This file is that build input.

It is the loader from modded-nanogpt `data/cached_fineweb10B.py`: the same dataset
repository, the same filenames, the same destination rule of
`dirname(__file__)/fineweb10B`, and the same "skip what is already present" behaviour.
Every byte it lands is upstream's own byte. Nothing here generates a token.

ONE DELIBERATE DIVERGENCE FROM UPSTREAM, AND IT IS A SPLIT FENCE. Upstream fetches
`fineweb_val_000000.bin` unconditionally before the train chunks, because upstream runs
one training script that needs both. This bundle does not: the validation split is the
graded surface and belongs to the verifier's image alone. So the val fetch here sits
behind an explicit `--val` flag and happens on no other path.

`environment/bootstrap.py stage_corpus` invokes this file as
`[sys.executable, loader, str(int(shards))]`. It passes a shard COUNT and it cannot
pass a flag, so the agent-visible corpus stage has no expressible route to the
validation split. The absence on that surface is a matter of what this file was never
asked for rather than of a filter that could be misconfigured.

ON THE FETCH ITSELF. Upstream calls `huggingface_hub.hf_hub_download`. That package is
not in the digest-pinned base image and the image is PEP 668 externally managed, so a
`pip install` of it is a build step that fails on this base rather than a provisioning
detail. The package is used when it is present, and otherwise the same file is streamed
from the resolve URL `hf_hub_download` itself resolves to, for the same repository, the
same revision and the same filename. Either route lands the same bytes and both are
checked against the same header and byte count before anything reads them.

Usage:
    python3 cached_fineweb10B.py N        stage train chunks 1..N
    python3 cached_fineweb10B.py --val    stage the validation shard, verifier only
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request

REPO_ID = "kjj0/fineweb10B-gpt2"
REPO_TYPE = "dataset"
REVISION = "main"
RESOLVE = "https://huggingface.co/datasets/" + REPO_ID + "/resolve/" + REVISION + "/"

# The shard layout environment/model.py read_shard binds: a 1024-byte header of int32
# carrying the magic, the version and the token count, then uint16 GPT-2 BPE token ids.
SHARD_BYTES = 200001024
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def local_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fineweb10B")


def verify(path: str) -> str:
    """Refuse a shard that is not the upstream artifact, and return its digest.

    Checked rather than assumed, because a truncated or redirected download that landed
    a short file would otherwise reach the graded path as a corpus. The header is read
    with the same magic and version environment/model.py enforces, so a file that gets
    past this one gets past that one too.
    """
    size = os.path.getsize(path)
    if size != SHARD_BYTES:
        raise SystemExit(
            "staged shard " + path + " is " + str(size) + " bytes and the upstream shard is "
            + str(SHARD_BYTES)
        )
    import array  # noqa: PLC0415

    header = array.array("i")
    with open(path, "rb") as handle:
        header.fromfile(handle, HEADER_INTS)
    if sys.byteorder != "little":
        header.byteswap()
    if header[0] != SHARD_MAGIC or header[1] != SHARD_VERSION:
        raise SystemExit("staged shard " + path + " does not carry the bound header")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(fname: str, root: str) -> None:
    os.makedirs(root, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError:
        partial = os.path.join(root, fname + ".partial")
        with urllib.request.urlopen(RESOLVE + fname, timeout=300) as response:
            if response.status != 200:
                raise SystemExit("upstream answered " + str(response.status) + " for " + fname)
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 22)
        os.replace(partial, os.path.join(root, fname))
        return
    hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type=REPO_TYPE, local_dir=root)


def get(fname: str) -> str:
    root = local_dir()
    target = os.path.join(root, fname)
    if not os.path.exists(target):
        fetch(fname, root)
    digest = verify(target)
    print(fname + " " + digest, flush=True)
    return digest


def main(argv) -> int:
    if argv and argv[0] == "--val":
        get("fineweb_val_%06d.bin" % 0)
        return 0
    num_chunks = int(argv[0]) if argv else 103
    for i in range(1, num_chunks + 1):
        get("fineweb_train_%06d.bin" % i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
