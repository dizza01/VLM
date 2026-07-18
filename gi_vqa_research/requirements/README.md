# Dependency policy

`pyproject.toml` defines the initial direct dependency set. The GPU versions mirror the
known-working T4 environment recorded in the organizer starter notebook,
including exact `huggingface-hub` and `sentencepiece` versions; they are a
reference environment, not yet a validated Study 1 lock.

PyTorch is supplied by the execution platform rather than the `gpu` extra:
the reference Docker image pins `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`,
while the Colab contract test must record and verify its preinstalled PyTorch,
CUDA and driver versions before installing `.[gpu]`. Do not treat an arbitrary
PyTorch version as equivalent to the reference environment.

Before a paper-defining run:

1. build `docker/Dockerfile.gpu`;
2. run the complete smoke configuration;
3. verify deterministic PaliGemma generation, teacher-forced score parity and
   both attribution methods;
4. verify that the built-in ms-swift 3.7.0 token-type difference is the known
   one-token boundary defect and that the corrected project training template
   exactly matches the direct processor;
5. record `pip freeze`, the image digest, CUDA and driver versions;
6. create and commit a platform lock file from that validated environment;
7. do not update the environment during a locked experiment series.

The local macOS Python installation is not the authoritative GPU environment.
