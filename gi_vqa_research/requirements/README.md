# Dependency policy

`pyproject.toml` defines the initial direct dependency set. The GPU versions mirror the
known-working T4 environment recorded in the organizer starter notebook; they are a reference
environment, not yet a validated Study 1 lock.

Before a paper-defining run:

1. build `docker/Dockerfile.gpu`;
2. run the complete smoke configuration;
3. record `pip freeze`, the image digest, CUDA and driver versions;
4. create and commit a platform lock file from that validated environment;
5. do not update the environment during a locked experiment series.

The local macOS Python installation is not the authoritative GPU environment.

