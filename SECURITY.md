# Security Policy

## Supported versions

This repository is currently pre-1.0. Security fixes target the latest commit
on `main`; older snapshots are not maintained.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Use GitHub's
private vulnerability reporting for this repository when available, or contact
the repository owner privately through their verified GitHub profile. Include
the affected component, reproduction steps, impact, and any suggested
mitigation. Please allow reasonable time for triage before public disclosure.

## Deployment boundaries

The included Compose stack is a local integration environment. Its example
credentials are not suitable for an exposed network. A real deployment must:

- use managed secrets and TLS;
- require authentication and bind the authenticated principal to `tenant_id`;
- restrict access to MLflow, PostgreSQL, and object storage;
- load model artifacts only from a trusted, access-controlled registry;
- enable rate limiting, audit retention, dependency scanning, and centralized
  monitoring;
- rotate credentials and review model promotion permissions.

Serialized model bundles can execute code when loaded. Treat model registry
write access as equivalent to application code-deployment access.
