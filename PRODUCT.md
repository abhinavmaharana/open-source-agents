# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated per project. The repository is framework-agnostic and language-neutral: each contribution under `agents/<slug>/` or `tools/<slug>/` chooses its own stack, and any UI it ships uses that project's existing stack. No shared site or repo-wide frontend exists or is planned.

## Users

Two equally primary audiences:

- **Contributors**, including participants in the FORK. BUILD. MERGE. initiative (Road to DevFest 2026, GDG Cloud Bengaluru), who want to add a working, documented agent or tool through a fork-and-PR workflow.
- **Builders browsing the catalog**, meaning developers and researchers looking for working reference implementations of agents, MCP servers, developer utilities, workflows and evaluation tools they can study and reuse.

## Product Purpose

`open-source-ai` is a community-driven monorepo that collects practical AI agents and developer tools. Each one is open source and documented. Success means that every project in it is reproducible on a clean machine, documented with the shared template, free of committed secrets and self-contained, so browsers can trust and reuse it and contributors can land work without friction.

## Positioning

The catalog is curated for working, documented projects in any framework. Contributions are reviewed against four standards: reproducibility, documentation quality, security and privacy, and focused scope. Being seeded at a community event but maintained as a lasting public catalog is part of its identity.

## Operating Context

- Fork, branch (`feat/<slug>`), add a project folder, open a PR against `main` and complete `.github/PULL_REQUEST_TEMPLATE.md`.
- New projects start from `template/` (`README.md`, `.gitignore`, `.env.example`).
- Detailed contributor steps live in `CONTRIBUTING.md`.

## Capabilities and Constraints

- Categories: AI agents, MCP servers and integrations, AI developer utilities, AI-powered workflows, evaluation/testing/safety utilities, and other practical AI tooling.
- No mandated cloud provider, framework or model vendor.
- Configuration goes through environment variables or sanitized config files; credentials, private data and virtual environments are never committed.
- Each contribution stays in its own directory and does not modify unrelated parts of the repository.
- **Design scope:** Impeccable work targets UIs inside individual contributions (per-project UIs), not a repo-wide site. Which projects will ship UIs is undecided.

## Brand Commitments

- Name: `open-source-ai`; tagline "Build it. Document it. Open-source it."
- Community attribution: GDG Cloud Bengaluru, FORK. BUILD. MERGE., Road to DevFest 2026.
- MIT License © 2026 GDG Cloud Bengaluru Community.

## Evidence on Hand

- `README.md`, `CONTRIBUTING.md`, `agents/README.md`, `tools/README.md`, `template/`, `.github/PULL_REQUEST_TEMPLATE.md`.
- No contributed agents or tools yet, and no logos, screenshots, testimonials, usage numbers or press. Do not fabricate any of these.

## Product Principles

1. Working beats impressive: a project must run as documented on a clean environment.
2. Documentation is part of the deliverable, not an afterthought.
3. Contributors choose their stack freely; consistency comes from the shared standards and template, not shared technology.
4. Keep every contribution isolated and secret-free.
5. Serve both newcomers making a first PR and experienced builders looking for reusable patterns.
