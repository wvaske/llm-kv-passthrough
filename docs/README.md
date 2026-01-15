# KV-Bench Documentation

Documentation for the KV-Bench distributed KV cache benchmarking system.

## Building Documentation

### Prerequisites

```bash
pip install mkdocs mkdocs-material
```

### Local Development

```bash
cd docs
mkdocs serve
```

Visit http://localhost:8001 to view documentation.

### Build Static Site

```bash
cd docs
mkdocs build
```

Output in `site/` directory.

## Documentation Structure

```
docs/
├── mkdocs.yml                    # MkDocs configuration
├── index.md                      # Home page
├── getting-started/
│   ├── installation.md           # Installation guide
│   ├── quickstart.md             # Quick start tutorial
│   └── configuration.md          # Configuration reference
├── architecture/
│   ├── overview.md               # System architecture
│   ├── gpu-emulation.md          # GPU emulation details
│   ├── storage.md                # Storage backends
│   └── connectors.md             # KV cache connectors
├── deployment/
│   ├── docker.md                 # Docker deployment
│   └── ansible.md                # Ansible deployment
└── benchmarking/
    ├── genai-perf.md             # GenAI-Perf integration
    └── lmcache.md                # LMCache integration
```

## Contributing

1. Follow Markdown best practices
2. Add code examples where relevant
3. Test all code snippets
4. Update navigation in `mkdocs.yml`
