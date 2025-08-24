# LEANN Vector Database Development Guide

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Test the Repository

**CRITICAL**: Set timeouts to 60+ minutes for builds, 30+ minutes for tests. NEVER CANCEL long-running commands.

```bash
# Install system dependencies (Linux/Ubuntu)
sudo apt-get update
sudo apt-get install -y libomp-dev libboost-all-dev protobuf-compiler libzmq3-dev \
  pkg-config libabsl-dev libaio-dev libprotobuf-dev patchelf build-essential cmake \
  libopenblas-dev libblas-dev liblapack-dev

# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR: pip install uv

# Clone and setup
git submodule update --init --recursive  # Required for C++ backends

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate

# Install build dependencies
uv pip install scikit-build-core pybind11 numpy cmake

# Build packages (TIMING CRITICAL - builds take 4+ minutes, NEVER CANCEL)
cd packages/leann-core && uv build        # Fast: ~2 seconds
cd ../leann-backend-hnsw && uv build      # Slow: ~4+ minutes. NEVER CANCEL. Set timeout to 60+ minutes.

# DiskANN backend requires Intel MKL - may fail without it
cd ../leann-backend-diskann && uv build   # May fail without Intel MKL

# Build meta package
cd ../leann && uv build                    # Fast: ~2 seconds

# Install packages
uv pip install packages/leann-core/dist/*.whl
uv pip install packages/leann-backend-hnsw/dist/*.whl
uv pip install packages/leann/dist/*.whl

# Install test dependencies
uv pip install -e ".[test]"
```

### Run Tests - NEVER CANCEL, can take 15+ minutes

```bash
# Quick minimal tests (~5 seconds)
pytest tests/test_ci_minimal.py -v

# Basic functionality tests (~30 seconds, may skip tests in CI environment)
pytest tests/test_basic.py -v

# Full test suite (15+ minutes. NEVER CANCEL. Set timeout to 30+ minutes)
pytest tests/ -v

# Run specific test categories
pytest tests/ -m "not slow"           # Skip slow tests
pytest tests/ -m "not openai"         # Skip tests requiring OpenAI API key
```

### Lint and Format - Always run before committing

```bash
# Install ruff linter
uv tool install ruff

# Lint code (required for CI, ~0.1 seconds)
ruff check .

# Format check (required for CI, ~0.1 seconds) 
ruff format --check .

# Auto-fix format issues
ruff format .
```

### CLI Usage and Validation

```bash
# Test CLI installation
leann --help

# Basic functionality (requires internet for model download)
leann build my-docs --docs ./documents
leann search my-docs "your query"
leann list
```

## Validation

**MANUAL VALIDATION REQUIREMENT**: After building, you MUST test actual functionality by running complete user scenarios.

- The application requires **internet access** to download embedding models (e.g., facebook/contriever from HuggingFace)
- Basic CLI functionality can be tested without internet using `leann --help`
- Full functionality testing requires network access for model downloads
- Always test the build → search → list workflow after making changes
- ALWAYS run `ruff check .` and `ruff format --check .` before committing or CI will fail

## Build System Architecture

### Package Structure
- **leann-core**: Pure Python package, fast build (~2 seconds)
- **leann-backend-hnsw**: C++ compilation with Faiss, slow build (~4+ minutes)
- **leann-backend-diskann**: C++ compilation requiring Intel MKL, may fail without MKL
- **leann**: Meta package that depends on core and backends

### Critical Timing Information
- **NEVER CANCEL**: HNSW backend build takes 4+ minutes, this is NORMAL
- **NEVER CANCEL**: Test suite takes 15+ minutes, this is NORMAL  
- **NEVER CANCEL**: Set explicit timeouts of 60+ minutes for build commands and 30+ minutes for test commands
- Core package build: ~2 seconds
- Meta package build: ~2 seconds
- Linting: ~0.1 seconds
- CI minimal tests: ~5 seconds

### Dependencies
- **System**: CMake, OpenMP, Boost, Protocol Buffers, ZeroMQ, BLAS/LAPACK
- **Build**: scikit-build-core, pybind11, numpy, cmake
- **Runtime**: PyTorch, sentence-transformers, various ML libraries
- **DiskANN backend**: Requires Intel MKL (may not be available in all environments)

## Common Tasks

### Development Workflow
1. Make changes to source code
2. Rebuild affected packages: `cd packages/[package] && uv build`
3. Reinstall: `uv pip install packages/[package]/dist/*.whl --force-reinstall`
4. Test: `pytest tests/test_ci_minimal.py -v`
5. Lint: `ruff check . && ruff format --check .`
6. Test full functionality if models are available

### CI/CD Alignment
The CI pipeline (.github/workflows/build-reusable.yml) includes:
- Linting and format checking with ruff
- Multi-platform builds (Ubuntu, macOS) 
- Python version matrix (3.9-3.13)
- Wheel building and repair (auditwheel/delocate)
- Full test suite execution

Match these steps locally:
```bash
# Same as CI
ruff check .
ruff format --check .
pytest tests/ -v
```

### Examples and Applications
- Check `examples/` directory for working usage patterns
- `examples/basic_demo.py`: Simple API usage
- `examples/spoiler_free_book_rag.py`: Document processing example
- Run examples with: `python examples/basic_demo.py`

### Key Project Structure
```
.
├── packages/
│   ├── leann-core/           # Core Python functionality
│   ├── leann-backend-hnsw/   # HNSW C++ backend (slow build)
│   ├── leann-backend-diskann/# DiskANN C++ backend (requires MKL)
│   └── leann/               # Meta package
├── tests/                   # Test suite
├── examples/                # Usage examples
├── docs/                    # Documentation
├── .github/workflows/       # CI/CD pipelines
└── scripts/build_and_test.sh # Build automation script
```

### Troubleshooting

**Build Issues:**
- Missing system dependencies: Install the apt packages listed above
- DiskANN build failure: Intel MKL not available, this is expected in some environments
- Python version mismatch: Ensure wheel matches your Python version

**Runtime Issues:**
- Network errors: HuggingFace model downloads require internet access
- Import errors: Ensure packages are installed in correct order (core → backends → meta)

**Performance:**
- Index building: 2-5 seconds per distance function for small datasets
- Search queries: 50-200ms typical response time
- Memory usage: ~500MB including model loading

Remember: **NEVER CANCEL builds or tests**. The C++ compilation and test execution can take significant time but will complete successfully.