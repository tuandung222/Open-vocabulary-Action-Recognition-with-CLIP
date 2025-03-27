# Development Guide

This guide provides instructions for developers contributing to the CLIP HAR project, including code style guidelines, formatting tools, and development best practices.

## Code Style and Formatting

The CLIP HAR project follows the [PEP 8](https://peps.python.org/pep-0008/) style guide with some modifications enforced by [Black](https://black.readthedocs.io/).

### Automatic Formatting with Black

We use Black as our code formatter to maintain consistent code style across the project. Black is an opinionated formatter that reformats entire files to conform to its style.

To format a file or directory with Black:

```bash
# Format a single file
black path/to/file.py

# Format all Python files in a directory
black directory_name/

# Format the entire project
black .
```

### Setup and Configuration

The project includes a `pyproject.toml` file with Black configuration:

```toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | wandb
  | mlruns
  | venv
  | ENV
)/
'''
```

### Import Sorting with isort

We use [isort](https://pycqa.github.io/isort/) to sort imports consistently. isort is configured to be compatible with Black's formatting:

```bash
# Sort imports in a file
isort path/to/file.py

# Sort imports in all Python files in a directory
isort directory_name/

# Sort imports in the entire project
isort .
```

### Linting with flake8

We use [flake8](https://flake8.pycqa.org/) for linting Python code:

```bash
# Lint a file
flake8 path/to/file.py

# Lint all Python files in a directory
flake8 directory_name/

# Lint the entire project
flake8 .
```

## Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) hooks to ensure code quality before committing changes. The hooks automatically format code, sort imports, and catch common errors.

### Setting Up Pre-commit

1. Install pre-commit:

```bash
pip install pre-commit
```

2. Install the git hooks:

```bash
pre-commit install
```

Once installed, the hooks will run automatically on `git commit`. You can also run them manually:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run all hooks on staged files
pre-commit run
```

## Development Environment Setup

### Docker Development Environment

For consistent development environments, we recommend using the Docker containers:

```bash
# Build the training container
docker-compose build clip-har-train

# Run a shell in the training container for development
docker-compose run --rm clip-har-train bash
```

### Local Development Environment

If you prefer a local development environment:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install black isort flake8 mypy pre-commit pytest
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Pull Request Guidelines

When submitting a pull request:

1. Ensure all code is formatted with Black
2. Make sure all tests pass
3. Add tests for new features
4. Update documentation as needed
5. Follow the pull request template

## Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=CLIP_HAR_PROJECT

# Run tests for a specific module
pytest tests/test_module.py
```

## Documentation

When adding or modifying features, update the relevant documentation:

- Docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Update README.md for user-facing changes
- Update appropriate docs/ files for detailed documentation

## Branching Strategy

- `main`: Main development branch
- `feature/<feature-name>`: For new features
- `bugfix/<bug-name>`: For bug fixes
- `release/<version>`: For release candidates

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
feat: add new feature
fix: fix bug in X
docs: update documentation
style: format code
refactor: refactor code without changing functionality
test: add or modify tests
chore: update build scripts, etc.
```
