# Contributing to High-Speed Fuse Break Analysis

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Development Environment

1. Clone the repository:
```bash
git clone https://github.com/Oussama-Guelfaa/High-Speed-Fuse-Break-Analysis.git
cd High-Speed-Fuse-Break-Analysis
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Code Style

This project follows the [Black](https://black.readthedocs.io/en/stable/) code style. We also use:
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/en/latest/) for linting
- [mypy](https://mypy.readthedocs.io/en/stable/) for type checking

You can run all style checks with:
```bash
# Format code
black notebooks/
isort notebooks/

# Check code
flake8 notebooks/
mypy notebooks/
```

## Testing

We use [pytest](https://docs.pytest.org/en/stable/) for testing. Run the tests with:
```bash
pytest
```

To run tests with coverage:
```bash
pytest --cov=notebooks tests/
```

## Pull Request Process

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests and style checks
5. Submit a pull request

## Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) for documentation. To build the docs:
```bash
cd docs
make html
```

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
