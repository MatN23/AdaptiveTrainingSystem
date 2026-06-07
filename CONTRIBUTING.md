# Contributing to LuminaAI

Thank you for your interest in contributing to LuminaAI! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)
- [Community](#community)

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git
- Understanding of transformer architectures (recommended)
- Experience with deep learning training (helpful)

### First Steps

1. **Read the Documentation**
   - Review the [README](README.md) thoroughly
   - Understand the adaptive training system
   - Familiarize yourself with MoE/MoD architectures

2. **Set Up Your Environment**
   - Fork the repository
   - Clone your fork locally
   - Install dependencies (see [Development Setup](#development-setup))

3. **Explore the Codebase**
   - Run the `debug` configuration to understand the system
   - Review existing code in `Src/` directory
   - Read inline documentation and comments

---

## Development Setup

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/luminaai.git
cd luminaai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks (if configured)
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_trainer.py

# Run with coverage
pytest --cov=Src tests/

# Run debug configuration (quick validation)
cd Src/Main_Scripts
python Main.py  # Should complete in ~5 minutes
```

### Code Quality Tools

```bash
# Format code
black Src/

# Lint code
flake8 Src/
pylint Src/

# Type checking
mypy Src/

# Sort imports
isort Src/
```

---

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**: Fix training instabilities, crashes, or incorrect behavior
2. **Features**: Add new architectures, optimizers, or training techniques
3. **Performance**: Optimize throughput, memory usage, or convergence
4. **Documentation**: Improve guides, docstrings, or examples
5. **Tests**: Add test coverage for untested components
6. **Research**: Implement papers or experimental techniques

### Finding Issues to Work On

- Check the [Issues](https://github.com/matn23/luminaai/issues) page
- Look for labels:
  - `good first issue`: Great for newcomers
  - `help wanted`: Maintainers need assistance
  - `bug`: Known issues needing fixes
  - `enhancement`: New features or improvements
  - `documentation`: Docs improvements
  - `performance`: Optimization opportunities

### Before Starting Work

1. **Check for existing work**: Search issues and PRs to avoid duplication
2. **Discuss major changes**: Open an issue first for significant features
3. **Assign yourself**: Comment on the issue to claim it
4. **Ask questions**: Reach out if you need clarification

---

## Code Standards

### Code Style

- **Python Style**: Follow [PEP 8](https://pep8.org/)
- **Line Length**: Maximum 88 characters (Black default)
- **Imports**: Use `isort` for consistent import ordering
- **Naming Conventions**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

### Documentation Standards

#### Docstrings

Use Google-style docstrings:

```python
def adjust_capacity_factor(self, new_factor: float) -> None:
    """Dynamically adjust MoE capacity factor during training.
    
    This method modifies how many tokens can be routed to each expert,
    affecting load balancing and training stability.
    
    Args:
        new_factor: New capacity factor (typically 1.0-2.0).
            Higher values allow more tokens per expert but increase
            memory usage. Lower values improve efficiency but may
            cause token dropping.
    
    Raises:
        ValueError: If new_factor <= 0.
        
    Example:
        >>> trainer.adjust_capacity_factor(1.5)
        >>> # Allows 50% more tokens per expert
        
    Note:
        Changes take effect on the next forward pass. This is safe
        to call during training.
    """
    if new_factor <= 0:
        raise ValueError(f"Capacity factor must be positive, got {new_factor}")
    
    self.config.capacity_factor = new_factor
    self.logger.info(f"Adjusted capacity factor to {new_factor}")
```

#### Comments

- **Why over what**: Explain reasoning, not obvious operations
- **Complex logic**: Add comments for non-trivial algorithms
- **TODOs**: Include TODOs with issue numbers when applicable

```python
# Good: Explains why
# Use emergency LR reduction to prevent gradient explosion
# rather than simple clipping, which masks the underlying issue
self.emergency_lr_reduction(factor=10.0)

# Bad: States the obvious
# Reduce learning rate
self.learning_rate *= 0.1
```

### Architecture Principles

Follow these design principles from the codebase:

1. **Autonomy**: Systems should self-optimize when possible
2. **Observability**: Expose metrics for monitoring and debugging
3. **Recovery**: Handle failures gracefully with automatic recovery
4. **Flexibility**: Support multiple architectures and configurations
5. **Production-Ready**: Include error handling, logging, and checkpointing

---

## Testing Guidelines

### Test Structure

```
tests/
 unit/                 # Unit tests for individual components
    test_trainer.py
    test_model.py
    test_config.py
 integration/          # Integration tests for workflows
    test_training_loop.py
    test_checkpointing.py
 fixtures/             # Shared test data and fixtures
     sample_configs.py
```

### Writing Tests

```python
import pytest
import torch
from src.trainer import Trainer

class TestTrainer:
    @pytest.fixture
    def trainer(self):
        """Create a minimal trainer for testing."""
        config = Config(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            vocab_size=1000
        )
        return Trainer(config)
    
    def test_adjust_capacity_factor(self, trainer):
        """Test capacity factor adjustment."""
        original = trainer.config.capacity_factor
        trainer.adjust_capacity_factor(2.0)
        assert trainer.config.capacity_factor == 2.0
        assert trainer.config.capacity_factor != original
    
    def test_adjust_capacity_factor_invalid(self, trainer):
        """Test capacity factor validation."""
        with pytest.raises(ValueError):
            trainer.adjust_capacity_factor(-1.0)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_training_on_gpu(self, trainer):
        """Test training stability on GPU."""
        # Implementation
        pass
```

### Test Coverage

- **Minimum coverage**: Aim for 80% coverage on new code
- **Critical paths**: 100% coverage for training loop, checkpointing, recovery
- **Edge cases**: Test boundary conditions and error paths
- **Hardware**: Mark hardware-specific tests with appropriate decorators

### Performance Testing

For optimization PRs, include benchmarks:

```python
def test_flash_attention_speedup():
    """Verify Flash Attention improves throughput."""
    baseline_throughput = benchmark_without_flash_attention()
    optimized_throughput = benchmark_with_flash_attention()
    
    assert optimized_throughput > baseline_throughput * 1.5  # At least 1.5x faster
```

---

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Create a feature branch
git checkout -b feature/add-moe-pruning

# Make your changes
# ...

# Commit with clear messages
git add .
git commit -m "feat: Add dynamic MoE expert pruning

- Implement automatic expert removal based on utilization
- Add pruning threshold configuration option
- Include tests for pruning logic
- Update documentation with pruning examples

Fixes #123"
```

### 2. Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(moe): Add dynamic expert pruning

Implements automatic removal of underutilized experts during training
to improve efficiency and reduce memory usage.

Fixes #123
```

```
perf(attention): Optimize Flash Attention memory usage

Reduce peak memory consumption by 25% through better kernel fusion.

Benchmark results:
- A100 40GB: 1200 tok/s  1500 tok/s
- Memory: 32GB  24GB
```

### 3. Before Submitting

- [ ] Run all tests: `pytest tests/`
- [ ] Check code style: `black Src/ && flake8 Src/`
- [ ] Update documentation if needed
- [ ] Add tests for new functionality
- [ ] Ensure CI passes locally
- [ ] Rebase on latest `main` branch

```bash
# Update your branch
git fetch upstream
git rebase upstream/main

# Run final checks
pytest tests/
black Src/
flake8 Src/
```

### 4. Submit Pull Request

1. Push to your fork:
   ```bash
   git push origin feature/add-moe-pruning
   ```

2. Open PR on GitHub using the [Pull Request template](.github/PULL_REQUEST_TEMPLATE.md)

3. Fill out all sections thoroughly

4. Link related issues (e.g., "Fixes #123")

### 5. Review Process

- **Automated checks**: CI must pass (tests, linting, etc.)
- **Code review**: At least one maintainer approval required
- **Feedback**: Address reviewer comments promptly
- **Updates**: Push additional commits to same branch
- **Merge**: Maintainer will merge once approved

### Review Timeline

- Initial review: Within 3-5 days
- Follow-up reviews: Within 2-3 days
- Complex PRs may take longer

---

## Areas for Contribution

### High Priority

1. **Training Stability**
   - Improve gradient explosion detection
   - Add more sophisticated recovery mechanisms
   - Implement better loss spike handling

2. **Performance Optimization**
   - Flash Attention improvements
   - Better memory management
   - Faster data loading

3. **Architecture Support**
   - Additional sparse architectures
   - Quantization-aware training
   - Distillation support

4. **Testing**
   - Multi-GPU test coverage
   - Edge case testing
   - Integration test suite

### Medium Priority

1. **Documentation**
   - Tutorial notebooks
   - Architecture deep-dives
   - Troubleshooting guides

2. **Monitoring**
   - Improved metrics dashboard
   - Better visualization
   - Real-time alerts

3. **Data Pipeline**
   - More efficient data loading
   - Better preprocessing
   - Dataset utilities

### Research Contributions

Implementing recent papers:

- New attention mechanisms
- Advanced MoE routing strategies
- Novel optimization techniques
- Improved scaling laws

**Requirements for research PRs:**
- Cite original paper
- Provide ablation studies
- Include reproducible configs
- Document limitations

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews

### Getting Help

- Check [documentation](docs/)
- Search [existing issues](https://github.com/matn23/luminaai/issues)
- Ask in [Discussions](https://github.com/matn23/luminaai/discussions)
- Tag maintainers for urgent issues

### Recognition

Contributors are recognized in:
- [README.md](README.md) contributors section
- Release notes for significant contributions
- Special thanks for major features

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

---

## Questions?

If you have questions about contributing, please:
1. Check this guide thoroughly
2. Search existing issues and discussions
3. Open a new issue with the `question` label

**Thank you for contributing to LuminaAI!** 

We're building the future of autonomous training systems together.