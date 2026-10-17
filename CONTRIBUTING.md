# Contributing to RAG Llama 3 Multilingual

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback

## Getting Started

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/rag-llama3-multilingual.git
   cd rag-llama3-multilingual
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use Black for code formatting: `black src/`
- Use type hints where appropriate
- Write descriptive docstrings

### Testing

- Write tests for new features
- Run tests before submitting: `pytest`
- Aim for good test coverage
- Test both success and error cases

### Documentation

- Update README if adding features
- Add docstrings to new functions/classes
- Include usage examples where helpful

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, etc.)
- Reference issues when applicable

Example:
```
Add support for DOCX file ingestion

- Implement DOCX loader in document_processor.py
- Add tests for DOCX processing
- Update README with supported formats

Fixes #123
```

## Pull Request Process

1. **Before Submitting**
   - Run tests: `pytest`
   - Format code: `black src/ tests/`
   - Check for linting issues: `flake8 src/`
   - Update documentation if needed

2. **Submit PR**
   - Provide clear description of changes
   - Link related issues
   - Include screenshots for UI changes
   - Ensure CI passes

3. **Review Process**
   - Address reviewer feedback
   - Keep PR focused and small when possible
   - Be patient and respectful

## Areas for Contribution

### High Priority
- Additional language support (French, German, Italian)
- More document format support (DOCX, HTML, CSV)
- Performance optimizations
- Enhanced error handling

### Documentation
- Tutorial videos
- More usage examples
- Translation of documentation

### Testing
- Integration tests
- Performance benchmarks
- Edge case coverage

### Features
- Conversation history/memory
- Advanced filtering capabilities
- Cloud storage integration
- Audio/video transcription

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
