# Contributing to Realtime Face Attendance

Thank you for considering contributing to this project! Here's how you can help.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yxshee/realtime-face-attendance/issues)
2. If not, create a new issue with:
   - Clear title describing the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Screenshots if applicable

### Suggesting Features

1. Open an issue with the "enhancement" label
2. Describe the feature and its use case
3. Explain why it would benefit users

### Code Contributions

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/realtime-face-attendance.git
   cd realtime-face-attendance
   ```

2. **Set Up Development Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

5. **Test Your Changes**
   ```bash
   python -m pytest
   python codes/ultimate_system.py  # Manual testing
   ```

6. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Reference any related issues
   - Describe your changes clearly

## Code Style Guidelines

- Use meaningful variable and function names
- Follow PEP 8 style guide
- Add docstrings to functions
- Keep functions focused and small
- Use type hints where appropriate

## Commit Message Format

```
<type>: <short description>

<optional longer description>
```

Types:
- `Add` - New feature
- `Fix` - Bug fix
- `Update` - Changes to existing feature
- `Refactor` - Code improvement without changing behavior
- `Docs` - Documentation changes
- `Test` - Adding tests

## Questions?

Open an issue or contact the maintainer at yash999901@gmail.com.
