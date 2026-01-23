# Security Policy

## Supported Versions

We actively support the latest version of the project with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Security Updates

### Latest Security Patches (2024)

All dependencies have been updated to address known vulnerabilities:

#### Critical Updates
- **FastAPI** 0.109.0 → 0.115.0
  - CVE: ReDoS in Content-Type Header parsing
  - Severity: Medium
  - Status: ✅ Patched

- **PyTorch** 2.1.2 → 2.6.0
  - Multiple CVEs: Heap buffer overflow, use-after-free, RCE in torch.load
  - Severity: High
  - Status: ✅ Patched

- **Transformers** 4.36.2 → 4.48.0
  - CVE: Deserialization of untrusted data
  - Severity: High
  - Status: ✅ Patched

#### Important Updates
- **LangChain Community** 0.0.10 → 0.3.27
  - CVEs: XXE attacks, SSRF, pickle deserialization
  - Severity: High
  - Status: ✅ Patched

- **LangChain Text Splitters** 0.0.1 → 0.3.9
  - CVE: XXE vulnerability in XSLT parsing
  - Severity: Medium
  - Status: ✅ Patched

- **python-multipart** 0.0.6 → 0.0.18
  - CVEs: DoS via malformed boundary, ReDoS in Content-Type
  - Severity: Medium
  - Status: ✅ Patched

## Security Best Practices

### For Users

1. **Environment Variables**
   - Never commit `.env` files to version control
   - Use strong, unique values for tokens and credentials
   - Rotate credentials regularly

2. **File Uploads**
   - Only upload trusted documents
   - Be cautious with PDFs from untrusted sources
   - Scan uploaded files with antivirus before processing

3. **API Security**
   - Use HTTPS in production
   - Implement rate limiting
   - Add authentication/authorization for production deployments
   - Restrict CORS origins in production

4. **Docker Deployment**
   - Use specific version tags, not `latest`
   - Run containers as non-root users
   - Regularly update base images
   - Scan images for vulnerabilities

### For Developers

1. **Dependency Management**
   - Regularly update dependencies
   - Review security advisories
   - Use `pip-audit` or similar tools
   - Pin dependency versions

2. **Code Security**
   - Validate all user inputs
   - Use parameterized queries/safe APIs
   - Implement proper error handling
   - Don't log sensitive information

3. **LLM-Specific Security**
   - Validate and sanitize prompts
   - Be aware of prompt injection risks
   - Set appropriate token limits
   - Monitor for unusual usage patterns

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

1. **Email**: Send details to the project maintainers (check repository for contact info)
2. **GitHub Security Advisory**: Use the "Security" tab → "Report a vulnerability"

### What to Include

- Type of vulnerability
- Affected version(s)
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

- Acknowledgment within 48 hours
- Regular updates on progress
- Credit in security advisory (if desired)
- Coordinated disclosure timeline

## Security Scanning

This project uses:
- CodeQL for static analysis
- Dependency vulnerability scanning
- Regular security audits

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Docker Security](https://docs.docker.com/engine/security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

## Changelog

### 2024-01-23
- Updated all vulnerable dependencies
- Added security documentation
- Implemented input validation
- Enhanced error handling

---

Last Updated: 2024-01-23
