# Secrets Management

This file is intentionally kept out of version control. Use it to store notes about
sensitive credentials required for local development (for example, the PyPI token
used for publishing). Do **not** store the token itself in this repository.

Recommended setup:
- Create an environment variable such as `TWINE_PASSWORD` exporting your PyPI token.
- Optionally store encrypted credentials using your system keychain or a secrets
  manager (1Password, Bitwarden, macOS Keychain, etc.).
- Document the location of the stored secret here for your personal reference.

If you need to share access with collaborators, rotate the token and communicate it
out-of-band. Avoid pasting secrets into git history, logs, or pull requests.
