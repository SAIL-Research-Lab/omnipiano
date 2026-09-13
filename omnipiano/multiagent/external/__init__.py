"""Vendored reference implementations. See README.md for the three rules.

Deliberately empty of imports: some subpackages are pure torch and safe to
import at runtime, others exist only for tests and would pull in a conflicting
gym. Import the specific module you want, never this package's contents.
"""