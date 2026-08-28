"""Framework-specific glue code (SB3 / OmniSafe / CleanRL bridges).

The packages under ``omnipiano/integrations/`` hold adapters that
translate OmniPiano's framework-agnostic contract (info-dict schema,
``BenchmarkProtocolConfig``, the gymnasium env factory) into
framework-specific conveniences such as training callbacks, evaluator
subclasses, or logger adapters.

OmniPiano's core contract (``omnipiano/envs/``, ``omnipiano/configs/``,
``omnipiano/wrappers/``) has no dependency on anything under
``integrations/``. Callers opt in by importing the specific integration
package for the framework they use.
"""
"""Optional integrations for third-party training frameworks."""
