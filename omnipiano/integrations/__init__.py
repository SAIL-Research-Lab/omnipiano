"""Framework-specific glue code (SB3 / OmniSafe / CleanRL bridges).

The packages under ``OmniPiano/integrations/`` hold adapters that
translate OmniPiano's framework-agnostic contract (info-dict schema,
``BenchmarkProtocolConfig``, the gymnasium env factory) into
framework-specific conveniences such as training callbacks, evaluator
subclasses, or logger adapters.

OmniPiano's core contract (``OmniPiano/envs/``, ``OmniPiano/configs/``,
``OmniPiano/wrappers/``) has no dependency on anything under
``integrations/``. Callers opt in by importing the specific integration
package for the framework they use.
"""
