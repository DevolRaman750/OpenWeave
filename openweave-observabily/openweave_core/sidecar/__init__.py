"""
openweave_core.sidecar — FastAPI compute service for the OpenWeave UI.

Exposes ``POST /analyze`` which runs the full pipeline for one trace
(anomaly detection -> incident classification -> deep evaluation), builds the
abstracted graph the UI renders, and returns a single ``AnalysisPayload`` JSON
mirroring the ``AnalysisSeed`` shape consumed by the tRPC ``analyze.run``
mutation / Prisma writer. The sidecar is *compute-only* — it never touches
Postgres; persistence happens on the Node side.
"""

from openweave_core.sidecar.assemble import assemble_analysis

__all__ = ["assemble_analysis"]
