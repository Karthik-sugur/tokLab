"""Dataset ingestion, deduplication, and repository-level splits."""

from toklab.data.splits import ManifestView, build_dataset

__all__ = ["ManifestView", "build_dataset"]
