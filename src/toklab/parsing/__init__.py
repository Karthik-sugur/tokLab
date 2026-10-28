"""Source segmentation and identifier roles."""

from toklab.parsing.roles import assign_ident_roles
from toklab.parsing.segmenters import NativeSegmenter, TreeSitterSegmenter

__all__ = ["NativeSegmenter", "TreeSitterSegmenter", "assign_ident_roles"]
