# Third-party software notice

R2LDM builds on open-source research software. The copied code snapshot did not preserve complete provenance metadata for every file, so the items below must be reviewed before redistribution.

## Sparse2Dense and CenterPoint / Det3D

The `s2dense/`, `det3d/`, and related voxel utilities contain adapted code or structural elements from:

- [Sparse2Dense](https://github.com/stevewongv/Sparse2Dense), released under the MIT License.
- [CenterPoint](https://github.com/tianweiy/CenterPoint), released under the MIT License and itself incorporating components from Det3D, CenterNet, CenterTrack, MMCV, MMDetection, and other projects described in its upstream notice.

Before publication, compare the retained files with the exact upstream revisions, restore all applicable copyright headers, and include the upstream `LICENSE` and `NOTICE` files verbatim.

## spconv

R2LDM depends on [spconv](https://github.com/traveller59/spconv), licensed under Apache-2.0. The original working directory contained a full copied source/build tree. This release candidate removes that tree and installs `spconv-cu120==2.3.6` as an external dependency instead.

## NVIDIA Apex

The original working directory also contained a copied NVIDIA Apex tree, but R2LDM's released entry points do not import it. It has therefore been excluded from this release candidate.

## PointNet feature checkpoint / SpareNet

The optional `--pointnet-fpd` metric uses the PointNet implementation under
`metrics/extractor/pointnet.py` and downloads a pretrained classification
checkpoint hosted by [Microsoft SpareNet](https://github.com/microsoft/SpareNet).
Confirm the checkpoint's redistribution and usage terms, pin its checksum, and
record the exact feature-extractor version before publishing benchmark values.

## Other file-level attributions

Several retained files contain their own copyright or attribution comments, including code derived from OpenPCDet, Facebook/Meta utilities, Microsoft utilities, and academic research repositories. Those notices must remain intact. A project-level license does not replace or override third-party terms.

This document is an engineering inventory, not legal advice.
