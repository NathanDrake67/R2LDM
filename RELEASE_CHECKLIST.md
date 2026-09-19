# Release and reproducibility checklist

The initial public source-code release is prepared. The following items should be resolved for a fully reproducible archival release.

## Required decisions

- [x] Select a project-level open-source license and add it as `LICENSE` (Apache-2.0).
- [ ] Confirm that all co-authors and the institution approve the code, model weights, sample data, and benchmark outputs selected for release.
- [ ] Audit `det3d/`, `s2dense/`, `tools_voxel/`, `models/`, and `utils/` against their exact upstream revisions; restore all required third-party `LICENSE`, `NOTICE`, and copyright files.
- [ ] Decide whether pretrained checkpoints may be redistributed and add stable download links plus checksums. Do not commit the 500+ MB checkpoints directly to normal Git history.

## Reproducibility gaps to resolve

- [ ] Release the exact ground-removal, field-of-view alignment, multi-frame aggregation, and binary-export preprocessing pipeline used for the paper.
- [ ] Confirm the final VoD train/test split. The supplied `train_files_all.txt` and `test_files_all.txt` contain 7,169 and 1,513 entries, while the paper reports 7,539 training frames and 1,143 testing frames.
- [ ] Confirm the voxelization settings. The supplied code uses `[0, 32] x [-16, 16] x [-2, 4]` with voxel size `[0.1, 0.1, 0.15]`; these values differ from the VoD settings described in the paper.
- [ ] Confirm whether the released inference configuration should use 128 sampling steps. The original script hard-coded 32, while the paper identifies 128 as the balanced default. This release candidate exposes the value and defaults to 128.
- [ ] Decide which LPCR offset mode to recommend publicly. `legacy_sigmoid` reproduces the paper implementation/checkpoints; `centered_tanh` matches the signed `point - voxel_center` target but requires retraining both stages. Report the selected mode with every released checkpoint.
- [ ] Add the GPAL data adapter and experiment configuration if both paper datasets are intended to be reproducible from this repository.

## Validation

- [ ] Create the Conda environment on a clean Linux/CUDA machine.
- [ ] Run a two-sample Stage-1 smoke test and verify forward, backward, validation, and checkpoint saving.
- [ ] Run a two-sample Stage-2 smoke test from the Stage-1 checkpoint.
- [ ] Run `test.py --max-samples 2` and confirm output shape, coordinate frame, numeric range, and timing.
- [ ] Run `eval_metric_2025.py` on a small paired set and verify the column counts, voxelization range, squared-distance convention, and F-score threshold.
- [ ] Reproduce at least one metric/table row from the paper and record the exact command, checkpoint checksum, dataset version, and random seed.
- [ ] Add automated checks for Python syntax, secret scanning, large files, and accidental dataset/checkpoint commits.

## Repository metadata

- [x] Add the final GitHub repository URL to `CITATION.cff`.
- [ ] Add release contact information and an issue-reporting policy.
- [ ] Add acknowledgements and citations for every retained upstream component.
- [ ] Create a tagged release only after the repository passes the clean-machine test.
