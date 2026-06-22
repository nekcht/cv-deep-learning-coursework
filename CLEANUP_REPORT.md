# Cleanup Report

## Included Notebooks

- `knn.ipynb` -> `notebooks/01_knn_cifar10.ipynb`
- `svm.ipynb` -> `notebooks/02_svm_cifar10.ipynb`
- `softmax.ipynb` -> `notebooks/03_softmax_cifar10.ipynb`
- `numpy_two_layer_net.ipynb` -> `notebooks/04_numpy_two_layer_net.ipynb`
- `pytorch_two_layer_net.ipynb` -> `notebooks/05_pytorch_two_layer_net.ipynb`
- `pytorch_convnets.ipynb` -> `notebooks/06_pytorch_convnets.ipynb`
- `pytorch_transformers.ipynb` -> `notebooks/07_vision_transformer.ipynb`

## Excluded Notebooks

- `intro_to_python.ipynb`: not present in the workspace during cleanup; also outside the final focused computer-vision notebook set.
- `numerical_python_1.ipynb`: not present in the workspace during cleanup; also outside the final focused computer-vision notebook set.
- `numerical_python_2.ipynb`: not present in the workspace during cleanup; also outside the final focused computer-vision notebook set.

## Cleanup Actions Performed

- Created the requested portfolio structure with `README.md`, `requirements.txt`, `.gitignore`, `LICENSE_OR_NOTICE.md`, `CLEANUP_REPORT.md`, `notebooks/`, and `assets/images/`.
- Copied the selected notebooks into `notebooks/` with numbered, descriptive filenames.
- Left the original raw notebooks at the project root untouched and added those filenames to `.gitignore` to reduce the risk of accidentally uploading uncleaned coursework material.
- Added a clean title, "What this notebook demonstrates", and academic-context section to each included notebook.
- Removed Google Colab Drive setup cells, hard-coded Colab paths, lab folder setup instructions, submission warnings, grading/bonus wording, and stale template markers.
- Normalized dataset paths to local ignored folders such as `data/`.
- Removed large embedded outputs, training logs, stale error outputs, and all remaining cell outputs for cleaner GitHub rendering.
- Removed an in-notebook `!pip install timm` command and listed `timm` in `requirements.txt` instead.
- Removed external remote image embeds from the Vision Transformer notebook.

## Not Fully Executable

- The notebooks were cleaned and JSON-validated, but they were not rerun end to end.
- `01_knn_cifar10.ipynb`, `02_svm_cifar10.ipynb`, `03_softmax_cifar10.ipynb`, and `04_numpy_two_layer_net.ipynb` import a local `engine` helper package that is not present in this workspace. They need that package restored before full execution.
- The PyTorch notebooks download CIFAR-10 through `torchvision` when run. Some pretrained-model sections may also require internet access or an existing model cache.

## Remaining Caveats

- Review any remaining markdown explanations of experimental outcomes before public upload, especially where the original notebooks discuss relative model performance without preserved outputs.
- If you want fully reproducible notebooks, add the missing local helper package or replace the `engine` imports with self-contained implementations.
- Confirm that publishing cleaned coursework notebooks is consistent with the relevant university and course policies.

## Manual Review Points

- Verify that no instructor-specific policy text or private course material remains.
- Re-run notebooks you want to showcase with fresh outputs if you want visible plots or metrics on GitHub.
- Check pretrained-model sections for current `torchvision` and `timm` API compatibility before upload.
