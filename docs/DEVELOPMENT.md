# Development

Use a fresh kernel per notebook and install only the required environment. Keep generated artifacts, local datasets, and credentials out of commits. Active notebook filenames describe their actual frameworks; TensorFlow VGG16 feature extraction is no longer filed under PyTorch.

The syntax check accepts IPython magic lines but does not execute training, downloads, or GPU code. Successful structural checks are not model benchmark validation. Adult Income has its own package, environment, and correctness checks under `projects/adult_income/`.

From the collection root, run `cd projects/adult_income`, then `python -m unittest discover -s tests -v`.

To add a project, provide a README with data requirements, install and run commands, artifact locations, and measured results. Add a source entry when migrating existing work. Preserve research repositories and their individual experiment records as separate projects.
