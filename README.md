# PepDream
PepDream: automatic feature extraction for peptide identification from tandem mass spectra

# License
PepDream is primarily licensed under the MIT License. However, specific portions are subject to different licenses as follows:

=== Open Software License 3.0 (OSL-3) ===

The following functions or portions of code are derived from external sources and are subject to the Open Software License 3.0 (OSL-3):

- pepdream/qvalue_acc.pyx - calculate_qvalue
  - Source: proteoTorch (https://github.com/proteoTorch/proteoTorch)
  - Modifications: Optimized the code for improved performance.

- pepdream/qvalue_utils.py - accumulate & calculate_qvalue
  - Source: proteoTorch (https://github.com/proteoTorch/proteoTorch)
  - Modifications: Removed unused code.

=== Apache License 2.0 ===

The following functions or portions of code are derived from external sources and are subject to the Apache License 2.0:

- pepdream/constants.py
  - Source: Prosit (https://github.com/kusterlab/prosit)
  - Modifications: Selected a subset of the constants.

- pepdream/process_msms.py - functions marked with `copied from Prosit (and modified)`
  - Source: Prosit (https://github.com/kusterlab/prosit)
  - Modifications: Check out process_msms.py for details.
