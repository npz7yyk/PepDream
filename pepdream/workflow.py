# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

class Workflow:
    """Class for data analysis workflow."""
    def __init__(self, params):
        self.params = params

    def process_msms(self):
        # Import modules
        from .process_msms import process_msms

        # Process msms file
        andromeda, psm = process_msms(
            msms_file=self.params.msms_file,
            csv_dir=self.params.csv_dir,
            save_dir=self.params.cache_dir
        )
        self.andromeda = andromeda
        self.psm = psm

    def predict_spectrum(self):
        # Import modules
        from .predict_spectrum import set_device, predict_spectrum
        from .prosit import config_state_dict as config_prosit_weight
        from .pdeep2 import config_state_dict as config_pdeep2_weight

        # Check psm is attribute
        if not hasattr(self, "psm"):
            from .process_msms import load_processed_raw
            self.psm = load_processed_raw(self.params.cache_dir)

        # Generate prediction spectrum
        set_device(self.params.device)
        config_prosit_weight(self.params.spectrum_model_weight)
        config_pdeep2_weight(self.params.spectrum_model_weight)
        intensities = predict_spectrum(
            self.psm, self.params.cache_dir,
            self.params.spectrum_model_type
        )
        self.observe_intensities = self.psm["observe_intensities"]
        self.predict_intensities = intensities

    def process_pin(self):
        # Import functions
        from .process_pin import load_feature_matrix
        from .pepdream import set_num_features

        # Load pin file
        pin_inform = load_feature_matrix(self.params.pin_file)
        pep_inform, feature_names, features, labels, scan_numbers = pin_inform
        self.pep_inform = pep_inform
        self.feature_names = feature_names
        self.features = features
        self.labels = labels
        self.scan_numbers = scan_numbers
        self.size, self.dim = features.shape

        # Set number of features
        set_num_features(self.dim)

        # Early return if prediction spectrum is given
        if self.params.predict_spectrum:
            return

        # Load parsed msms file
        from .process_msms import load_processed_raw
        post = self.params.spectrum_model_type
        self.psm = load_processed_raw(self.params.cache_dir)
        self.observe_intensities = self.psm["observe_intensities"]
        self.predict_intensities = self.psm["predict_intensities_" + post]

    def run_pepdream(self):
        """Run PepDream using loaded data"""
        # Import modules
        from .process_pin import search_init_dir, normalize_spectrum
        from .train_utils import split_data, run_algorithm, save_footprint

        params = self.params

        # Score initialization
        deepq = params.deepq
        init_dir, init_scores = search_init_dir(self.features, self.labels, deepq)
        init_feature = self.feature_names[init_dir]
        print(f"Using feature No.{init_dir}: {init_feature} as initial direction.")

        pre = normalize_spectrum
        features = [
            self.features, self.psm["sequences"],
            pre(self.observe_intensities), pre(self.predict_intensities)
        ]

        split_data(self.scan_numbers)
        scores, weights = run_algorithm(
            features=features,
            labels=self.labels,
            scores=init_scores,
            psm_info=self.pep_inform,
        )

        self.scores, self.weights = scores, weights

        # save training result
        save_footprint(
            scores=scores,
            weights=weights
        )

    def finetune_spectrum_model(self):
        """Finetune spectrum prediction model"""
        params = self.params

        # Import modules
        from .process_pin import normalize_spectrum
        from .predict_spectrum import prepare_data
        from .train_utils import load_footprint, run_finetune
        from .prosit import config_state_dict as config_prosit_weight
        from .pdeep2 import config_state_dict as config_pdeep2_weight

        # Config pretrained model weight
        config_prosit_weight(params.spectrum_model_weight)
        config_pdeep2_weight(params.spectrum_model_weight)

        # Check scores is an attribute
        # Happens when fine-tuning spectrum model only
        if not hasattr(self, "scores"):
            self.scores, self.weights = load_footprint()

        pre = normalize_spectrum
        # Generate prediction data
        input1 = prepare_data(self.psm, tensorize=False)
        features = [
            input1, self.features, self.psm["sequences"],
            pre(self.observe_intensities)
        ]
        run_finetune(
            feature_model_weights=self.weights,
            features=features, labels=self.labels,
            scores=self.scores, psm_info=self.pep_inform
        )

    def run(self):
        """Run the whole pipline"""

        # Process msms file
        if self.params.parse_maxquant_output:
            self.process_msms()
        if self.params.ending_step == 1:
            return

        # Predict spectrum
        if self.params.predict_spectrum:
            self.predict_spectrum()
        if self.params.ending_step == 2:
            return

        # Process pin file
        self.process_pin()

        from .train_utils import set_hyperparams as set_train_hyperparams
        from .pepdream import set_hyperparams as set_model_hyperparams
        set_train_hyperparams(self.params)
        set_model_hyperparams(self.params)

        if self.params.seed is not None:
            from .train_utils import set_seed
            set_seed(self.params.seed)

        # Run single PepDream
        if self.params.rescore_psm:
            self.run_pepdream()
        if self.params.ending_step == 3:
            return

        # Finetune spectrum prediction model
        if self.params.finetune_spectrum_model:
            self.finetune_spectrum_model()
        if self.params.ending_step == 4:
            return
