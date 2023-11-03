# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import argparse


DEFAULT_VALUES = {
    "json_file": "",
    "json_example": False,
    "gpu_id": -1,
    "working_dir": "",
    "spectrum_model_type": "prosit",
    "starting_step": "parse-maxquant-output",
    "ending_step": "finetune-spectrum-model",
    "spectrum_model_weight": "",
    "msms_file": "",
    "csv_dir": "",
    "pin_file": "",
    "seed": None,
    "k_fold": 3,
    "total_iteration": 8,
    "output_interval": 1,
    "main_q_threshold": 0.01,
    "deep_q_threshold": 0.07,
    "eval_q_threshold": 0.1,
    "batch_size": 5000,
    "learning_rate": 0.001,
    "dropout_rate": 0.4,
    "adam_weight_decay": 2e-05,
    "total_snapshot": 10,
    "snapshot_in_ensemble": 16,
    "epoch_per_snapshot": 6,
    "positive_smoothing": 0.99,
    "negative_smoothing": 0.99,
    "false_positive_loss_factor": 4.0,
    "gamma": 0.5
}


def parse_args():
    # Parse arguments
    description = """
    PepDream v0.1.0
    Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>

    Copyright (C) 2023 Yikang Yue
    Licensed under MIT License with third-party code under other licenses.
    See LICENSE and README for more details
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s 0.1.0",
        help="show version information and exit"
    )

    # general arguments
    group = parser.add_argument_group(
        title="General arguments",
        description="Arguments for general usage."
    )
    group.add_argument(
        "-j", "--json-file", type=str, default=None, action="store",
        help=(
            "Path to a json file containing all arguments. "
            "Arguments specified in command line will overwrite "
            "arguments specified in json file."
        )
    )
    group.add_argument(
        "-jx", "--json-example", action="store_true",
        help=(
            "If specified, an example json file containing all arguments "
            "will be generated and exit. This is useful when user "
            "want to use a json file to specify arguments."
        )
    )
    group.add_argument(
        "-g", "--gpu-id", type=int, default=None, action="store",
        help=(
            "Specify the GPU to use. Use -1 or omit for no GPU usage. "
            "Note: Using no GPU will significantly slow down the program. "
            "Default: -1 (no GPU usage)."
        )
    )
    group.add_argument(
        "-o", "--working-dir", type=str, default=None, action="store",
        help=(
            "Set the working directory for this script. "
            "Create this directory if it doesn't exist. "
            "Results and temporary files will be stored here. "
            "Default: experiment_YYYYMMDD_HHMM."
        )
    )
    group.add_argument(
        "-t", "--spectrum-model-type", type=str, default=None,
        choices=["prosit", "pdeep2"], action="store",
        help=(
            "Type of spectrum prediction model. "
            "Predicted spectrum will be used in PepDream PSM rescoring. "
            "It will also be the spectrum prediction model used in finetuning "
            "if --finetune-spectrum-model is specified. "
            "Currently model type Prosit and PDeep2 are supported. "
            "Please choose from 'prosit' and 'pdeep2'. "
            "Default: 'prosit'."
        )
    )
    # end of general arguments

    # critical workflow arguments
    description = """Critical arguments for controlling workflow.
    The following features are supported (in sequential steps):

    1. Parse raw MaxQuant outputs - msms.txt and *.csv files
            msms.txt contains PSMs information
            *.csv files contain raw ion m/z and intensity information
       Parsing results will be used in later steps
    2. Predict spectrum using a pre-trained spectrum prediction model
            Predicted spectrum will be used in PepDream PSM rescoring
       To run this feature, please run step 1 first
    3. Rescore PSMs with percolator input file - pin file
            Pin files typically have the following columns:
                id <tab> label <tab> scannr <tab> feature1 <tab> ... <tab>
                featureN <tab> peptide <tab> proteinId1 <tab> .. <tab> proteinIdM
            Labels are interpreted as: 1 - positive and test set, -1 - negative set
       To run this feature, please run step 1 and 2 first
    4. Finetune a pre-trained spectrum prediction model
            Finetuned model will be used in predicting spectrum and rescoring PSMs
            Typically this step leads to more PSMs identified with lower q-values
       This feature can be used after features 1, 2 and 3 are run

    User can specify which consecutive steps to run by specifying arguments:
    --starting-step and --ending-step. For example, if user want to run step 2 and 3,
    please specify --starting-step STEP2_NAME --ending-step STEP3_NAME.
    Here are the names of each step:
        STEP1_NAME: parse-maxquant-output, maxquant, m, or 1
        STEP2_NAME: predict-spectrum, spectrum, s, or 2
        STEP3_NAME: rescore-psm, rescore, r, or 3
        STEP4_NAME: finetune-spectrum-model, finetune, f, or 4"""
    group = parser.add_argument_group(
        title="Critical workflow arguments",
        description=description
    )
    group.add_argument(
        "-s", "--starting-step", type=str,
        choices=[
            "parse-maxquant-output", "maxquant", "m", "1",
            "predict-spectrum", "spectrum", "s", "2",
            "rescore-psm", "rescore", "r", "3",
            "finetune-spectrum-model", "finetune", "f", "4"
        ],
        default=None, action="store",
        help=(
            "Specify the starting step of the workflow. "
            "Please check out the description of critical "
            "workflow arguments for more details. "
            "Default: parse-maxquant-output."
        )
    )
    group.add_argument(
        "-e", "--ending-step", type=str,
        choices=[
            "parse-maxquant-output", "maxquant", "m", "1",
            "predict-spectrum", "spectrum", "s", "2",
            "rescore-psm", "rescore", "r", "3",
            "finetune-spectrum-model", "finetune", "f", "4"
        ],
        default=None, action="store",
        help=(
            "Specify the starting step of the workflow. "
            "Please check out the description of critical "
            "workflow arguments for more details. "
            "Default: parse-maxquant-output."
        )
    )
    # end of critical workflow arguments

    # input files arguments
    group = parser.add_argument_group(
        title="Input files arguments",
        description="Arguments for specifying input files."
    )
    group.add_argument(
        "-w", "--spectrum-model-weight", type=str, default=None, action="store",
        help=(
            "Path to the spectrum prediction model weight. "
            "Please check out https://github.com/npz7yyk/PepDream "
            "and download corresponding model weight."
        )
    )
    group.add_argument(
        "-m", "--msms-file", type=str, default=None, action="store",
        help=(
            "MaxQuant msms.txt file containing PSMs information. "
            "Required if starting from processing MaxQuant output."
        )
    )
    group.add_argument(
        "-c", "--csv-dir", type=str, default=None, action="store",
        help=(
            "Directory containing MaxQuant *.csv files. "
            "Required if starting from processing MaxQuant output."
        )
    )
    group.add_argument(
        "-p", "--pin-file", type=str, default=None, action="store",
        help=(
            "Path to a custom percolator input file (pin file). "
            "Use this option if you want to provide your own pin file. "
            "If not specified, pin file parsed from MaxQuant output will be used. "
        )
    )
    # end of input files arguments

    # PepDream analysis section
    group = parser.add_argument_group(
        title="PepDream arguments",
        description="Arguments for running PepDream analysis."
    )
    group.add_argument(
        "-sd", "--seed", type=int, default=None, action="store",
        help=(
            "Random seed used in training for reproducibility. "
            "If not specified, seed will not be fixed. "
            "Developer hint: seed 3407 really works well. "
            "(https://doi.org/10.48550/arXiv.2109.08203)"
        )
    )
    group.add_argument(
        "-k", "--k-fold", type=int, default=None, action="store",
        help=(
            "K-fold defines the number of folds employed in cross-validation, "
            "with a minimum requirement of 2. During each iteration, models are "
            "trained on k-1 folds and predict on the remaining one. "
            "This process repeats k times, and the final identification results "
            "are merged from predictions on remaining folds to prevent cheating. "
            "A larger k value generally yields a more accurate model and "
            "identifies more PSMs but extends training time proportionally."
            "Default: 3."
        )
    )
    group.add_argument(
        "-i", "--total-iteration", type=int, default=None, action="store",
        help=(
            "The total number of global training iterations. "
            "In each iteration, since each model is trained on its own folds, "
            "it gains the ability to rescore the PSMs effectively. This new "
            "scoring allows for the selection of additional training data, "
            "which in turn leads to more accurate models. By doing so "
            "iteratively, more PSMs with lower q-values can be identified. "
            "Default: 8."
        )
    )
    group.add_argument(
        "-oi", "--output-interval", type=int, default=None, action="store",
        help=(
            "Interval of iterations between two consecutive output. "
            "Typically, script performance increases with the number of "
            "iterations, but the rate of improvement decreases. "
            "Therefore, it's recommended to output the rescored PSMs "
            "at regular intervals to choose ideal identification results. "
            "Default: 1."
        )
    )
    group.add_argument(
        "-q", "--main-q-threshold", type=float, default=None, action="store",
        help=(
            "The script will provide counts of PSMs with q-values below this "
            "specified threshold upon execution, allowing users to monitoring "
            "identification counts meeting this q-value threshold. "
            "Note: this threshold doesn't impact training process. "
            "Default: 0.01."
        )
    )
    group.add_argument(
        "-dq", "--deep-q-threshold", type=float, default=None, action="store",
        help=(
            "During rescoring, PSMs with q-values below this threshold are "
            "included as model training data. It's crucial that this threshold "
            "exceeds the --main-q-threshold, retaining less statistically "
            "significant PSMs pre-training. This enriches training data "
            "diversity and enhances models' performance, resulting in "
            "more PSMs meeting the --main-q-threshold post-training."
            "Default: 0.07."
        )
    )
    group.add_argument(
        "-eq", "--eval-q-threshold", type=float, default=None, action="store",
        help=(
            "During model selection phase, AUC is used to evaluate individual "
            "models. PSMs with q-values below this threshold contribute to "
            "the AUC curve, guiding model selection. To include a wider "
            "range of potential PSMs as training data in subsequent training, "
            "this threshold is usually set higher than the --deep-q-threshold. "
            "This strategy promotes training diversity by including less "
            "significant PSMs and leads to a more comprehensive model."
            "Default: 0.10."
        )
    )
    group.add_argument(
        "-b", "--batch-size", type=int, default=None, action="store",
        help="Batch size used in training. Default: 5000."
    )
    group.add_argument(
        "-lr", "--learning-rate", type=float, default=None, action="store",
        help="Initial learning rate used in training. Default: 0.001."
    )
    group.add_argument(
        "-dr", "--dropout-rate", type=float, default=None, action="store",
        help=(
            "Dropout rate used in the fine-tuning process. "
            "Default: 0.4."
        )
    )
    group.add_argument(
        "-wd", "--adam-weight-decay", type=float, default=None, action="store",
        help="Adam weight decay factor used for regularization. Default: 2e-5."
    )
    group.add_argument(
        "-ss", "--total-snapshot", type=int, default=None, action="store",
        help=(
            "Number of model snapshots taken when trained on each (k-1) folds. "
            "During training, model snapshots are taken at regular intervals. "
            "In this way, snapshots of varying fitting levels are collected. "
            "Ensembling these snapshots will yield a more robust model. "
            "Default: 10."
        )
    )
    group.add_argument(
        "-se", "--snapshot-in-ensemble", type=int, default=None, action="store",
        help=(
            "Number of snapshots (may duplicate) in model ensemble. "
            "When forming an ensemble, the final output is determined by "
            "a linear combination of each snapshot model and each individual "
            "snapshot model's output is multiplied by an integer scalar. "
            "This argument determines the maximum sum of these scalars. "
            "Default: 16."
        )
    )
    group.add_argument(
        "-es", "--epoch-per-snapshot", type=int, default=None, action="store",
        help=(
            "Number of epochs between two consecutive model snapshots. "
            "Note: (total-snapshot * epoch-per-snapshot) determines "
            "the total number of training epochs for each (k-1) folds. "
            "Default: 6."
        )
    )
    group.add_argument(
        "-ps", "--positive-smoothing", type=float, default=None, action="store",
        help=(
            "Smoothing factor used for positive class - PSMs with label 1. "
            "Stands for 'the probability of a PSM being mislabeled as "
            "positive (label = 1) but actually negative (label = -1)'. "
            "Default: 0.99."
        )
    )
    group.add_argument(
        "-ns", "--negative-smoothing", type=float, default=None, action="store",
        help=(
            "Smoothing factor used for negative class - PSMs with label -1. "
            "Stands for 'the probability of a PSM being mislabeled as "
            "negative (label = -1) but actually positive (label = 1)'. "
            "Default: 0.99."
        )
    )
    group.add_argument(
        "-fp", "--false-positive-loss-factor",
        type=float, default=None, action="store",
        help=(
            "Factor used to reweight the loss of false positive PSMs. "
            "A false positive PSM: a PSM with label -1 but predicted as 1. "
            "Penalizing false positive PSMs more will help the script "
            "identify more PSMs with lower q-values. "
            "Default: 4.0."
        )
    )
    group.add_argument(
        "-gm", "--gamma", type=float, default=None, action="store",
        help=(
            "The loss factor of spectral angle loss used in fine-tuning. "
            "When fine-tuning, the loss function can be interpreted as: "
            "L = cross_entropy_loss + gamma * spectral_angle_loss. "
            "Default: 0.5."
        )
    )
    # end of PepDream analysis section

    args = parser.parse_args()

    import os
    import json

    # generate default json file
    if args.json_example:
        args.json_example = False
        with open("arguments.json", "w") as file:
            json.dump(DEFAULT_VALUES, file, indent=4)
        path = os.path.abspath("arguments.json")
        print(f"Default arguments are saved in {path}.")
        exit(0)

    from warnings import warn
    # try to load arguments from json file
    if args.json_file:
        assert os.path.exists(args.json_file), (
            "Specified json file doesn't exist."
        )
        with open(args.json_file) as file:
            json_args = json.load(file)
        for key, value in json_args.items():
            # if key is not specified in command line, use value in json file
            if key in vars(args) and vars(args)[key] is None:
                setattr(args, key, value)

    # override remaining not specified arguments with default values
    for key, value in DEFAULT_VALUES.items():
        if vars(args)[key] is None:
            setattr(args, key, value)

    # critical workflow arguments
    def step_name_to_number(step: str):
        step_name = step.lower() if isinstance(step, str) else step
        if step_name in ["parse-maxquant-output", "maxquant", "m", "1", 1]:
            return 1
        if step_name in ["predict-spectrum", "spectrum", "s", "2", 2]:
            return 2
        if step_name in ["rescore-psm", "rescore", "r", "3", 3]:
            return 3
        if step_name in ["finetune-spectrum-model", "finetune", "f", "4", 4]:
            return 4
        print(
            "User specified unknown step name. "
            "Here are the names of each step:\n"
            "STEP1_NAME: parse-maxquant-output, maxquant, m, or 1\n"
            "STEP2_NAME: predict-spectrum, spectrum, s, or 2\n"
            "STEP3_NAME: rescore-psm, rescore, r, or 3\n"
            "STEP4_NAME: finetune-spectrum-model, finetune, f, or 4\n"
        )
        raise ValueError("Unknown step name.")

    starting_step = step_name_to_number(args.starting_step)
    ending_step = step_name_to_number(args.ending_step)
    assert starting_step <= ending_step, (
        "Starting step is after ending step."
    )
    args.starting_step = starting_step
    args.ending_step = ending_step

    steps = list(range(starting_step, ending_step + 1))
    args.parse_maxquant_output = 1 in steps
    args.predict_spectrum = 2 in steps
    args.rescore_psm = 3 in steps
    args.finetune_spectrum_model = 4 in steps

    # path to cache
    args.cache_dir = os.path.join(args.working_dir, "cache")

    # check arguments for parsing MaxQuant output
    parsed_file = os.path.join(args.cache_dir, "msms_raw.hdf5")
    parsed_file_exists = os.path.exists(parsed_file)
    if args.parse_maxquant_output:
        assert args.msms_file and args.csv_dir, (
            "Please specify msms file and csv directory "
            "when parsing MaxQuant output."
        )
        assert os.path.exists(args.msms_file), (
            "Specified msms file doesn't exist."
        )
        assert os.path.exists(args.csv_dir), (
            "Specified csv directory doesn't exist."
        )
        # check if there is any csv file
        csv_files = os.listdir(args.csv_dir)
        csv_files = [file for file in csv_files if file.endswith(".csv")]
        assert len(csv_files) > 0, (
            "Specified csv directory doesn't contain any csv file."
        )
        if parsed_file_exists:
            warn(
                f"{parsed_file} already exists. "
                "To skip parsing MaxQuant output, "
                "please set --starting-step to STEP2_NAME or later."
            )
            print(
                f"Do you want to overwrite {parsed_file}? "
                "Existing results will be lost. "
            )
            prompt = "Type [yes/no] to continue: "
            while True:
                user_input = input(prompt).strip().lower()
                if user_input in ['yes', 'y']:
                    print("Confirm to overwrite.")
                    # remove existing files
                    os.remove(parsed_file)
                    parsed_file_exists = False
                    break
                elif user_input in ['no', 'n']:
                    print(
                        "Skip parsing MaxQuant output. "
                        "Will use existing results."
                    )
                    args.parse_maxquant_output = False
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
    elif starting_step >= 2:
        # check if results are already saved
        assert parsed_file_exists, (
            "We assume that user has already parsed MaxQuant output "
            "and results are stored in --working-dir. "
            f"But parsed results {parsed_file} is not found."
        )
    # end of checking arguments for parsing MaxQuant output

    # check if spectrum model type is supported
    if ending_step >= 2:
        args.spectrum_model_type = args.spectrum_model_type.lower()
        assert args.spectrum_model_type in ["prosit", "pdeep2"], (
            "Specified spectrum model type is not supported."
            "Please choose from 'prosit' and 'pdeep2'."
        )

    # check if spectrum model weight is available
    if args.predict_spectrum or args.finetune_spectrum_model:
        assert args.spectrum_model_weight, (
            "Please specify spectrum model weight, which is used "
            "in predicting spectrum and finetuning spectrum model."
        )
        assert os.path.exists(args.spectrum_model_weight), (
            "Specified spectrum model weight doesn't exist."
        )

    # check arguments for predicting spectrum
    import h5py
    predict = "predict_intensities_" + args.spectrum_model_type
    if parsed_file_exists:
        with h5py.File(parsed_file, "r") as file:
            predict_exists = predict in file
    else:
        predict_exists = False
    if args.predict_spectrum:
        if predict_exists:
            warn(
                f"{predict} already exists. "
                "To skip predicting spectrum, "
                "please set --starting-step to STEP3_NAME or later."
            )
            print(
                "Do you want to predict spectrum again? "
                "This will overwrite existing results. "
            )
            prompt = "Type [yes/no] to continue: "
            while True:
                user_input = input(prompt).strip().lower()
                if user_input in ['yes', 'y']:
                    print("Comfirm to re-predict.")
                    break
                elif user_input in ['no', 'n']:
                    print(
                        "Skip predicting spectrum. "
                        "Will use existing results."
                    )
                    args.predict_spectrum = False
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
    elif starting_step >= 3:
        assert predict_exists, (
            "We assume that user has already predicted spectrum "
            "and results are stored in --working-dir. "
            f"But predicted spectrum {predict} is not found in {parsed_file}."
        )
    # end of checking arguments for predicting spectrum

    # pin file
    if ending_step >= 3:
        if args.pin_file:
            assert os.path.exists(args.pin_file), (
                "Specified pin file doesn't exist."
            )
        else:
            pin_file = os.path.join(args.cache_dir, "andromeda_features.tab")
            args.pin_file = pin_file

    # checking arguments for rescore PSMs
    footprint = os.path.join(args.cache_dir, "training_footprint.hdf5")
    footprint_exists = os.path.exists(footprint)
    if args.rescore_psm:
        if footprint_exists:
            warn(
                f"{footprint} already exists. "
                "To skip rescore PSMs, "
                "please set --starting-step to STEP4_NAME or later."
            )
            print(
                "Do you want to rescore PSMs again? "
                "This will overwrite existing results. "
            )
            prompt = "Type [yes/no] to continue: "
            while True:
                user_input = input(prompt).strip().lower()
                if user_input in ['yes', 'y']:
                    print("Confirm to re-rescore PSMS.")
                    os.remove(footprint)
                    break
                elif user_input in ['no', 'n']:
                    print(
                        "Skip rescore PSMs. "
                        "Will use existing results."
                    )
                    args.rescore_psm = False
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
    if starting_step >= 4:
        assert footprint_exists, (
            "We assume that user has already rescored PSMs "
            "and results are stored in --working-dir. "
            f"But training footprint {footprint} is not found."
        )
    # end of checking arguments for rescore PSMs

    # checking arguments for finetuning spectrum model
    finetuned_models = os.path.join(
        args.working_dir, f"finetuned_{args.spectrum_model_type}"
    )
    if args.finetune_spectrum_model and os.path.exists(args.working_dir):
        # check if there is any finetuned model
        if os.path.exists(finetuned_models) and os.listdir(finetuned_models):
            warn(f"Finetuned {args.spectrum_model_type} model already exists. ")
            print(
                "Do you want to finetune spectrum model again? "
                "This will overwrite existing results. "
            )
            prompt = "Type [yes/no] to continue: "
            while True:
                user_input = input(prompt).strip().lower()
                if user_input in ['yes', 'y']:
                    print("Confirm to re-finetune spectrum model.")
                    # remove existing files
                    from shutil import rmtree
                    rmtree(finetuned_models)
                    break
                elif user_input in ['no', 'n']:
                    print(
                        "Skip finetuning spectrum model. "
                        "Will use existing results."
                    )
                    args.finetune_spectrum_model = False
                    break
                else:
                    print("Please enter 'yes' or 'no'.")

    # working directory
    if not args.working_dir:
        from datetime import datetime
        args.working_dir = datetime.now().strftime("experiment_%Y%m%d_%H%M")

    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)
        print(f"Working directory: {args.working_dir} created.")
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
        print("Cache directory: $working_dir/cache created.")
    if args.finetune_spectrum_model and not os.path.exists(finetuned_models):
        os.makedirs(finetuned_models)
        print(
            f"Finetuned model directory: "
            f"$working_dir/finetuned_{args.spectrum_model_type} created."
        )

    # make paths absolute for disambiguation when stored arguments
    def absolute_path(path):
        return os.path.abspath(path) if path else ""

    args.json_file = absolute_path(args.json_file)
    args.working_dir = absolute_path(args.working_dir)
    args.spectrum_model_weight = absolute_path(args.spectrum_model_weight)
    args.msms_file = absolute_path(args.msms_file)
    args.csv_dir = absolute_path(args.csv_dir)
    args.pin_file = absolute_path(args.pin_file)

    # save arguments as json
    with open(os.path.join(args.working_dir, "arguments.json"), "w") as file:
        json.dump(vars(args), file, indent=4)

    # translate arguments
    import torch
    args.device = torch.device("cpu")
    if args.gpu_id >= 0 and args.predict_spectrum \
            or args.rescore_psm or args.finetune_spectrum_model:
        try:
            args.device = torch.device("cuda:" + str(args.gpu_id))
        except BaseException:
            warn(f"GPU {args.gpu_id} not found. Will use CPU instead.")
    del args.gpu_id

    args.output_dir = args.working_dir
    del args.working_dir

    # no need to check training arguments
    if ending_step < 3:
        return args

    if args.k_fold < 2:
        warn("k-fold is less than 2. Will set it to 2.")
        args.k_fold = 2
    args.k = args.k_fold
    del args.k_fold

    if args.total_iteration < 1:
        warn("total iteration is less than 1. Will set it to 1.")
        args.total_iteration = 1
    args.total_itrs = args.total_iteration
    del args.total_iteration

    if args.output_interval < 1:
        warn("output interval is less than 1. Will set it to 1.")
        args.output_interval = 1

    if args.main_q_threshold < 0 or args.main_q_threshold > 1:
        warn("main-q-threshold is not in range [0, 1]. Will set it to 0.01.")
        args.main_q_threshold = 0.01

    if args.deep_q_threshold < args.main_q_threshold:
        print(
            "Warning: deep-q-threshold is less than main-q-threshold."
            "Will raise it to main-q-threshold."
        )
        args.deep_q_threshold = args.main_q_threshold

    if args.eval_q_threshold < args.deep_q_threshold:
        print(
            "Warning: eval-q-threshold is less than deep-q-threshold."
            "Will raise it to deep-q-threshold."
        )
        args.eval_q_threshold = args.deep_q_threshold

    args.q = args.main_q_threshold
    args.deepq = args.deep_q_threshold
    args.evalq = args.eval_q_threshold
    del args.main_q_threshold
    del args.deep_q_threshold
    del args.eval_q_threshold

    if args.batch_size < 1:
        warn("Batch size is less than 1. Will set it to 1.")
        args.batch_size = 1

    if args.learning_rate < 0:
        warn("Learning rate is less than 0. Will set it to 0.001.")
        args.learning_rate = 0.001
    args.lr_init = args.learning_rate
    del args.learning_rate

    if args.dropout_rate < 0 or args.dropout_rate > 1:
        warn("Dropout rate is not in range [0, 1]. Will set it to 0.4")
        args.dropout_rate = 0.4

    if args.adam_weight_decay < 0:
        warn("Adam weight decay is less than 0. Will set it to 1e-5.")
        args.adam_weight_decay = 1e-5
    args.weight_decay = args.adam_weight_decay
    del args.adam_weight_decay

    if args.total_snapshot < 1:
        warn("Total snapshot is less than 1. Will set it to 1.")
        args.total_snapshot = 1
    args.ensemble_snaps = args.total_snapshot
    del args.total_snapshot

    if args.snapshot_in_ensemble < 1:
        warn("Snapshot in ensemble is less than 1. Will set it to 1.")
        args.snapshot_in_ensemble = 1
    args.ensemble_count = args.snapshot_in_ensemble
    del args.snapshot_in_ensemble

    if args.epoch_per_snapshot < 1:
        warn("Epoch per snapshot is less than 1. Will set it to 1.")
        args.epoch_per_snapshot = 1
    args.ensemble_epoch = args.epoch_per_snapshot
    del args.epoch_per_snapshot

    if args.positive_smoothing < 0 or args.positive_smoothing > 1:
        warn("Positive smoothing is not in range [0, 1]. Will set it to 0.99.")
        args.positive_smoothing = 0.99
    args.pos_smoothing = args.positive_smoothing
    del args.positive_smoothing

    if args.negative_smoothing < 0 or args.negative_smoothing > 1:
        warn("Negative smoothing is not in range [0, 1]. Will set it to 0.99.")
        args.negative_smoothing = 0.99
    args.neg_smoothing = args.negative_smoothing
    del args.negative_smoothing

    if args.false_positive_loss_factor < 0:
        warn("False positive loss factor is less than 0. Will set it to 0.")
        args.false_positive_loss_factor = 0

    return args


def main():
    args = parse_args()
    # placing import workflow here to speed up --help call
    from .workflow import Workflow
    Workflow(args).run()
