from __future__ import annotations

import os
import torch
import onnx
import tempfile
import onnxruntime
import numpy as np

from ruamel import yaml
from pathlib import Path
from copy import deepcopy
from sklearn.utils import resample
from torch.utils.data import DataLoader
from typing import Union, Optional, Dict, Iterable, Tuple, Any, List

from ..losses import get_loss
from ..models import get_model
from ..protocols import Protocol
from ..optimizers import get_optimizer
from ..datasets import get_dataset, get_collate_function
from ..solvers import get_solver, get_mean_and_confidence_range, MetricsCalculator
from ..utilities import get_device, seed_all, DatasetSample, MASK_AND_LABELS_PAD_VALUE, revert_mappings, __version__


class Inferencer:

    def __init__(
            self,
            # Constant parameters for all split solvers
            protocol: str,
            embedder_name: str,
            n_features: int,
            # Optional constant parameters for all split solvers
            class_int_to_string: Optional[Dict[int, str]] = None,
            class_str_to_int: Optional[Dict[str, int]] = None,
            device: Union[None, str, torch.device] = None,
            disable_pytorch_compile: Optional[bool] = None,
            allow_torch_pt_loading: bool = True,
            # Everything else
            **kwargs
    ):
        self.protocol = Protocol.from_string(protocol)
        self.embedder_name = embedder_name
        self.embedding_dimension = n_features
        self.class_int2str = class_int_to_string
        self.class_str2int = class_str_to_int
        self.device = get_device(device)
        self.disable_pytorch_compile = True if disable_pytorch_compile is None else disable_pytorch_compile
        self.allow_torch_pt_loading = allow_torch_pt_loading
        self.collate_function = get_collate_function(self.protocol)

        self.solvers_and_loaders_by_split = self._create_solvers_and_loaders_by_split(**kwargs)
        print(f"Got {len(self.solvers_and_loaders_by_split.keys())} split(s): "
              f"{', '.join(self.solvers_and_loaders_by_split.keys())}")

    @classmethod
    def create_from_out_file(cls, out_file_path: str,
                             automatic_path_correction: bool = True,
                             allow_torch_pt_loading: bool = True
                             ) -> Tuple[Inferencer, Dict[str, Any]]:
        """
        Create the inferencer object from the out.yml file generated by biotrainer.
        Reads the out.yml file without the split ids which would blow up the file unnecessarily.

        :param out_file_path: Path to out.yml file generated by biotrainer
        :param automatic_path_correction: If True, the method tries to correct the log_dir path if it is not found.
                                          In this case, checkpoints are searched at
                                          out_file_path/model_choice/embedder_name
        :param allow_torch_pt_loading: If False, only safetensors model serialization checkpoints (biotrainer >0.9.1)
                                       are allowed to be loaded for safety reasons

        :return: Tuple with Inferencer object configured with the output variables from the out.yml file,
                 and the output variables as dict
        """

        print(f"Reading {out_file_path}..")
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_output_path = tmp_dir_name + "/tmp_output.yml"
            with open(out_file_path, "r") as output_file, open(tmp_output_path, "w") as tmp_output_file:
                ids_list = False
                for line in output_file.readlines():
                    if line.strip() == "training_ids:" or line.strip() == "validation_ids:":
                        ids_list = True
                        continue
                    elif ids_list and ("-" in line and ":" not in line):
                        continue
                    else:
                        ids_list = False
                    if not ids_list:
                        tmp_output_file.write(line)

            with open(tmp_output_path, "r") as tmp_output_file:
                output_vars = yaml.load(tmp_output_file, Loader=yaml.RoundTripLoader)

        if automatic_path_correction:
            log_dir = output_vars["log_dir"]
            log_dir_path = Path(log_dir)
            if not log_dir_path.exists():
                # Expect checkpoints to be in output/model_choice/embedder_name
                checkpoints_path = output_vars["model_choice"] + "/" + output_vars["embedder_name"]
                new_log_dir_path = Path("/".join(
                    [directory for directory in out_file_path.split("/")[0:-1]])) / Path(checkpoints_path)

                if not new_log_dir_path.exists():
                    print(f"Could not automatically correct the checkpoint file paths! "
                          f"Tried: {str(new_log_dir_path)} but it does not exist.")
                elif len(os.listdir(str(new_log_dir_path))) == 0:
                    print(f"Found corrected path ({str(new_log_dir_path)}), but it does not contain any files!")
                else:
                    print(f"Reading checkpoint(s) from directory: {new_log_dir_path}..")
                    output_vars["log_dir"] = new_log_dir_path

        output_vars["allow_torch_pt_loading"] = allow_torch_pt_loading

        if output_vars["biotrainer_version"] != __version__:
            print("WARNING: The loaded model was trained on a different biotrainer version than currently running.\n"
                  "This may lead to unexpected behaviour if another torch version was used for training.")
        return cls(**output_vars), output_vars

    def _create_solvers_and_loaders_by_split(self, **kwargs) -> Dict[str, Tuple[Any, Any]]:
        result_dict = {}
        splits = kwargs["split_results"].keys()
        log_dir = kwargs["log_dir"]
        split_checkpoints = {file.split("_checkpoint.")[0]: file for file in os.listdir(log_dir) if
                             (Path(log_dir) / Path(file)).is_file()}
        for split in splits:
            # Ignore average or best result
            if "average" in split or "best" in split:
                continue
            split_config = deepcopy(kwargs)
            for key, value in kwargs["split_results"][split]["split_hyper_params"].items():
                split_config[key] = value

            # Positional arguments
            model_choice = split_config.pop("model_choice")
            n_classes = split_config.pop("n_classes")
            loss_choice = split_config.pop("loss_choice")
            optimizer_choice = split_config.pop("optimizer_choice")
            learning_rate = split_config.pop("learning_rate")
            log_dir = split_config.pop("log_dir")
            checkpoint_path = Path(log_dir) / Path(split_checkpoints[split])

            model = get_model(protocol=self.protocol, model_choice=model_choice,
                              n_classes=n_classes, n_features=self.embedding_dimension,
                              disable_pytorch_compile=self.disable_pytorch_compile,
                              **split_config
                              )
            loss_function = get_loss(protocol=self.protocol, loss_choice=loss_choice,
                                     device=self.device,
                                     **split_config
                                     )
            optimizer = get_optimizer(protocol=self.protocol, optimizer_choice=optimizer_choice,
                                      model_parameters=model.parameters(), learning_rate=learning_rate,
                                      **split_config
                                      )

            solver = get_solver(protocol=self.protocol, name=split, network=model, optimizer=optimizer,
                                loss_function=loss_function, device=self.device, log_dir=log_dir,
                                num_classes=n_classes)
            solver.load_checkpoint(checkpoint_path=checkpoint_path, resume_training=False,
                                   allow_torch_pt_loading=self.allow_torch_pt_loading)

            def dataloader_function(dataset):
                return DataLoader(dataset=dataset, batch_size=split_config["batch_size"],
                                  shuffle=False, drop_last=False,
                                  collate_fn=self.collate_function)

            result_dict[split] = (solver, dataloader_function)
        return result_dict

    def _convert_class_str2int(self, to_convert: str):
        if type(to_convert) is str:
            if self.protocol in Protocol.per_residue_protocols():
                return [self.class_str2int[t] for t in to_convert]
            else:
                return self.class_str2int[to_convert]
        else:
            return to_convert

    @staticmethod
    def _pad_tensor(protocol: Protocol, target: Union[Any, torch.Tensor], length_to_pad: int, device):
        target_tensor = torch.as_tensor(target, device=device)
        if protocol in Protocol.per_residue_protocols():
            if target_tensor.shape[0] < length_to_pad:
                padding_size = length_to_pad - target_tensor.shape[0]
                padding = torch.full((padding_size,), MASK_AND_LABELS_PAD_VALUE, dtype=target_tensor.dtype,
                                     device=device)
                return torch.cat([target_tensor, padding])
            else:
                return target_tensor
        else:
            return target_tensor

    def _convert_target_dict(self, target_dict: Dict[str, str]):
        if self.protocol in Protocol.classification_protocols():
            if self.protocol in Protocol.per_residue_protocols():
                max_prediction_length = len(max(target_dict.values(), key=len))
                return {seq_id: self._pad_tensor(protocol=self.protocol, target=self._convert_class_str2int(prediction),
                                                 length_to_pad=max_prediction_length, device=self.device)
                        for seq_id, prediction in target_dict.items()}
            else:
                return {seq_id: torch.tensor(self._convert_class_str2int(prediction),
                                             device=self.device)
                        for seq_id, prediction in target_dict.items()}
        return {seq_id: torch.tensor(prediction,
                                     device=self.device)
                for seq_id, prediction in target_dict.items()}

    def _load_solver_and_dataloader(self, embeddings: Union[Iterable, Dict],
                                    split_name, targets: Optional[List] = None):
        if split_name not in self.solvers_and_loaders_by_split.keys():
            raise Exception(f"Unknown split_name {split_name} for given configuration!")

        if isinstance(embeddings, Dict):
            embeddings_dict = embeddings
        else:
            embeddings_dict = {str(idx): embedding for idx, embedding in enumerate(embeddings)}

        if targets and self.protocol in Protocol.classification_protocols():
            targets = [self._convert_class_str2int(target) for target in targets]

        solver, loader = self.solvers_and_loaders_by_split[split_name]
        dataset = get_dataset(self.protocol, samples=[
            DatasetSample(seq_id, torch.tensor(np.array(embedding)),
                          torch.empty(1) if not targets else torch.tensor(np.array(targets[idx])))
            for idx, (seq_id, embedding) in enumerate(embeddings_dict.items())
        ])
        dataloader = loader(dataset)
        return solver, dataloader

    def convert_all_checkpoints_to_safetensors(self) -> None:
        """
        Converts all checkpoint files for the splits from .pt to .safetensors, if not already stored as .safetensors
        """
        for split_name, (solver, _) in self.solvers_and_loaders_by_split.items():
            if "pt" in solver.checkpoint_type:
                solver.save_checkpoint(solver.start_epoch)

    def convert_to_onnx(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Converts the model to ONNX format for the given embedding dimension.

        :param output_dir: The directory to save the ONNX files. If not provided, the ONNX files will be saved in the
            solver's log directory. Defaults to None.

        :return: A list of file paths where the ONNX files are saved.
        """
        result_file_paths = []
        for split_name, (solver, _) in self.solvers_and_loaders_by_split.items():
            onnx_save_path = solver.save_as_onnx(embedding_dimension=self.embedding_dimension, output_dir=output_dir)
            result_file_paths.append(onnx_save_path)
        return result_file_paths

    def from_embeddings(self, embeddings: Union[Iterable, Dict], targets: Optional[List] = None,
                        split_name: str = "hold_out",
                        include_probabilities: bool = False) -> Dict[str, Union[Dict, str, int, float]]:
        """
        Calculate predictions from embeddings.

        :param embeddings: Iterable or dictionary containing the input embeddings to predict on.
        :param targets: Iterable that contains the targets to calculate metrics
        :param split_name: Name of the split to use for prediction. Default is "hold_out".
        :param include_probabilities: If True, the probabilities used to predict classes are also reported.
                                      Is only useful for classification tasks, otherwise the "probabilities" are the
                                      same as the predictions.
        :return: Dictionary containing the following sub-dictionaries:
                 - 'metrics': Calculated metrics if 'targets' are given, otherwise 'None'.
                 - 'mapped_predictions': Class or value prediction from the given embeddings.
                 - 'mapped_probabilities': Probabilities for classification tasks if include_probabilities is True.
                 Predictions and probabilities are either 'mapped' to keys from an embeddings dict or indexes if
                 embeddings are given as a list.
        """
        solver, dataloader = self._load_solver_and_dataloader(embeddings, split_name, targets)

        inference_dict = solver.inference(dataloader, calculate_test_metrics=targets is not None)
        predictions = inference_dict["mapped_predictions"]

        # For class predictions, revert from int (model output) to str (class name)
        inference_dict["mapped_predictions"] = revert_mappings(protocol=self.protocol, test_predictions=predictions,
                                                               class_int2str=self.class_int2str)
        inference_dict["mapped_probabilities"] = {k: v.cpu().tolist() if v is torch.tensor else v for k, v in
                                                  inference_dict["mapped_probabilities"].items()}

        if not include_probabilities:
            return {k: v for k, v in inference_dict.items() if k != "mapped_probabilities"}
        else:
            return inference_dict

    def from_embeddings_with_bootstrapping(self, embeddings: Union[Iterable, Dict], targets: List,
                                           split_name: str = "hold_out",
                                           iterations: int = 30,
                                           sample_size: int = -1,
                                           confidence_level: float = 0.05,
                                           seed: int = 42) -> Dict[str, Dict[str, float]]:
        """
        Calculate predictions from embeddings.

        :param embeddings: Iterable or dictionary containing the input embeddings to predict on.
        :param targets: Iterable that contains the targets to calculate metrics
        :param split_name: Name of the split to use for prediction. Default is "hold_out".
        :param iterations: Number of iterations to perform bootstrapping
        :param sample_size: Sample size to use for bootstrapping. -1 defaults to all embeddings which is recommended.
                            It is possible, but not recommended to use a sample size larger or smaller
                            than the number of embeddings, because this might render the variance estimate unreliable.
                            See: https://math.mit.edu/~dav/05.dir/class24-prep-a.pdf (6.2)
        :param confidence_level: Confidence level for result error intervals (0.05 => 95% percentile)
        :param seed: Seed to use for the bootstrapping algorithm
        :return: Dictionary containing the following sub-dictionaries:
                 - 'metrics': Calculated metrics if 'targets' are given, otherwise 'None'.
                 - 'mapped_predictions': Class or value prediction from the given embeddings.
                 - 'mapped_probabilities': Probabilities for classification tasks if include_probabilities is True.
                 Predictions and probabilities are either 'mapped' to keys from an embeddings dict or indexes if
                 embeddings are given as a list.
        """
        if not 0 < confidence_level < 1:
            raise Exception(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

        seed_all(seed)

        if isinstance(embeddings, Dict):
            embeddings_dict = embeddings
        else:
            embeddings_dict = {str(idx): embedding for idx, embedding in enumerate(embeddings)}

        seq_ids = list(embeddings_dict.keys())

        all_predictions = self.from_embeddings(embeddings_dict, targets)["mapped_predictions"]
        all_predictions_dict = self._convert_target_dict(all_predictions)

        all_targets_dict = {seq_id: targets[idx] for idx, seq_id in enumerate(seq_ids)}
        all_targets_dict = self._convert_target_dict(all_targets_dict)

        solver, _ = self.solvers_and_loaders_by_split[split_name]

        return self._do_bootstrapping(iterations=iterations, sample_size=sample_size, confidence_level=confidence_level,
                                      seq_ids=seq_ids, all_predictions_dict=all_predictions_dict,
                                      all_targets_dict=all_targets_dict, metrics_calculator=solver.metrics_calculator)

    @staticmethod
    def _do_bootstrapping(iterations: int,
                          sample_size: int,
                          confidence_level: float,
                          seq_ids: List[str],
                          all_predictions_dict: Dict,
                          all_targets_dict: Dict,
                          metrics_calculator: MetricsCalculator):
        """

        :param iterations: Number of iterations to perform bootstrapping
        :param sample_size: Sample size to use for bootstrapping. -1 defaults to all embeddings which is recommended.
                            It is possible, but not recommended to use a sample size larger or smaller
                            than the number of embeddings, because this might render the variance estimate unreliable.
                            See: https://math.mit.edu/~dav/05.dir/class24-prep-a.pdf (6.2)
        :param confidence_level: Confidence level for result error intervals (0.05 => 95% percentile)
        :param seq_ids: List of sequence IDs
        :param all_predictions_dict: Dictionary of all predictions
        :param all_targets_dict: Dictionary of all targets
        :param metrics_calculator: Metrics calculator object
        :return:
        """
        if sample_size == -1:
            sample_size = len(seq_ids)
        # Bootstrapping: Resample over keys to keep track of associated targets
        iteration_results = []

        for iteration in range(iterations):
            bootstrapping_sample = resample(seq_ids, replace=True, n_samples=sample_size)
            sampled_predictions = torch.stack([all_predictions_dict[seq_id] for seq_id in bootstrapping_sample])
            sampled_targets = torch.stack([all_targets_dict[seq_id] for seq_id in bootstrapping_sample])
            iteration_result = metrics_calculator.compute_metrics(predicted=sampled_predictions,
                                                       labels=sampled_targets)
            iteration_results.append(iteration_result)

        # Calculate mean and error margin for each metric
        metrics = list(iteration_results[0].keys())
        result_dict = {}
        for metric in metrics:
            all_metric_values = [iteration_result[metric] for iteration_result in iteration_results]
            mean, confidence_range = get_mean_and_confidence_range(
                values=torch.tensor(all_metric_values, dtype=torch.float16),
                dimension=0,
                confidence_level=confidence_level)
            result_dict[metric] = {"mean": mean.item(), "error": confidence_range.item()}
        return result_dict

    def from_embeddings_with_monte_carlo_dropout(self, embeddings: Union[Iterable, Dict],
                                                 split_name: str = "hold_out",
                                                 n_forward_passes: int = 30,
                                                 confidence_level: float = 0.05,
                                                 seed: int = 42) -> Dict:
        """
        Calculate predictions by using Monte Carlo dropout.
        Only works if the model has at least one dropout layer employed.
        Method to quantify the uncertainty within the model.

        :param embeddings: Iterable or dictionary containing the input embeddings to predict on.
        :param split_name: Name of the split to use for prediction. Default is "hold_out".
        :param n_forward_passes: Number of times to repeat the prediction calculation
                                with different dropout nodes enabled.
        :param confidence_level: Confidence level for the result confidence intervals. Default is 0.05,
                                which corresponds to a 95% percentile.
        :param seed: Seed to use for the dropout predictions

        :return: Dictionary containing with keys that will either be taken from the embeddings dict or
         represent the indexes if embeddings are given as a list. Contains the following values for each key:
                 - 'prediction': Class or value prediction based on the mean over `n_forward_passes` forward passes.
                 - 'mcd_mean': Average over `n_forward_passes` forward passes for each class.
                 - 'mcd_lower_bound': Lower bound of the confidence interval using a normal distribution with the given
                                      confidence level.
                 - 'mcd_upper_bound': Upper bound of the confidence interval using a normal distribution with the given
                                      confidence level.
        """

        # Necessary because dropout layer have a random part by design
        seed_all(seed)

        solver, dataloader = self._load_solver_and_dataloader(embeddings, split_name)

        predictions = solver.inference_monte_carlo_dropout(dataloader=dataloader,
                                                           n_forward_passes=n_forward_passes,
                                                           confidence_level=confidence_level)["mapped_predictions"]

        # For class predictions, revert from int (model output) to str (class name)
        if self.protocol in Protocol.per_residue_protocols():
            for seq_id, prediction_list in predictions.items():
                for prediction_dict in prediction_list:
                    prediction_dict["prediction"] = list(revert_mappings(protocol=self.protocol,
                                                                         test_predictions={
                                                                             seq_id: prediction_dict["prediction"]
                                                                         },
                                                                         class_int2str=self.class_int2str).values())[0]
        else:
            for seq_id, prediction_dict in predictions.items():
                prediction_dict["prediction"] = list(revert_mappings(protocol=self.protocol,
                                                                     test_predictions={
                                                                         seq_id: prediction_dict["prediction"]
                                                                     },
                                                                     class_int2str=self.class_int2str).values())[0]

        return predictions

    @staticmethod
    def from_onnx_with_embeddings(model_path: str, embeddings: Union[Iterable, Dict],
                                  protocol: Optional[Protocol] = None):
        if isinstance(embeddings, Dict):
            embeddings_dict = embeddings
        else:
            embeddings_dict = {str(idx): embedding for idx, embedding in enumerate(embeddings)}

        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)

        ep_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession(model_path, providers=ep_list)

        result_dict = {}
        for seq_id, embedding in embeddings_dict.items():
            input_feed = {ort_session.get_inputs()[0].name: np.expand_dims(embedding, axis=0)}
            ort_outs = ort_session.run(None, input_feed=input_feed)
            if protocol is not None and protocol in Protocol.classification_protocols():
                result_dict[seq_id] = torch.softmax(torch.tensor(ort_outs[0][0]), dim=0).tolist()
            else:
                result_dict[seq_id] = ort_outs[0][0].tolist()[0]
        return result_dict
