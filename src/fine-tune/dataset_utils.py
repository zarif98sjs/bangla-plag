import os
import json
import logging
import datasets
import huggingface_hub as hub
from typing import Optional
from datasets.io.abc import AbstractDatasetReader
from datasets.utils.typing import NestedDataStructureLike, PathLike
from datasets import Features, NamedSplit
from datasets.tasks import QuestionAnsweringExtractive
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput

with open('/root/.cache/huggingface/token') as f:
    token=f.readlines()[0]


hub.login(os.getenv("HUB_LOGIN"))
logger = logging.getLogger(__name__)

class DatasetBuilder(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        for filepath in files:
            with open(filepath, encoding="utf-8") as f:
                squad = json.load(f)
                for example in squad["data"]:
                    title = example.get("title", "")
                    for paragraph in example["paragraphs"]:
                        context = paragraph["context"]
                        for qa in paragraph["qas"]:
                            question = qa["question"]
                            id_ = qa["id"]

                            answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                            answers = [answer["text"] for answer in qa["answers"]]

                            yield id_, {
                                "title": title,
                                "context": context,
                                "question": question,
                                "id": id_,
                                "answers": {
                                    "answer_start": answer_starts,
                                    "text": answers,
                                },
                            }

class DatasetReader(AbstractDatasetReader):

    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ):
        super().__init__(
            path_or_paths, split=split, features=features, cache_dir=cache_dir, keep_in_memory=keep_in_memory, **kwargs
        )
        path_or_paths = path_or_paths if isinstance(path_or_paths, dict) else {self.split: path_or_paths}
        self.builder = DatasetBuilder(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            **kwargs,
        )

    def read(self):
        download_config = None
        download_mode = None
        ignore_verifications = True
        try_from_hf_gcs = False
        use_auth_token = None
        base_path = None

        self.builder.download_and_prepare(
            download_config=download_config,
            download_mode=download_mode,
            ignore_verifications=ignore_verifications,
            try_from_hf_gcs=try_from_hf_gcs,
            base_path=base_path,
            use_auth_token=use_auth_token,
        )

        dataset = self.builder.as_dataset(
            split=self.split, ignore_verifications=ignore_verifications, in_memory=self.keep_in_memory
        )
        return dataset

class BanglaPlagTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)