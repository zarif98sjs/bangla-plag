import os
import json
import pickle
import random
import logging
import collections
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Tuple
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    default_data_collator,
)
from normalizer import normalize
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from dataset_utils import BanglaPlagTrainer

logger = logging.getLogger(__name__)
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert", use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained("csebuetnlp/banglabert")
model = model.to("cuda")

def show_members_and_methods(obj):
    object_methods = [method_name for method_name in dir(obj)
                  if callable(getattr(obj, method_name))]
    object_members = [member_name for member_name in dir(obj)
                  if not callable(getattr(obj, member_name))]
    print("MEMBERS:")
    for o in object_members:
        print(o)
    print("METHODS:---------------")
    for o in object_methods:
        print(o)

raw_dataset = load_dataset('zarif98sjs/bangla-plagiarism-dataset', use_auth_token=True)

column_names = raw_dataset["train"].column_names
question_column_name = "input_sentence" if "input_sentence" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]

normalization_kwargs = {
            "unicode_norm": "NFKC",
        }
required_column_names = [
    question_column_name,
    context_column_name,
    answer_column_name
]


def find_all_indices(pattern_str, source_str, overlapping=True):
    index = source_str.find(pattern_str)
    while index != -1:
        yield index
        index = source_str.find(
            pattern_str,
            index + (1 if overlapping else len(pattern_str))
        )

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    allow_null_ans: bool = True,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        allow_null_ans (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`allow_null_ans` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`allow_null_ans=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if allow_null_ans:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[start_index]) == 0
                        or len(offset_mapping[end_index]) == 0
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if allow_null_ans:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if allow_null_ans and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not allow_null_ans:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if allow_null_ans:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, ensure_ascii=False, indent=4) + "\n")
        if allow_null_ans:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, ensure_ascii=False, indent=4) + "\n")

    return all_predictions


def normalize_example(example):
    required_row_values = [example[k] for k in required_column_names if k in example]
    question, context = required_row_values[:2]
    example[question_column_name] = normalize(question, **normalization_kwargs)
    example[context_column_name] = normalize(context, **normalization_kwargs)

    if len(required_row_values) == 3:
        answer = required_row_values[2]
        for i, ans in enumerate(answer["text"]):
            prev_position = answer["answer_start"][i]
            answer["text"][i] = normalize(ans, **normalization_kwargs)

            replace_index = -1
            for j, pos in enumerate(find_all_indices(ans, context)):
                replace_index = j
                if pos == prev_position:
                    break

            if replace_index != -1:
                index_iterator = find_all_indices(
                    answer["text"][i],
                    example[context_column_name]
                )
                for j, pos in enumerate(index_iterator):
                    if j == replace_index:
                        answer["answer_start"][i] = pos
                        assert answer["text"][i] == example[context_column_name][pos: pos + len(answer["text"][i])]
                        break

        example[answer_column_name] = answer

    return example

raw_dataset = raw_dataset.map(
    normalize_example,
    desc="Running normalization on dataset",
)

pad_on_right = tokenizer.padding_side

max_seq_length = 512
doc_stride=256
pad_to_max_length = True


def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples


def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

eval_examples = raw_dataset["validation"]
answerable_indices = [i for i, data in enumerate(eval_examples)
                                     if data['answers']['text']]
unanswerable_indices = [i for i, data in enumerate(eval_examples)
                                     if not data['answers']['text']]
len(answerable_indices), len(unanswerable_indices)


predict_examples = raw_dataset["test"]
answerable_indices = [i for i, data in enumerate(predict_examples)
                                     if data['answers']['text']]
unanswerable_indices = [i for i, data in enumerate(predict_examples)
                                     if not data['answers']['text']]
len(answerable_indices), len(unanswerable_indices)

train_dataset = raw_dataset["train"]

answerable_indices = [i for i, data in enumerate(train_dataset)
                                     if data['answers']['text']]
unanswerable_indices = [i for i, data in enumerate(train_dataset)
                                     if not data['answers']['text']]
len(answerable_indices), len(unanswerable_indices)


training_args = TrainingArguments(
    output_dir = "outputs",
    learning_rate=2e-5,
    warmup_ratio=0.1,
    gradient_accumulation_steps=2,
    weight_decay=0.1,
    lr_scheduler_type="linear",
    per_device_train_batch_size=16,
    logging_strategy = "epoch",
    save_strategy = "epoch",
    num_train_epochs=25,
    fp16=True,
)


with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=2,
        remove_columns=column_names,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

with training_args.main_process_first(desc="validation dataset map pre-processing"):
      eval_dataset = eval_examples.map(
          prepare_validation_features,
          batched=True,
          num_proc=2,
          remove_columns=column_names,
          # load_from_cache_file=not data_args.overwrite_cache,
          desc="Running tokenizer on validation dataset",
      )

with training_args.main_process_first(desc="prediction dataset map pre-processing"):
      predict_dataset = predict_examples.map(
          prepare_validation_features,
          batched=True,
          num_proc=2,
          remove_columns=column_names,
          # load_from_cache_file=not data_args.overwrite_cache,
          desc="Running tokenizer on prediction dataset",
      )


pad_to_max_length = True

data_collator = (
    default_data_collator
    if pad_to_max_length
    else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
)

index = random.sample(range(len(train_dataset)), 1)
print(f"Sample {index} of the training set: {train_dataset[index]}.")

def post_processing_function(examples, features, predictions, stage="eval"):
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        allow_null_ans=True,
        n_best_size=30,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir="outputs",
        log_level=logging.WARNING,
        prefix=stage,
    )

    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
    ]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

metric = load_metric("squad_v2")


def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

trainer = BanglaPlagTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

last_checkpoint = get_last_checkpoint("outputs")

checkpoint = last_checkpoint
os.environ["WANDB_DISABLED"] = "true"
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()

metrics = train_result.metrics
max_train_samples = len(train_dataset)
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

metrics = trainer.evaluate()
max_eval_samples = len(eval_dataset)
metrics["eval_samples"] = len(eval_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


results = trainer.predict(predict_dataset, predict_examples)
metrics = results.metrics
max_predict_samples = len(predict_dataset)
metrics["predict_samples"] = len(predict_dataset)
trainer.log_metrics("predict", metrics)
trainer.save_metrics("predict", metrics)


pickle.dump({"results": results, "predict_examples": predict_examples}, open("results.pkl", "wb"))