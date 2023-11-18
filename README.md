# Bangla Plagiarism Dataset and Detection

Term Project for ***CSE 472: Machine Learning Sessional*** offered by the Department of CSE, BUET.

![](static/images/bangla_plag.png)

## Project Overview
This work presents a comprehensive exploration into the domain of plagiarism detection in Bangla language documents, with a focus on the development of a deep learning-based system for accurate identification and quantification of instances of plagiarism. The primary objective of this project was to establish a benchmark for Bangla plagiarism detection by creating a dedicated dataset that encompasses various types of plagiarism.

Our contributions are summarized as follows:

- We create a synthetic dataset that encompasses a variety of plagiarism types, including copy and paste, paraphrasing, and word replacement. This dataset serves as a valuable benchmark for evaluating plagiarism detection systems in the Bangla language.

- We develop two distinct approaches for detecting plagiarism in Bangla documents: First, We fine-tune the pre-trained Bangla BERT model for plagiarism detection. This approach effectively captures the contextual and morphological features of Bangla text, enabling accurate plagiarism identification. Secondly, We utilize sentence embeddings to detect plagiarism across multiple documents. This approach is particularly effective in identifying paraphrased plagiarism, where the meaning of the text remains largely unchanged despite alterations in word choice.

The experimental results highlight the system's effectiveness in accurately detecting and categorizing instances of plagiarism in Bangla text. The proposed benchmark dataset and detection system contribute to the advancement of research in Bangla plagiarism detection, providing a valuable resource for evaluating the performance of future systems and fostering the development of robust solutions in the field.

A presentation on the project is available [here](https://docs.google.com/presentation/d/1vZfF1XGjA3iTFckBCU5wVIA9ClNB3bIm_9j5_PK5Fvs/edit?usp=sharing)

## Dataset

Available at ***Hugging Face*** 🤗 datasets   [bangla-plagiarism-dataset](https://huggingface.co/datasets/zarif98sjs/bangla-plagiarism-dataset)

```bibtex
@misc{bn-plag:2023,
  title={Bangla plagiarism dataset},
  author={Alam, Md. Zarif Ul and Alam, Ramisa and Mahmood, Md. Tareq},
  year={2023},
  publisher={Hugging Face},
  journal={Hugging Face Datasets},
  howpublished={\url{https://huggingface.co/datasets/zarif98sjs/bangla-plagiarism-dataset}},
}
```