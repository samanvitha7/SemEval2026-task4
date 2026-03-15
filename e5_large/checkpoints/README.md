---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:2076
- loss:TripletLoss
base_model: intfloat/e5-large-v2
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on intfloat/e5-large-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) <!-- at revision f169b11e22de13617baa190a028a32f3493550b6 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'The weather is lovely today.',
    "It's so sunny outside!",
    'He drove to the stadium.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7574, 0.3992],
#         [0.7574, 1.0000, 0.4145],
#         [0.3992, 0.4145, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2,076 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                           | sentence_2                                                                           |
  |:--------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                               | string                                                                               | string                                                                               |
  | details | <ul><li>min: 23 tokens</li><li>mean: 179.87 tokens</li><li>max: 347 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 190.12 tokens</li><li>max: 366 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 200.05 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>In a small coastal town, an elderly artisan named Samuel struggles to keep his traditional craft of shipbuilding alive amidst the rise of modern technology and mass production. As a large corporation plans to establish a factory nearby, Samuel's granddaughter, Lily, returns home from the city, torn between her affection for her grandfather and the allure of a lucrative job offer. Samuel attempts to teach Lily the intricacies of his craft, hoping to pass down his knowledge and preserve their family's legacy. Meanwhile, Lily grapples with her desire to modernize their approach to shipbuilding, believing it could attract younger customers. Tensions escalate as the town's residents are divided over the impending factory, forcing Lily to confront the implications of progress versus tradition. Ultimately, she must choose between the comfort of the new world and the enduring value of her grandfather's time-honored skills. As the story unfolds, a storm threatens the town, leading to an unexpec...</code> | <code>In a quaint village nestled by the sea, an aging craftsman named Harold battles to preserve his age-old trade of pottery as industrialization looms on the horizon. When a major conglomerate proposes to build a plant in the vicinity, Harold’s daughter, Emma, returns from the metropolis, caught between her loyalty to her father and the temptation of a promising career opportunity. Harold strives to impart the delicate art of pottery to Emma, yearning to safeguard their family's traditions. Concurrently, Emma wrestles with her ambition to modernize their pottery techniques, convinced that innovation could draw in a younger clientele. As the community becomes polarized over the factory's arrival, Emma is compelled to face the conflict between advancement and heritage. Ultimately, she must decide whether to embrace the conveniences of modernity or uphold her father's cherished craftsmanship. As the narrative unfolds, a fierce storm approaches, prompting an unforeseen alliance that harmonize...</code> | <code>In a bustling metropolis, a talented but disillusioned graffiti artist named Marco feels the pressure of gentrification as a powerful developer plans to transform his vibrant neighborhood into luxury condos. When a local gallery offers him a chance to showcase his work, Marco faces a dilemma: accept the opportunity that could elevate his career or stay true to his roots and the community that inspired him. As he grapples with this decision, he forms a bond with a young street artist, Mia, who urges him to use his platform to protest the impending changes. Together, they organize a massive mural project that captures the essence of their neighborhood's culture and history, igniting a movement among the residents. As the unveiling approaches, tensions rise between the community and the developer's representatives, leading to a dramatic standoff. In a surprising turn of events, the mural becomes a symbol of resilience, uniting the diverse voices of the neighborhood. Ultimately, Marco real...</code> |
  | <code>A mid-level pharmaceutical researcher discovers that her company has been suppressing clinical trial data showing severe side effects of their new arthritis medication. When she attempts to report the findings to regulatory authorities, she is framed for data theft and fired from her position. The researcher teams up with an investigative journalist to expose the cover-up, but they soon realize the conspiracy extends to multiple pharmaceutical companies and involves bribing FDA officials. As they gather evidence, the researcher's apartment is burglarized and the journalist receives threatening phone calls from unknown individuals. The corporate executives hire a private security firm to discredit the whistleblowers by planting false evidence of mental instability and drug use. In the climax, the researcher testifies before a congressional hearing while assassins attempt to eliminate her before she can present the incriminating documents. The film concludes with several executives being...</code> | <code>A senior environmental scientist at a major chemical corporation uncovers internal documents revealing that the company has been concealing toxic contamination data from their new pesticide product. After she tries to alert environmental protection agencies, she is accused of corporate espionage and terminated from her job. The scientist partners with a freelance investigative reporter to reveal the deception, but they discover the scandal involves several chemical manufacturers and includes payoffs to regulatory inspectors. During their investigation, the scientist's home is broken into and the reporter begins receiving menacing messages from unidentified callers. The company executives employ a private investigation agency to undermine the whistleblowers' credibility by fabricating evidence of psychological disorders and substance abuse. During the final confrontation, the scientist gives testimony at a federal commission hearing while hired operatives try to prevent her from submitt...</code> | <code>A veteran food critic at a prestigious culinary magazine begins experiencing strange memory lapses after dining at several high-end restaurants throughout the city. She initially dismisses the episodes as stress-related, but becomes suspicious when she discovers glowing reviews in her own handwriting for meals she cannot remember enjoying. Her investigation reveals that an exclusive network of restaurateurs has been using experimental flavor enhancers that induce temporary amnesia, allowing them to manipulate critics' experiences and secure positive coverage. When the critic confronts the magazine's editor about her findings, she learns that he has been complicit in the scheme, accepting payments to assign specific critics to targeted establishments. The critic decides to write an exposé, but finds herself unable to trust her own memories or notes, unsure which of her recent experiences were authentic. She ultimately chooses to retire from food criticism entirely, opening a small cooki...</code> |
  | <code>During a global conflict, intelligence officer Daniel Mercer is assigned to infiltrate an enemy-occupied port city under the guise of a traveling merchant. His mission is to locate and sabotage a hidden radio transmitter used to coordinate submarine attacks against allied supply ships. Posing as a trader, Mercer gains the trust of a local resistance cell led by Elise Vautrin, who provides him with access to restricted areas. As enemy patrols intensify, Mercer narrowly avoids capture while gathering coded documents that reveal the transmitter’s location. The pair orchestrate a diversion, allowing Mercer to plant explosives in the facility without alerting nearby troops. Following the successful detonation, Mercer and Elise escape through the city’s sewer tunnels, pursued by enemy agents. The film concludes with them boarding a covert extraction vessel at dawn, as allied ships begin to arrive in the now-secure harbor.</code>                                                                           | <code>Amid a devastating world war, covert operative Adrian Holt is dispatched to infiltrate the enemy-controlled harbor city of Brackenford, posing as an itinerant spice dealer. His objective is to find and destroy a concealed signal tower used to direct enemy submarine strikes on allied convoys. Under his merchant disguise, Holt befriends a resistance leader, Maren Duval, who secures him entry into guarded districts. As surveillance and patrols tighten, Holt narrowly evades arrest while securing encrypted maps pinpointing the tower’s position. Together, they stage a staged warehouse fire to distract the garrison, enabling Holt to rig the transmitter site with explosives undetected. After the blast cripples enemy communications, Holt and Maren flee into the city’s underground aqueducts, trailed by pursuing soldiers. The story ends with their escape aboard a hidden patrol boat at first light, as allied fleets approach the newly liberated port.</code>                                                     | <code>In a remote desert settlement plagued by mysterious sandstorms, former cartographer Liora is hired by a reclusive archaeologist to locate a long-buried obelisk said to influence the winds. Disguising her true intent from the wary townsfolk, she befriends a local glassmaker, Marek, who knows hidden routes through the shifting dunes. As they venture deeper into the wastelands, they discover the obelisk is guarded by a band of nomads who believe dismantling it will unleash even greater storms. Forced to improvise, Liora convinces the nomads to let her “repair” the artifact, secretly altering its mechanism to weaken its pull on the winds. When a sudden storm strikes, she and Marek use the chaos to flee back toward the settlement. The final scene shows the skies clearing for the first time in decades, while Liora quietly departs at dawn, leaving the townsfolk to wonder what truly changed.</code>                                                                                                            |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.COSINE",
      "triplet_margin": 0.35
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 4
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: None
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.9231 | 500  | 0.0216        |
| 3.8462 | 1000 | 0.0002        |


### Framework Versions
- Python: 3.11.7
- Sentence Transformers: 5.2.2
- Transformers: 5.0.0
- PyTorch: 2.5.1+cu121
- Accelerate: 1.10.1
- Datasets: 4.2.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->