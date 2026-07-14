# Benchmark benchmark-v0.2 training and evaluation protocol

Normalization is benchmark-owned. Raw means No-op + canonical ingestion. Closed-form, Adam-DLinear, and LSTM share eligibility, windows, ingestion, and normalization.

One model is trained per (program, scenario, dose, corruption replicate, dataset) on that dataset's inner-train with series-equal weighting. The frozen spec fixes trainer internals but never fixed the training pool's scope; v0.2 fixes it at dataset. The pool is never sliced by regime_tag, which is a benchmark-private diagnostic label, and dataset_id is public metadata, so slicing by it leaks nothing.

Oracles are RETRAINED: once a policy picks a program per (cell, scenario, dose), the corpus those picks produce is assembled and a model is trained on it, through the same path a Method takes. An oracle read off single-program models describes a corpus no model was ever fitted to and is reported as descriptive only.

Aggregation order: model seed -> corruption replicate -> scenario and dose -> one row per uid -> cell series mean -> dataset macro mean within regime.

Final-Query is sealed until one frozen evaluation campaign records durable unseal/access events.
