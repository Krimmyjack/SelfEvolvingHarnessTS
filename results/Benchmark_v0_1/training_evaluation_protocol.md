# Benchmark benchmark-v0.1 training and evaluation protocol

Normalization is benchmark-owned. Raw means No-op + canonical ingestion. Closed-form, Adam-DLinear, and LSTM share eligibility, windows, ingestion, and normalization.

One shared model is trained per (program, scenario, dose, corruption replicate) on the role's pooled inner-train with series-equal weighting. The training pool is never sliced by regime_tag, which is a benchmark-private diagnostic label.

Aggregation order: model seed -> corruption replicate -> scenario and dose -> one row per uid -> cell series mean -> dataset macro mean within regime.

Final-Query is sealed until one frozen evaluation campaign records durable unseal/access events.
