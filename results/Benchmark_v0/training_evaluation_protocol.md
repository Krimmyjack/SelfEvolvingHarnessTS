# Benchmark v0 training and evaluation protocol

Normalization is benchmark-owned. Raw means No-op + canonical ingestion. Closed-form, Adam-DLinear, and LSTM share eligibility, windows, ingestion, and normalization.

Aggregation order: model seed -> corruption replicate -> scenario and dose -> one row per uid -> cell series mean -> dataset macro mean within regime.

Final-Query is sealed until one frozen evaluation campaign records durable unseal/access events.
