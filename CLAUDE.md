# Generazione test set Q&A con ragas

## Obiettivo

Generare un test set sintetico di Q&A con la libreria Python `ragas`, a partire
dalla knowledge base descritta sotto, per valutare la pipeline RAG che ci
costruisco sopra.

## La knowledge base

`C:\Users\peppe\Desktop\coding\pdf-ingestion\test\output_run_1`

Un sample di qualche PDF passato attraverso la mia pipeline di ingestion.
Non è la KB completa: serve a lavorarci sopra.

PDF parsati in Markdown con **Docling**, con immagini e tabelle tenute fuori
dal testo:

- il testo sta nei `.md`
- le immagini sono file `.png`, le tabelle file `.html`, salvati a parte
- nei `.md` ogni asset è referenziato inline da un anchor `[[IMG:<id_hash>]]`
- per ogni asset c'è un record JSON con i metadati di arricchimento

## Record JSON di un asset

Prodotto da una pipeline AI/computer vision che descrive ogni asset o lo scarta
se non informativo.

Campi che contano:

- `id` — corrisponde all'`<id_hash>` dell'anchor nel markdown
- `type` — `image` o `table`
- `content_file` — nome del `.png` / `.html`
- `caption` — didascalia nativa del documento, spesso corta o inutile
- `retrieval_text` — descrizione generata dalla pipeline CV: è il campo buono
- `embedded_labels` — testo letto dentro l'immagine
- `section_path` — sezione del documento da cui viene l'asset
- `doc_id`, `page_no`, `bbox` — provenienza
- `filtered_out` / `filtered_reason` — se `true`, l'asset è da ignorare

## Funzionamento del retrieval

In fase di retrieval il vector store conterrà 3 tipi di chunk:

- `testuali`: indice e payload entrambi testuali.
- `immagini`: payload=file.png indicizzato nel vector store tramite il suo `retrieval_text`
- `tabelle`:  payload=file.html indicizzato nel vector store tramite il suo `retrieval_text`

Il retrieval che intendo implementare con questo vector store è il seguente: la query viene usata per
recuperare dal vector store i topK chunks, che possono essere sia testuali che immagine/tabella: quelli
testuali (md) vengono arricchiti andando a recuperare anche le immagini e tabelle, grazie agli anchor
[[IMG]] e [[TABLE]]. Tutti questi dati vengono usati per generare il contesto (multimodale) per la query.

## Note sui dati

- I contenuti della knowledge base sono documenti tecnici navali: manuali, brochure, datasheet, ecc.. . Il contesto di appartenenza è quello navale