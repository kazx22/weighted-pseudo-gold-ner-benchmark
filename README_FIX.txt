MEDGEMMA CHUNKED EXTRACTION FIX
===============================

Problem fixed
-------------
MedGemma sometimes entered a repetitive structured-output loop when asked to
extract every entity from an entire dense BC5CDR abstract. Increasing
num_predict did not help because the model kept generating until the limit.

What changed
------------
1. Each BC5CDR document is split deterministically into short sentence-aligned
   segments (target: 450 characters).
2. MedGemma extracts entities from one segment at a time.
3. Segment-local offsets are shifted back to the original document offsets.
4. If a segment still reaches the generation limit, it is automatically split
   into smaller segments and retried.
5. The schema limits a segment response to 64 entities and the prompt tells the
   model to close the JSON object after the final entity.
6. The tie-breaker experiment is unchanged.

Important cache behaviour
-------------------------
The earlier 26 whole-document results are NOT reused because mixing two
inference protocols would make the experiment inconsistent. This fix creates a
new cache:

results\dev\medgemma\zero_shot_doc_results_chunked_v2.jsonl

The old cache is left untouched. The chunked run starts from document 1 once,
then resumes normally from the new cache after any interruption.

Install
-------
Copy the included src folder into the project root and allow these files to be
replaced:

src\medgemma_common.py
src\medgemma_bc5cdr.py

Run
---
05_RUN_MEDGEMMA_BC5CDR.cmd

Expected startup text
---------------------
Chunk target: 450 characters
Already completed in chunked cache: 0
Cache: ...\zero_shot_doc_results_chunked_v2.jsonl

Validation performed
--------------------
- Python compilation passed.
- Existing five MedGemma extension tests passed.
- New chunking and offset-shift tests passed.
- All 9,591 development gold entities and all 9,809 test gold entities were
  checked against the 450-character boundaries; zero gold entities crossed a
  chunk boundary.
- The failing document row_id 11573852 is split into four segments of 415, 359,
  409, and 194 characters.
