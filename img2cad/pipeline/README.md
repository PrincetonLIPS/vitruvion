# Pipeline


This module contains a number of pipeline scripts. Eventually, we wish to create a nearly fully automated pipeline, with the following stages.


1. `filter_sequences_from_source.py`: This first step filters the raw sketches from the JSON tarballs into one large sequence file according
to the criteria we set, including size of sketch and some filtering of sketches with pathological rendering.


2. `tokenize_sequences.py`: This step creates tokens from the given sequence file, and ensures that they are unique. During this step, it also
creates a file of unique sequences according to their tokenization.

3. `prerender_images.py`: This step creates image renderings from the given set of sequences, applying noise if necessary.

