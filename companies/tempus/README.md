
Dear candidate,

This is a take home that should not take more than 2 hours to complete.  In this 
directory are just over 100 documents.  There is a reference file called 
entity_names.txt that carry examples of entities we want to extract from these
documents.  For intent of this exercise you only need to look for the listed 
entities in this file.  However for cancertype not all cancertypes are listed, 
you can reference  https://www.cancer.gov/types for the full list.

Expected output:
For each file that has entities of interest as specified in entity_names.txt
produce tsv with 1 entity per line:
filename	entity_type	extracted_entity

e.g.
j###.txt	cancertype	colon cancer

Please submit in zipped file the following
notebook/code for training and inference
brief description of choice of approach for text processing and modeling
extraction output as described above.

