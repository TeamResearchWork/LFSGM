# LFSRM
Codes for LFSRM model and Diagram-sentence Matching Dataset AI2D#  
----This repository includes the AI2D# dataset for *Diagram-sentence* matching task and the codes for our LFSRM model.

## AI2D#
This newly proposed dataset is developed by ourselves to support the novel research on *diagram-sentence* matching. Thanks to Kembhavi A, Salvato M, Kolve E, et al for offering the original AI2D dataset designed for diagram understanding and TQA tasks.   
For details of AI2D#, we choose 1,400 diagrams that represent topics in primary school natural sciences, such as life cycles, food webs and circuits from AI2D. Each diagram is assigned to five sentences to describe its global or local content. We split AI2D# into 1,000 diagrams and 400 diagrams for training and testing, respectively. 
 
You can click on the link below to download the AI2D# from aliyun drive:
 
https://www.aliyundrive.com/s/6d7mLHLFck8

*NOTES:* For easier use, AI2D# dataset is now undergoing careful cleaning and format standardizing. We first offer the original diagrams that AI2D# includes for better understanding the diagram-sentence matching task. The full AI2D# dataset, including the annotation and work details, will be released before 15/12/2021.

## Codes
The codes for LFSRM is put in the directory 'codes', incuding *model.py* and *train.py*. In *model.py*, we have realized the whole LFSRM model which includes: 1)the local-feedback self-regulating memory module for storing the useful multi-modal information, especially uncommon ones and 2)attention mechanism on local fragments and strengthening measures on sentence-to-diagram alignment direction. These two effectively overcome the serious few-shot content problem and incomplete description problem in diagrams, respectively. In *train.py*, we showed how we trained the model. More detailed codes with comments and explanations are on the way.
