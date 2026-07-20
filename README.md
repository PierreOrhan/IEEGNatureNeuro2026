# Code to replicate the results of the Nature Neuroscience 2026 submission: "Tree-like neural codes for syntax in the human brain"

Requirements:

    - Python 3.12 with torch and cuda well setup.

    - datalad: https://www.datalad.org/

    - neuralset: https://github.com/facebookresearch/neuroai

The exact versions of the packages used to run the analyses are listed in the requirements.txt file. You can install them using pip:

        pip install -r requirements.txt

Hardware requirements:

    - 32GB RAM

    - 8 CPUs

    - 1 GPU 

How to install the package:

        pip install -e ./

Folder structure:
- alpes: repository for encoding classes

- ieegNatureNeuro2026: repository for study definition.

- scratchieegNatureNeuro2026: repository for analysis script, data preprocessing and dataset generation
        

    
Then follow these steps:
# Download and format the dataset:
Steps:
1) Download the dataset:
        datalad install https://github.com/OpenNeuroDatasets/ds005574.git
        cd ds005574
        datalad get ./sub-*
        datalad get ./stimuli/podcast.wav
        cp /path/to/package/features/transcript_withsentence.tsv /path/to/ds005574/stimuli/syntactic/transcript_withsentence.tsv 
(The last line adds the transcript annotation inside the dataset)

2) Run the data preprocessing script:
First, change the path in
        scratchieegNatureNeuro2026/datasetGen/genPodcast.py 

to fit the path of the Podcast dataset on your machine

Then:

        python scratchieegNatureNeuro2026/datasetGen/genPodcast.py

Typically, this will take about 30 minutes to run on a machine with 32GB RAM and 8 CPUs.

# Encoding Analyses:
3) Banded Ridge regression analysis: run all scripts in scratchieegNatureNeuro2026/analysis

4) Display the results: run the bestFit and modelComparison scripts in scratchieegNatureNeuro2026/display

The output of these scripts will be saved in scratchieegNatureNeuro2026/figures and can be used to replicate the figures in the paper.

The expected run time for these analyses is about 4 hours.