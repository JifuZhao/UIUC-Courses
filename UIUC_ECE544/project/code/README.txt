# UIUC ECE544NA Final Project Instructions

########################################################################################
Instruction for running the python3 code

All the required codes are in the folder named code.
        
All data will be automatically downloaded to the folder named ./data/

########################################################################################
To run the code (in Ubuntu 16.04 with Anaconda Python)

    1. $: source activate tensorflow
    
    2. $: python PCA_vs_kernelPCA.py
    
    3. $: python Autoencoder_and_RBM.py    
   
** Note **
    The result will print out on the terminal and/or save to local folder named ./result/

** Important Notice **
    This code will generate huge amount of data and use huge computation resources. You may need to modify the file path in PCA_vs_kernelPCA.py and Autoencoder_and_RBM.py in order to save files into appropriate locations.
    The learned feature for kernel PCA and Auto-encoders has been saved locally, but the size is very big. Please reach the author jzhao59@illinois.edu for details
