This space is for learning nn machine learning using pytorch
I think pytorch is a pthon wrapped version of LibTorch (written in C++), but not 100% sure about that
These programs (mostly) follow the tutorial found here: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html?highlight=blitz
...with some modifications to break up things into different modules.

TO RUN EVERYTHING:
1. INSTALL all 3 modules (_NETdef.py, cudaclassifier.py, and _tester.py) to the same folder

2. Make sure you have the following modules installed to PATH from PyPI: torch, torchvision, numpy, matplotlib

3. Run _NETdef.py to initialize things - This will download 2 PIL image libraries (test and training) and save them in a /data subfolder, Images are classified as: 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
NOTE: If switching to more paralellized processing on a graphics card (see troubleshooting below if needed), the shape of the tensor can be drmaatically increased without a drop in performance (presumably up to the number of cores used?)
This is the second variable in self.conv1 on line 34 and the first variable in self.conv2 on line 36 (THESE VALUES MUST MATCH)

4. Run cudaclassifier.py - this trains the network and saves all of the weights from the trianing run 
NOTE: the epoch in line 99 is set to 2 by defualt and could be changed to increases the number of training runs / accuracy. The best I have obtained from a 25 epoch training run at home over about 20 minutes is ~70%

5. Run _tester.py - this uses the saved training weights to classify unknown (to the program) images in the test set and records the % correct for each prediciton
NOTE: There are 10 catagories possible, so even soemthing like 50% accuracy is pretty good / well beyond random chance.

TROUBLESHOOTING:
pytorch supports running the nn on graphics card processors
IF torch.cuda.is_available() = False THEN The default device output in cuda classifier.py is cpu (indicating that the nn is run on the cpu processor)
IF torch.cuda.is_available() = True THEN device output in cuda classifier.py is cuda (indicating that the nn is run on the graphics card)

FOR running on cuda in WINDOWS 10:
num_workers in lines 39 and 42 must be set to 0
One of the NVidia GEForce Drivers must be installed
