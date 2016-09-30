ECE 544NA Data
Homework 1: binary vowel classification w/linear classifier
Homework 2: 9-class consonant classification w/CNN

This corpus contains three sub-corpora:
  train: use this corpus to train parameters of each algorithm
  dev: use this to compare different hyperparameter settings for each algorithm
  eval: use this corpus to compare different algorithms

Homework 1________________________________________

Distinguish letters in the ee-set (B,C,D,E,G,P,T,V,Z) from letters in
the eh-set (F,L,M,N,S,X) using a linear classifier.  Test different
ways of training the linear classifier: linear node, logistic node,
perceptron, linear SVM.

Labels: are in the files ${corpus}/lab/hw1${corpus}_labels.txt.  Each
line of this file is of the form "digit filename", where digit is
either "0" (ee-set) or "1" (eh-set).  Note that, to train an SVM or
perceptron, you'll need to replace "0" with "-1".

Features: ASCII text files in the directory ${corpus}/mfc each contain
one 16-dimensional mel-frequency-cepstral vector, extracted from the
peak of the vowel.

Auxiliary: the directory ${corpus}/voice contains the audio waveforms.
You do not need this directory to do the lab, but you can use it if
you like.  For example, if the classifier indicates that one
particular waveform is wierd, you can listen to it.  You can even
re-calculuate the MFC coefficients if you want:

FBC(f,t) = ln(sum_k(B(f,k)*abs(sum_n(voice(t*S+n)w(n)exp(-j*2*pi*k*n/N)))))
MFC(m) = sum_f(FBC(f,tmax)*cos(pi*(f+0.5)*m))

where
FBC = filterbank coefficients, computed as in Waibel et al., 1989, thus:
fs = sampling frequency = 16000 Hz
N = round(0.025*fs) = window length = DFT size
n = sample index, 0 <= n <= N-1
k = DFT frequency bin, 0 <= k <= N/2
pi = 3.14159...
j = sqrt(-1)
w(n) = Hamming window = 0.54 - 0.46*cos(2*pi*n/(N-1))
S = window skip parameter = round(0.01*fs)
t = frame index
voice(t*S+n) = audio waveform from the voice directory
f = frequency index of the mel-frequency filter
B(f,k) = k'th coefficient of the f'th Mel filter,
  there are 16 mel filters between 0 and 6000Hz, as in Waibel et al., 1989
ln(y) = natural logarithm of y

MFC = mel-frequency cepstral coefficient.
  MFC is the discrete cosine transform of FBC(:,tmax).
tmax = index of the frame containing peak of the vowel (automatically detected)
m = cepstral lag


--------------------------------------------------------

Homework 2________________________________________

Classify the consonant at the beginning of a letter into one of nine
different consonants, using a time-domain neural net (TDNN), which is
a kind of convolutional neural net (CNN).  

Labels: are in the files ${corpus}/lab/hw2${corpus}_labels.txt.  Each
line of this file is of the form "digit filename", where digit is
0=B, 1=C, 2=D, 3=E, 4=G, 5=P, 6=T, 7=V, 8=Z.

Features: ASCII text files in the directory ${corpus}/fbc each contain
a matrix of filterbank coefficients from the corresponding waveform.
Each row contains one frame, which is a 16-dimensional vector.  The
number of frames depends on the waveform.

Auxiliary: the directory ${corpus}/voice contains the audio waveforms.
You do not need this directory to do the lab, but you can use it if
you like.  For example, if the classifier indicates that one
particular waveform is wierd, you can listen to it.  You can even
re-calculuate the FBC coefficients if you want:

FBC(f,t) = ln(sum_k(B(f,k)*abs(sum_n(voice(t*S+n)w(n)exp(-j*2*pi*k*n/N)))))

where
FBC = filterbank coefficients, computed as in Waibel et al., 1989, thus:
fs = sampling frequency = 16000 Hz
N = round(0.025*fs) = window length = DFT size
n = sample index, 0 <= n <= N-1
k = DFT frequency bin, 0 <= k <= N/2
pi = 3.14159...
j = sqrt(-1)
w(n) = Hamming window = 0.54 - 0.46*cos(2*pi*n/(N-1))
S = window skip parameter = round(0.01*fs)
t = frame index
voice(t*S+n) = audio waveform from the voice directory
f = frequency index of the mel-frequency filter
B(f,k) = k'th coefficient of the f'th Mel filter,
  there are 16 mel filters between 0 and 6000Hz, as in Waibel et al., 1989
ln(y) = natural logarithm of y
