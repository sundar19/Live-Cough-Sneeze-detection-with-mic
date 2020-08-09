# Live-Cough-Sneeze-detection-with-mic
Watch a live demo of this project https://drive.google.com/file/d/1bax30KmnOahCLjbw7z3pqwct8suVJelR/view

This repository aims at detecting live cough/sneeze with mic . It also detects all other sounds included in ESC-50 dataset

Thanks  https://github.com/hasithsura/Environmental-Sound-Classification and https://github.com/karolpiczak/ESC-50

Get the .pth file here https://drive.google.com/file/d/1vba2Fttk4JNssZvUR-p0JhFLsCIURsWA/view?usp=sharing

If you want to train dataset on your own  try https://github.com/hasithsura/Environmental-Sound-Classification  and for dataset https://github.com/karolpiczak/ESC-50

This repository aims just in giving the inference model not required for you to train!! You of course need a Nvidia CUDA enabled GPU for inference.

To use this program ,
1. Run rec_2_db.py and try coughing or sneezing.
2. After some seconds rec_2_detect.py starts running and detects whether its a cough/sneeze or something else. (NOTE: The project is not 100% accurate and is fairly slow and need to be optimized for quicker detection and inference)

Libraries required:
1.Pytorch
2.Librosa
3.pyaudio
4.numpy
and also you require a microphone!!!

NOTE: Specify the paths correctly as mentioned in the program for proper accessing of required file by the program
