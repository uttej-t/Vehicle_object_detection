To-do:
1. Extract the contents of the file ODLS
2. The input files(images or videos) are to be placed in "Source files" folder
3. Then to execute the algorithms on the files run "main.py"
4. You can run the main.py file by executing "python main.py" from a terminal opened in the "ODLS" folder.
5. Final output of the algorithms is written to the "Output" folder with a prefix "out_" to the actual file.


Folder Structure:

ODLS
|-> Lib
|	|
|	|-> site-packages -> packages files
|
|-> LS
|	|
|	|-> __init__.py
|	|-> LaneSegmentation.py
|
|-> OD
|	|
|	|-> Configuration files
|	|	|
|	|	|-> Classes.txt
|	|	|-> yolov3.cfg
|	|	|-> yolov3.weights
|	|
|	|-> __init__.py
|	|-> ObjectDetection.py
|	
|-> Output
|
|-> Scripts -> python files
|
|-> Source files
|
|-> main.py
