To generate new figures, run the following commands after adding new results:

PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe plot_probabilities.py open --save

PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe plot_probabilities.py womens --save


To add new results: 

Use the add command once per game:


python main.py open add <white_player> <black_player> <result> --round <N>
Result values: white (or 1 or w) · draw (or d or ½) · black (or 0 or b)

Player IDs (Open section):

ID	Player
nakamura	Hikaru Nakamura
caruana	Fabiano Caruana
sindarov	Javokhir Sindarov
pragg	R. Praggnanandhaa
giri	Anish Giri
weiyi	Wei Yi
bluebaum	Matthias Blübaum
esipenko	Andrey Esipenko
Example — entering a full round:


PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe main.py open add sindarov pragg draw    --round 5
PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe main.py open add nakamura esipenko white --round 5
PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe main.py open add caruana bluebaum white --round 5
PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe main.py open add weiyi giri draw        --round 5
Then regenerate:


PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe main.py open simulate
PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe plot_probabilities.py open --save
Made a mistake? Remove a result and re-enter it:


PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe main.py open remove sindarov pragg
PYTHONUTF8=1 /c/Users/idoux/anaconda3/python.exe main.py open add sindarov pragg white --round 5
The --round flag is optional but recommended — the probability-over-time chart groups games by round number to draw each data point.
