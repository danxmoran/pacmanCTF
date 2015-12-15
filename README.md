# pacmanCTF
Capture the flag Pacman game - Final project for CS5100 at Northeastern University

The focus of this project was to design agents that could intelligently play the
CTF game constructed by the [CS188 team at UC Berkeley](http://ai.berkeley.edu).
To run the game with the default agents provided by CS188, run:
```
$ python capture.py
```
Two teams of new agents, a cautious team and an aggressive team, were created for this project. These teams can be used in the game
through the `-r` and `-b` flags of `capture.py`. For example:
```
$ python capture.py -r cautiousTeam
$ python capture.py -b aggressiveTeam
$ python capture.py -r aggressiveTeam -b cautiousTeam
```

All agents created for this project determine actions to take via a combination of Hidden Markov Model inference (for tracking
opponent positions) and expectimax search (for looking into the future assuming the tracked opponent positions).  Code for all
agents created for this project is located in `agents.py`.

If you're interested in playing around with the CTF game, many other options exist! Check them out with:
```
$ python capture.py --help
```
