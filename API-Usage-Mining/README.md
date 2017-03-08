# Thesis code
Nuts and bolts scripts I used for my thesis, regarding API usage examples mining. Since the audience for this will maybe be students with similar assignments I chose to provide some core scripts (something like an "interface") for reference. While they probably won't fit out of the box with your implementation they should provide a pretty detailed picture on how to build your system.

## Contents



- ### repofinder
  Python script to programmatically search github API for projects related to your library of choice. Accepts Github username/password combo as args. After you have your repo list, clone them and build them with Maven (skip tests ;) ) to get the corresponding .jars.

- #### javacg-0.1-SNAPSHOT-static.jar  
  The jar to produce text representations of callgraphs. 

  Call it by `java -jar javacg-0.1-SNAPSHOT-static.jar <jar_you_want_callgraphs_of> >> output.txt`. Find original repo [here](https://github.com/gousiosg/java-callgraph).
  
- ### preprocessing

    Applies preprocessing by grouping and filtering the callgraph files. Next step is the sequence mining algorithm.
- ### Sequence mining
    For that you have a lot of choices, one of many being BIDE algorithm. I personally used sparesort, which I tinkered a bit to accept input in a manner more consistent with 2017 and the needs of my thesis. My version is [here](https://github.com/themistoklik/sparesort), also check out [pymining](https://github.com/bartdag/pymining).

- #### postprocessing
  Frequent closed sequence mining algorithms produce a large number of examples, so you will want to trim them down a bit. One heuristic-based approach is the postprocessing script, where we simply absorb identical sequences and group them based on what package appears the most in each example. Again you have many choices, from clustering algorithms (did not seem very effective) to more complex heuristics based on papers.

- ### metrics
    Reference methods on how to compute various RSSE quality metrics. Some papers to consider [MLUP](http://ieeexplore.ieee.org/document/7081812/), [UP-Miner](http://taoxie.cs.illinois.edu/publications/msr13-upminer.pdf).
