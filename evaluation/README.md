FaceRec Evaluation
==================


## Ground Truth creation

We realised two ground truth for evaluating the system.

Approach:
- Create a list of people and train the system on them
- Retrieve from the Knowledge Graphs the shots/segments in which these people _should_ appear.
- Run a first face recognition, in order to artificially realise a subset of appearences of each person.
- We manually correct the obtained dataset, adding also some shots/segments not involving any of the specified people.

Notebooks for the generation:
- [ANTRACT](./ground_truth_antract.ipynb)
- [MeMAD](./ground_truth_memad.ipynb)

Some differences between the two datasets:

|                                   |            ANTRACT            |           MeMAD          |
|:---------------------------------:|:-----------------------------:|:------------------------:|
|                type               |       historical images       |          TV news         |
|               years               |           1945-1969           |           2014           |
|             resolution            |            512x384            |          455x256         |
|             colorspace            |              b/w              |          colour          |
|           shots division          |              yes              |            no            |
|   list of celebrities to search   | 13 (chosen by domain experts) |  6 (most present in KG)  |
|  represented fragment and length  |    shot, 3 seconds in avg.    | segment, up to 2 minutes |
|              records              |              216              |            100           |
|         distinct fragments        |              198              |            100           |
|           distinct media          |              129              |            30            |
| nb. fragments without known faces |               39              |            43            |

## Evaluation

We run face recognition on the two datasets and measure the matches with the expected people.

Notebooks for the evaluation:
- [ANTRACT](./evaluation_antract.ipynb)
- [MeMAD](./evaluation_memad.ipynb)

Some results:

| ANTRACT                 | Precision | Recall | F-Score | Support |
|-------------------------|:---------:|:------:|:-------:|:-------:|
| Ahmed Ben Bella         |    1.00   |  0.46  |   0.63  |    13   |
| François Mitterrand     |    1.00   |  0.92  |   0.96  |    13   |
| Pierre Mendès France    |    1.00   |  0.61  |   0.76  |    13   |
| Guy Mollet              |    0.92   |  0.92  |   0.92  |    13   |
| Georges Bidault         |    0.83   |  0.71  |   0.76  |    14   |
| Charles De Gaulle       |    1.00   |  0.57  |   0.73  |    19   |
| Nikita Khrouchtchev     |    1.00   |  0.38  |   0.55  |    13   |
| Vincent Auriol          |    1.00   |  0.46  |   0.63  |    13   |
| Konrad Adenauer         |    1.00   |  0.53  |   0.70  |    13   |
| Dwight Eisenhower       |    0.85   |  0.46  |   0.60  |    13   |
| Elisabeth II            |    1.00   |  0.71  |   0.83  |    14   |
| Viatcheslav Molotov     |    1.00   |  0.23  |   0.37  |    13   |
| Georges Pompidou        |    1.00   |  0.69  |   0.81  |    13   |
| -- unknown --           |    0.35   |  0.97  |   0.52  |    39   |
| average (unknown apart) |    0.97   |  0.59  |   0.71  |   216   |


| MeMAD                    | Precision | Recall | F-Score | Support |
|--------------------------|:---------:|:------:|:-------:|:-------:|
| Le Saint, Sophie         |    0.90   |  0.90  |   0.90  |    10   |
| Delahousse, Laurent      |    1.00   |  1.00  |   1.00  |    7    |
| Lucet, Elise             |    1.00   |  0.90  |   0.94  |    10   |
| Gastrin, Sophie          |    1.00   |  0.90  |   0.94  |    10   |
| Rincquesen, Nathanaël de |    1.00   |  0.80  |   0.88  |    10   |
| Drucker, Marie           |    1.00   |  1.00  |   1.00  |    10   |
| -- unknown ---           |    0.89   |  0.97  |   0.93  |    43   |
| average (unknown apart)  |    0.98   |  0.91  |   0.94  |   100   |