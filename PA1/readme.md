Libraries Used: 
 - from apyori import apriori
    Apriori is used to find the support sets between all items
 - import csv
    csv is used to read in the testData
 - import pandas as pd
    pandas is used to filter the data from the testData before the final array is built for apriori
 - import pickle
    pickle is used to dump objects(dict, list, set ...) to a file so progress can be saved. This grealy speeds
    up runtime for subsequent data processing

To use apyoriRecomendation.py just run the run_code.sh

If the program is not interrupted then the only thing that will need to be changed is filtering the results
array between support length =3 and length =2 on line 153.

If the program does get interrupted load the latest pickle save and comment out the previous code that ran. This
will enable 'saving' progress between data processing.

The original array took ~4 hours to build. Once the records array has been built once uncomment the pickle.load() on line 120-122 and comment out the code between 105-114