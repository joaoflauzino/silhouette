# silhouette

- Para executar o código é necessário:

    - incluir o arquivo que será processado na pasta dataset.

    - informar os argumentos de acordo com a lista abaixo:

        inputFile - o nome do arquivo que será processado. É importante que ele não tenha espaços no nome, pois o código entenderá como um novo argumento e dará erro. ("iris.csv" é o padrão)
        
        sep - identificador de nova coluna ("," é o padrão)
        
        dec - identificador de casa decimal ("." é o padrão)

    - exemplo de chamada do código no prompt:
        python purity.py -inputFile iris.csv -sep , -dec .

    - o código irá printar o resultado da silhueta para cada iteração na tela

    - é necessário que tenha as bibliotecas scipy, argparse, pandas, sklearn e numpy instaladas no ambiente, caso não tenha, instalar utilizando os seguintes comandos no prompt:

        - pip install scipy
        - pip install argparse
        - pip install pandas
        - pip install sklearn
        - pip install numpy