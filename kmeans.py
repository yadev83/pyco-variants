#représentation des individus dans le tableau IND :
#T[[attr1, attr2, attr..., attrn], [attr1, attr2, attr..., attrn], [attr1, attr2, attr..., attrn]]

import matplotlib.pyplot as plt
import numpy as np
import random

### DEFINITION ET ACTUALISATION DES CENTROIDES ###

#@fct    = init_centres() -> 
# pioche K centroïdes aléatoirement dans le tableau IND
#@param  = IND - Le tableau d'individus à étudier, de la forme précédement décrite en haut de fichier.
#@param  = k   - Le nombre de centroïdes voulus (donc de clusters à étudier)
#@return = C   - Un tableau 2D contenant k Centroïdes avec IND.shape[1] attributs.
def init_centres(IND, k):
    C = np.zeros((k, IND.shape[1]))
    for i in range(k):
        #on choisit de manière complétement aléatoire k centroides dans le tableau d'individus
        if(IND.shape[0] != 0):
            C[i] = IND[random.randint(0, IND.shape[0]-1)]
        else:
            C[i] = 0
    return C

#@fct   = update_centres() -> 
# fait la moyenne des attributs d'individus dans un cluster puis
# redéfinit les centroïdes en conséquence.
#@param = IND     - Le tableau d'individus à étudier, de la forme précédement décrite en haut de fichier.
#@param = CLUSTER - Tableau contenant les indicies permettant de relier chaque individu à son cluster.
#@param = C       - Un tableau 2D k Centroïdes avec IND.shape[1] attributs.
def update_centres(IND, CLUSTER, C):
    k = C.shape[0]
    CPT = np.zeros(k)
    NEW = np.zeros((k, IND.shape[1]))

    for i in range(k):
        #on compte le nombre d'individu appartenant à chaque cluster
        CPT[i] = np.sum(CLUSTER==i)
    indice = 0
    for i in range(k):
        #on prépare un tableau d'individus contenus dans le cluster "i"
        TMP = np.zeros((int(CPT[i]), IND.shape[1]))
        for j in range(IND.shape[0]):
            if(CLUSTER[j] == i):
                TMP[indice] = IND[j]
                indice = indice+1
        #une fois le tableau bien initialisé, on fait la moyenne des individus du cluster "i"
        #puis on réitère sur les k clusters passés en paramètres.
        NEW[i] = np.mean(TMP, axis=0)
        indice = 0

    return NEW


#@fct    = calc_clusters() ->
# Calcule la distance entre chaque individu et chaque centroïde de
# sorte à trouver la plus courte et associer un centroïde à chaque individu
#@param  = IND     - Le tableau d'individus à étudier, de la forme précédement décrite en haut de fichier.
#@param  = C       - Un tableau 2D k Centroïdes avec IND.shape[1] attributs.
#@return = CLUSTER - Tableau contenant les indicies permettant de relier chaque individu à son cluster.
def calc_clusters(IND, C):
    #on prépare un tableau qui contiendra les indices associant les individus à leur cluster
    CLUSTER = np.zeros(IND.shape[0])

    for i in range(IND.shape[0]):
        #calcule la distance entre chaque centroïde et chaque individu
        for k in range(C.shape[0]):
            if k == 0:
                distance = dist(IND[i], C[k])
                id_dist = k
            else:
                if(dist(IND[i], C[k]) <= distance):
                    distance = dist(IND[i], C[k])
                    id_dist = k
        #on assigne à l'individu un cluster pour lequel la distance est la plus courte
        CLUSTER[i] = id_dist  # OK

    return CLUSTER

#@fct    = draw_clusters() ->
# Affiche les centroïdes ainsi que les points leur étant associés pour les
# attributs I1 et I2. (on admet que I1 et I2 sont < à la taille max du tableau d'attributs...)
#@param  = IND      - Le tableau d'individus à étudier, de la forme précédement décrite en haut de fichier.
#@param  = CLUSTER  - Tableau contenant les indicies permettant de relier chaque individu à son cluster.
#@param  = C        - Un tableau 2D k Centroïdes avec IND.shape[1] attributs.
#@param  = (I1, I2) - Les indices des attributs selon lesquelles on affiche les données à l'écran.
def draw_clusters(IND, CLUSTER, C, GRAPH_NAME="Algorithme K-MEANS", I1_NAME="X", I2_NAME="Y", I1=0, I2=1):
    #on récupère le nombre d'individus
    size = CLUSTER.shape[0]
    #un tableau de couleurs pour différencier les clusters
    colors = ['blue', 'red', 'green', 'purple', 'yellow', 'brown']
    for i in range(size):
        #on affiche chaque individu en y affectant une couleur selon le cluster dont il fait partie
        plt.scatter(IND[i][I1], IND[i][I2], color=colors[int(CLUSTER[i])%6], marker='.')

    for i in range(C.shape[0]):
        #on affiche les centroïdes de chaque cluster (en noir, et sous forme d'une étoile)
        plt.scatter(C[i][I1], C[i][I2], color='black', marker='*')
        
    plt.xlabel(I1_NAME)
    plt.ylabel(I2_NAME)
    plt.title(GRAPH_NAME)
    plt.show()

#@fct    = dist() ->
# Renvoie la distance entre deux individus distincts
#@param  = (IND1, IND2) - Les individus que l'on souhaite comparer.
#@return = ___          - La distance qui sépare IND1 de IND2
def dist(IND1, IND2):
    #on récupère le nombre d'attributs
    size = IND1.shape[0]
    somme = 0
    for i in range(size):
        #on ajoute la distance entre chaque attribut à la somme
        #c'est la distance euclidienne
        somme += (IND1[i] - IND2[i])**2
    #application de la racine carrée avant le return (cf. formule Distance Euclidienne)
    return ((somme)**.5)

#@fct    = genVal() ->
# Renvoie un tableau d'individus aléatoires en fonction des paramètres donnés
# A utiliser pour obtenir des sets de données aléatoires (principalement pour tester des choses)
#@param  = NB   - Le nombre d'individus à créer.
#@param  = SIZE - Le nombre d'attributs à associer à chaque individu
#@param  = MAX  - La valeur MAX que pourra prendre le rng.
def genVal(NB, SIZE, MAX):
    IND = np.zeros((NB, SIZE))
    for i in range(NB):
        IND[i] = np.random.randint(MAX, size=SIZE)
    return IND

#tant que le centroïde change après un appel à update_centres(), on appelle update_centres()
#cpt permet de garder une trace du nombre de fois ou le changement à lieu.
#N.B : Pour un petit set de donnée, les centroïdes ne sont pas beaucoup édités, 1 à 2 fois seulement.
#c'est à croire que les premiers calculs de clusters suffisent souvent !
def kmeans(X, k, limit=-1):
    C = init_centres(X, k)
    CLUSTER = calc_clusters(X, C)

    diff = 1
    cpt = 0
    while(diff):
        NEW_C = update_centres(X, CLUSTER, C)
        # Si les nouveaux centres sont les mêmes que les anciens, on sort de l'algo
        if(np.array_equal(NEW_C, C) == True or (limit != -1 and (cpt == limit))):
            diff = 0
        else:
            C = NEW_C
            cpt += 1
    print(f'Sortie de l\'algo après avoir redéfini {cpt} fois les centres')
    return (CLUSTER, C)

