<img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/openai/baselines.svg?branch=master)](https://travis-ci.org/openai/baselines)
# Lignes de base

OpenAI Baselines est un ensemble d'implémentations de haute qualité d'algorithmes d'apprentissage par renforcement.

Ces algorithmes permettront aux chercheurs de reproduire, d’affiner et d’identifier de nouvelles idées plus facilement, et de créer de bonnes bases pour la recherche. Notre implémentation DQN et ses variantes sont à peu près à égalité avec les scores des articles publiés. Nous espérons qu’elles serviront de base pour l’ajout de nouvelles idées et d’outil permettant de comparer une nouvelle approche à celles existantes.

## Conditions préalables
Les lignes de base nécessitent python3 (> = 3.5) avec les en-têtes de développement. Vous aurez également besoin des packages système CMake, OpenMPI et zlib. Ceux-ci peuvent être installés comme suit
### Ubuntu
    
`` `bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
`` `
    
### Mac OS X
L'installation des packages système sur Mac nécessite [Homebrew] (https://brew.sh). Avec Homebrew installé, exécutez ce qui suit:
`` `bash
brasseur installer cmake openmpi
`` `
    
## Environnement virtuel
Du point de vue général de l'intégrité des paquets python, il est judicieux d'utiliser des environnements virtuels (virtualenvs) pour s'assurer que les packages de différents projets n'interfèrent pas les uns avec les autres. Vous pouvez installer virtualenv (qui est lui-même un paquet pip) via
`` `bash
pip installer virtualenv
`` `
Les virtualenv sont essentiellement des dossiers contenant des copies de l'exécutable python et de tous les packages python.
Pour créer un virtualenv appelé venv avec python3, on lance
`` `bash
virtualenv / chemin / vers / venv --python = python3
`` `
Pour activer un virtualenv:
`` `
. / chemin / vers / venv / bin / activate
`` `
Un tutoriel plus complet sur virtualenvs et les options peut être trouvé [ici] (https://virtualenv.pypa.io/en/stable/)


## Installation
- Cloner le repo et le cd dans celui-ci:
    `` `bash
    clone de git https://github.com/openai/baselines.git
    cd baselines
    `` `
- Si TensorFlow n'est pas déjà installé, installez votre version préférée de TensorFlow. Dans la plupart des cas,
    `` `bash
    pip installez tensorflow-gpu # si vous avez un gpu compatible CUDA et des pilotes appropriés
    `` `
    ou
    `` `bash
    pip installer tensorflow
    `` `
    devrait être suffisant. Reportez-vous au [Guide d'installation de TensorFlow] (https://www.tensorflow.org/install/).
    pour plus de détails.

- Installer le paquet de lignes de base
    `` `bash
    pip installer -e.
    `` `
### MuJoCo
Certains exemples de références utilisent un simulateur de physique [MuJoCo] (http://www.mujoco.org) (dynamique multi-articulation en contact), qui est propriétaire et qui nécessite des binaires et une licence (une licence temporaire de 30 jours peut être obtenue auprès de [www.mujoco.org] (http://www.mujoco.org)). Des instructions sur la configuration de MuJoCo sont disponibles [ici] (https://github.com/openai/mujoco-py).

## Test de l'installation
Tous les tests unitaires des lignes de base peuvent être exécutés à l'aide de pytest runner:
`` `
pip installer pytest
pytest
`` `

## Modèles de formation
La plupart des algorithmes du référentiel de lignes de base sont utilisés comme suit:
`` `bash
python -m baselines.run --alg = <nom de l'algorithme> --env = <id_environnement> [arguments supplémentaires]
`` `
### Exemple 1. PPO avec MuJoCo Humanoid
Par exemple, pour former un réseau entièrement connecté contrôlant l'humanoïde MuJoCo à l'aide de PPO2 pendant 20 millions de secondes
`` `bash
python -m baselines.run --alg = ppo2 --env = humanoïde-v2 --network = mlp --num_timesteps = 2e7
`` `
Notez que pour les environnements mujoco un réseau entièrement connecté est la valeur par défaut, nous pouvons donc omettre `--network = mlp`
Les hyperparamètres pour le réseau et l'algorithme d'apprentissage peuvent être contrôlés via la ligne de commande, par exemple:
`` `bash
python -m baselines.run --alg = ppo2 --env = Humanoid-v2 --network = mlp --num_timesteps = 2e7 --ent_coef = 0.1 --num_hidden = 32 --num_layers = 3 --value_network = copie
`` `
définira le coefficient d'entropie sur 0,1 et construira un réseau entièrement connecté avec 3 couches de 32 unités cachées et créera un réseau séparé pour l'estimation de la fonction de valeur (afin que ses paramètres ne soient pas partagés avec le réseau de règles, mais que la structure reste la même )

Voir la documentation dans [common / models.py] (lignes de base / common / models.py) pour obtenir une description des paramètres réseau pour chaque type de modèle.
docstring pour [lignes de base / ppo2 / ppo2.py / learn ()] (lignes de base / ppo2 / ppo2.py # L152) pour la description des hyperparamères de ppo2.

### Exemple 2. DQN sur Atari
DQN avec Atari est à ce stade un classique des points de repère. Pour exécuter l'implémentation de base de DQN sur Atari Pong:
`` `
python -m baselines.run --alg = deepq --env = PongNoFrameskip-v4 --num_timesteps = 1e6
`` `

## Enregistrement, chargement et visualisation de modèles
L'API de sérialisation des algorithmes n'est pas encore unifiée correctement; Cependant, il existe une méthode simple pour enregistrer / restaurer les modèles formés.
L'option de ligne de commande `--save_path` et` --load_path` charge l'état de tensorflow à partir d'un chemin donné avant l'entraînement et l'enregistre après l'entraînement, respectivement.
Imaginons que vous souhaitiez former ppo2 sur Atari Pong, enregistrer le modèle et ensuite visualiser ce qu'il a appris.
`` `bash
python -m baselines.run --alg = ppo2 --env = PongNoFrameskip-v4 --num_timesteps = 2e7 --save_path = ~ / models / pong_20M_ppo2
`` `
Cela devrait donner la récompense moyenne par épisode d'environ 20. Pour charger et visualiser le modèle, nous allons procéder comme suit: chargez le modèle, entraînez-le pendant 0 étapes, puis visualisez:
`` `bash
python -m baselines.run --alg = ppo2 --env = PongNoFrameskip-v4 --num_timesteps = 0 --load_path = ~ / models / pong_20M_ppo2 --play
`` `

* REMARQUE: * Pour le moment, la formation Mujoco utilise le wrapper VecNormalize pour l'environnement qui n'est pas correctement enregistré. le chargement des modèles formés sur Mujoco ne fonctionnera donc pas correctement si l'environnement est recréé. Si nécessaire, vous pouvez contourner ce problème en remplaçant RunningMeanStd par TfRunningMeanStd dans [baselines / common / vec_env / vec_normalize.py] (baselines / common / vec_env / vec_normalize.py # L12). De cette façon, les enveloppes de normalisation de la moyenne et de l’environnement seront enregistrées dans des variables tensorflow et incluses dans le fichier de modèle; cependant, la formation est plus lente de cette façon - donc ne l'incluez pas par défaut


## Utilisation des lignes de base avec TensorBoard
L'enregistreur de lignes de base peut enregistrer des données au format TensorBoard. Pour ce faire, définissez les variables d'environnement `OPENAI_LOG_FORMAT` et` OPENAI_LOGDIR`:
`` `bash
export OPENAI_LOG_FORMAT = 'Les formats # sont séparés par des virgules, journal, csv, tensorboard', mais pour le tensorboard, vous n'avez vraiment besoin que du dernier
export OPENAI_LOGDIR = chemin / vers / tensorboard / data
`` `
Et vous pouvez maintenant démarrer TensorBoard avec:
`` `bash
tensorboard --logdir = $ OPENAI_LOGDIR
`` `## Sous-paquets

- [A2C] (lignes de base / a2c)
- [ACER] (lignes de base / acer)
- [ACKTR] (lignes de base / acktr)
- [DDPG] (baselines / ddpg)
- [DQN] (lignes de base / deepq)
- [GAIL] (lignes de base / gail)
- [HER] (lignes de base / elle)
- [PPO1] (lignes de base / ppo1) (version obsolète, laissée ici temporairement)
- [PPO2] (lignes de base / ppo2)
- [TRPO] (baselines / trpo_mpi)



## Benchmarks
Les résultats des tests sur Mujoco (1 M timesteps) et Atari (10 M timesteps) sont disponibles
[ici pour Mujoco] (https://htmlpreview.github.com/?https://github.com/openai/baselines/blob/master/benchmarks_mujoco1M.htm)
et
[ici pour Atari] (https://htmlpreview.github.com/?https://github.com/openai/baselines/blob/master/benchmarks_atari10M.htm)
respectivement. Notez que ces résultats peuvent ne pas figurer sur la dernière version du code, un hachage de validation particulier avec lequel les résultats ont été obtenus est spécifié sur la page des tests.

Pour citer ce référentiel dans des publications:

    @misc {lignes de base,
      auteur = {Dhariwal, Prafulla et Hesse, Christopher et Klimov, Oleg et Nichol, Alex et Plappert, Matthias et Radford, Alec et Schulman, John et Sidor, Szymon et Wu, Yuhuai et Zhokhov, Peter},
      titre = {OpenAI Baselines},
      année = {2017},
      éditeur = {GitHub},
      journal = {référentiel GitHub},
      howpublished = {\ url {https://github.com/openai/baselines}},
    }
