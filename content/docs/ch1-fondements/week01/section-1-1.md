---
title: "1.1 Évolution du Traitement du Langage Naturel (NLP)"
weight: 2
---

## L'aube du Language AI : L'illusion des règles et la réalité statistique

Pour comprendre pourquoi les modèles que nous utilisons aujourd'hui (comme GPT-4 ou Claude) sont si performants, il faut d'abord réaliser que pendant quarante ans, nous avons traité le langage comme un simple puzzle de symboles rigides.

Comme vous pouvez l'observer sur la **Figure 1-1 : Timeline historique du NLP**, tout commence par des approches basées sur des règles manuelles. Dans les années 1950 et 1960, on pensait qu'il suffisait de coder toutes les règles de grammaire et tous les mots d'un dictionnaire pour qu'une machine "comprenne". 🔑 **Je dois insister :** cette approche symbolique était condamnée. Pourquoi ? Parce que le langage humain n'est pas un code informatique stable. Il est vivant, pétri d'ambiguïtés, d'ironie et de contextes culturels changeants. Essayer de coder le langage avec des instructions `if/then` (si/alors), c'est comme essayer de vider l'océan avec une petite cuillère.

{{< bookfig src="5.png" week="01" >}}

À partir des années 1990, un changement radical s'opère : nous cessons de dire à la machine *comment* le langage fonctionne, et nous commençons à lui montrer d'immenses quantités de textes pour qu'elle apprenne les statistiques d'usage. C'est l'essor du NLP statistique, capable de réaliser les **tâches typiques du Language AI** illustrées en **Figure 1-2** : la classification de spams, l'analyse de sentiments ou la traduction automatique rudimentaire.

{{< bookfig src="6.png" week="01" >}}

## Le mécanisme de la "Sacoche de mots" (Bag-of-Words)

C'est le point de départ technique de notre voyage. Imaginez que vous ayez un texte et que vous décidiez d'ignorer totalement la syntaxe, la conjugaison et l'ordre des mots. Vous jetez chaque mot dans un sac et vous comptez simplement combien de fois il apparaît. C'est ce qu'on appelle le **Bag-of-Words (BoW)**.

Le processus, méticuleusement détaillé dans les **Figures 1-3 à 1-5**, se déroule en trois étapes cruciales que vous devez maîtriser :
1.  **Tokenisation** : On découpe la phrase en morceaux de base (tokens). Pour les modèles simples, un token est souvent égal à un mot.
2.  **Construction du vocabulaire** : On liste tous les mots uniques rencontrés dans l'ensemble de nos textes (le corpus). Si nous avons 50 000 mots différents, notre "sac" a 50 000 étagères.
3.  **Vectorisation** : Pour chaque phrase, on crée un vecteur (une suite de nombres). Si le mot "chat" est présent deux fois, on inscrit "2" à l'index correspondant au mot "chat" dans notre immense liste.

{{< bookfig src="7.png" week="01" >}}

{{< bookfig src="8.png" week="01" >}}

{{< bookfig src="9.png" week="01" >}}

{{% hint warning %}}
**Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent que le Bag-of-Words est une relique du passé. En réalité, il reste une "baseline" (référence) solide pour des tâches simples. Mais regardez bien la faille sémantique : les phrases "Le chat mange la souris" et "La souris mange le chat" produiront le *même* vecteur exact dans un modèle BoW standard. Pour la machine, le prédateur et la proie sont statistiquement identiques. On perd la structure, donc on perd le sens.
{{% /hint %}}

## De TF-IDF aux limites de la représentation creuse (Sparse)

Pour affiner le comptage, les chercheurs ont introduit le **TF-IDF** (Term Frequency-Inverse Document Frequency). L'intuition est brillante : un mot qui apparaît partout (comme "le", "et", "est") n'apporte aucune information sur le sujet d'un texte. TF-IDF punit les mots trop fréquents et valorise les mots rares et spécifiques (comme "photosynthèse" ou "algorithme").

Cependant, nous restions prisonniers des **représentations creuses (sparse)**. 🔑 **Notez bien cette distinction :** dans une représentation creuse, la taille du vecteur est égale à la taille du dictionnaire. Si votre modèle connaît 100 000 mots, chaque petit SMS de 3 mots devient un vecteur de 100 000 dimensions rempli de 99 997 zéros. C'est un gaspillage immense de puissance de calcul, et surtout, cela ne permet pas de comprendre que "maison" et "demeure" sont des synonymes, car ce sont deux colonnes totalement distinctes dans la base de données.

## La révolution de 2013 : Les Embeddings Denses (Word2Vec)

C'est ici que l'histoire s'accélère brutalement. Avec l'arrivée de **Word2Vec** (Mikolov et al., 2013), nous sommes passés de la statistique de comptage à la géométrie neuronale.

**L'intuition fondamentale** : "Vous connaîtrez un mot par l'entreprise qu'il garde" (John Rupert Firth, 1957). Au lieu de compter les mots, nous allons entraîner un petit réseau de neurones à prédire un mot en fonction de ses voisins (ou inversement).

Le résultat est l'apparition des **embeddings denses**. Au lieu d'un vecteur géant de zéros, chaque mot est représenté par un vecteur compact (généralement 300 ou 768 dimensions) de nombres réels. Comme l'illustrent les **Figures 1-6 à 1-9**, on découvre alors une véritable géométrie du langage. Dans cet espace vectoriel, les mots qui partagent un sens similaire se retrouvent physiquement proches les uns des autres. Plus incroyable encore, ces vecteurs permettent des opérations mathématiques sur les concepts :
`Vecteur(Roi) - Vecteur(Homme) + Vecteur(Femme) ≈ Vecteur(Reine)`

{{< bookfig src="10.png" week="01" >}}

{{< bookfig src="11.png" week="01" >}}

{{< bookfig src="12.png" week="01" >}}

{{< bookfig src="13.png" week="01" >}}

{{% hint info %}}
🔑 **La distinction non-négociable :** Ces embeddings sont dits **statiques**. Cela signifie que dans le modèle, le mot "avocat" n'a qu'une seule "adresse" (un seul vecteur), qu'il s'agisse du fruit ou de la profession juridique.
{{% /hint %}}

## Le mur de la polysémie : L'exemple "bank"

C'est ici que nous touchons aux limites des modèles pré-2018. Prenons l'exemple du mot anglais "bank", très cher aux chercheurs en NLP.
1. "I am going to the **bank** to withdraw money." (Institution financière)
2. "The boat is near the river **bank**." (Rive d'un cours d'eau)

Dans les approches de Word2Vec ou GloVe, le mot "bank" n'a qu'un seul vecteur. Ce vecteur est une sorte de "moyenne" confuse entre la finance et la géographie. 🔑 **Je dois insister :** c'est la limite ultime des représentations non contextuelles. La machine ne peut pas changer sa vision d'un mot en fonction de ce qui l'entoure. Il nous manquait une technologie capable de générer des embeddings *dynamiques*, capables de se transformer selon la phrase. C'est ce défi qui a pavé la voie aux Transformers que nous étudierons en section 1.3.

## Tableau comparatif : Approches Symboliques vs Neuronales

| Dimension           | Approches Symboliques/Statistiques (BoW, TF-IDF)    | Approches Neuronales (Word2Vec, GloVe)             |
| :------------------ | :-------------------------------------------------- | :------------------------------------------------- |
| **Philosophie**     | Compter les mots (Fréquence brute)                  | Apprendre les relations (Voisinage sémantique)     |
| **Type de vecteur** | **Creux (Sparse)** : immense taille, plein de zéros | **Dense** : taille compacte, nombres réels partout |
| **Sens sémantique** | Nul (chaque mot est un îlot isolé)                  | Élevé (Similarité calculable par distance)         |
| **Indépendance**    | Ne comprend pas que "chien" et "chiot" sont liés    | Regroupe les synonymes dans l'espace vectoriel     |
| **Ambiguïté**       | Échec total sur les synonymes                       | Gère les synonymes, échoue sur la polysémie        |

## Éthique et Responsabilité : Les biais dans les vecteurs

{{% hint danger %}}
Avant de clore cette section, je veux que vous compreniez une chose fondamentale. Les vecteurs neuronaux ne sont pas des entités "pures" ou "logiques". Ils sont le reflet des données sur lesquelles ils sont entraînés.

Si vous entraînez un modèle sur des textes du web qui contiennent des préjugés sexistes ou racistes, ces préjugés vont se traduire par des distances géométriques dans l'espace vectoriel. Par exemple, des études célèbres ont montré que dans certains modèles Word2Vec, le vecteur "homme" était statistiquement plus proche de "programmeur" et le vecteur "femme" de "homemaker" (femme au foyer). 🔑 **C'est une leçon d'éthique cruciale :** en tant que futurs concepteurs de LLM, vous devez être conscients que la beauté mathématique d'un vecteur dense peut cacher des biais sociétaux profonds. La science des modèles de langage commence par une analyse critique de la donnée.
{{% /hint %}}
