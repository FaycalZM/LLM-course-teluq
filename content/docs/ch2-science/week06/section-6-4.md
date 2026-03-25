---
title: "6.4 Fine-tuning pour retrieval "
weight: 5
---

## Pourquoi un modèle généraliste ne suffit pas toujours
Bonjour à toutes et à tous ! Nous arrivons au sommet de notre semaine sur la recherche sémantique. Jusqu'ici, nous avons utilisé des modèles "sur l'étagère" (*off-the-shelf*), comme des couteaux suisses capables de couper un peu de tout. Mais imaginez que vous deviez opérer un patient : utiliseriez-vous un couteau suisse ou un scalpel de précision ? 

> [!IMPORTANT]
🔑 **Je dois insister :** dans des domaines pointus comme la médecine, le droit ou la maintenance industrielle, un modèle généraliste va passer à côté des nuances cruciales. Aujourd'hui, nous allons apprendre à forger votre propre scalpel sémantique grâce au **fine-tuning pour le retrieval**.


Le problème fondamental est le suivant : un modèle comme `all-mpnet-base-v2` a été entraîné sur des données web générales (Wikipédia, Reddit). Il "sait" que "chat" est proche de "félin". Mais sait-il que dans votre base de données technique, le code d'erreur `E104` est sémantiquement lié à une "surchauffe de la pompe hydraulique" ? Probablement pas. Sans entraînement spécifique, la distance vectorielle entre ces deux concepts sera trop grande, et votre moteur de recherche échouera. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'ingénieurs pensent qu'il faut changer de modèle alors qu'il suffit souvent d'ajuster les poids du modèle actuel sur quelques centaines d'exemples métier.


## La géométrie du changement : Analyser les Figures 6-6 et 6-7
Pour bien comprendre ce qui se passe dans "le cerveau" du modèle pendant cet entraînement, regardons les schémas de l'espace vectoriel suivants:

*   [**Figure 6-6 : État avant le fine-tuning**]({{< relref "section-6-2.md" >}}#fig-6-6) : Imaginez une carte du ciel. Vous avez une question (une requête) au centre. Autour d'elle, à des distances presque égales, gravitent plusieurs documents. Certains sont les bonnes réponses (exemples positifs), d'autres sont des hors-sujets qui partagent juste quelques mots en commun (exemples négatifs). Le modèle est confus : il ne "sent" pas la différence de pertinence. Tout se ressemble à ses yeux car il n'a pas appris votre contexte spécifique.
*   [**Figure 6-7 : État après le fine-tuning**]({{< relref "section-6-2.md" >}}#fig-6-7) : C'est ici que la magie opère. Après l'entraînement, la carte a été redessinée. Les documents pertinents ont été "aspirés" vers la requête (la distance diminue, le score de similarité augmente). À l'inverse, les documents non-pertinents ont été "repoussés" vers les bords de la galaxie. 

> [!NOTE]
🔑 **Notez bien cette intuition :** le fine-tuning est un processus de tension géométrique. On rapproche les alliés et on éloigne les intrus.


## La préparation des données : Triplets et paires contrastives
Comment dire au modèle qui rapprocher et qui repousser ?

On ne lui donne pas des notes, on lui donne des **exemples de comparaison**. La méthode la plus robuste utilise des **triplets** (Ancre, Positif, Négatif) :
1.  **L'Ancre (Anchor)** : C'est la requête type de l'utilisateur (ex: "Comment réinitialiser le mot de passe ?").
2.  **Le Positif (Positive)** : C'est le paragraphe qui contient la vraie solution.
3.  **Le Négatif (Negative)** : C'est un document qui ressemble à la question mais ne contient pas la réponse.

> [!TIP]
🔑 **Le concept avancé : Les "Hard Negatives" (Négatifs difficiles)**. 

> C'est le secret des meilleurs moteurs de recherche. Si vous donnez au modèle un négatif totalement absurde (ex: une recette de cuisine pour une question sur l'informatique), il n'apprendra rien. C'est trop facile. Pour qu'il devienne un expert, vous devez lui donner des documents qui parlent du même sujet mais qui sont techniquement incorrects pour cette question précise. C'est en échouant sur ces nuances qu'il affine sa compréhension.


## Les fonctions de perte : Multiple Negatives Ranking (MNR) Loss
Mathématiquement, pour réaliser ce "rapprochement/éloignement", nous utilisons souvent la **MNR Loss**. 
L'idée est brillante : dans un lot (*batch*) de données, on suppose que pour chaque question, il n'y a qu'une seule bonne réponse parmi toutes celles présentes dans le lot. Le modèle essaie de maximiser la similarité de la paire correcte tout en minimisant celle de toutes les autres combinaisons possibles. C'est une forme d'apprentissage par élimination extrêmement efficace sur GPU.


## Laboratoire de code : Fine-tuning rapide avec Sentence-Transformers
Voici comment transformer un modèle généraliste en un expert sur un petit jeu de données de support technique.

```python
# Testé pour Google Colab T4 16GB VRAM
# !pip install sentence-transformers datasets

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Chargement d'un modèle de base performant
model = SentenceTransformer('all-mpnet-base-v2')

# 2. Préparation de nos données métier (Triplets ou Paires)
train_examples = [
    InputExample(texts=['My internet is slow', 'Check your router settings and cables.'], label=0.9),
    InputExample(texts=['Blue screen error', 'This usually indicates a kernel panic or hardware failure.'], label=0.9),
    InputExample(texts=['How to pay my bill?', 'You can access the payment portal in your account settings.'], label=0.9),
    # On pourrait ajouter des exemples négatifs avec un label de 0.1
]

# 3. Création du DataLoader (gestion des lots pour le GPU)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 4. Définition de la fonction de perte (Loss)
# La CosineSimilarityLoss est parfaite pour débuter avec des scores de 0 à 1
train_loss = losses.CosineSimilarityLoss(model)

# 5. L'entraînement (Le Fine-tuning)
print("Début de l'entraînement métier...")
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)

# 6. Test après entraînement
# Le modèle est maintenant plus sensible à vos termes spécifiques
print("Modèle prêt pour votre domaine !")
```


## Évaluation : Comment savoir si on a progressé ?

> [!WARNING]
⚠️ Ne vous contentez pas de regarder la perte descendre pendant l'entraînement. C'est un mirage. 

> Pour valider un système de recherche, vous devez utiliser des métriques spécifiques au domaine du *Information Retrieval* :
> 1.  **Hit Rate @ K** : "Dans mon top K résultats (ex: top 5), est-ce que la bonne réponse est présente au moins une fois ?"
> 2.  **MRR (Mean Reciprocal Rank)** : Cette métrique est plus sévère. Elle vous donne plus de points si la bonne réponse est en 1ère position qu'en 5ème. 

> [!TIP]
> 🔑 **Je dois insister :** en recherche sémantique, l'utilisateur regarde rarement au-delà des trois premiers résultats. Le MRR est votre juge de paix.


## Éthique et Responsabilité : Le danger du sur-apprentissage (Overfitting)

> [!CAUTION]
⚠️ Mes chers étudiants, un modèle trop spécialisé devient un modèle aveugle.

Si vous entraînez trop fort votre modèle sur vos données internes, il risque de perdre ses connaissances générales. 
> 1.  **Le biais de domaine** : Si vous lui apprenez que "virus" désigne uniquement un logiciel malveillant, il pourrait devenir incapable de trouver des documents sur la biologie, ce qui peut être problématique si votre entreprise est dans la santé.
> 2.  **Confidentialité des données d'entraînement** :
> ⚠️ **Point crucial !** Si vos questions-réponses d'entraînement contiennent des secrets industriels ou des noms de clients, ces informations peuvent être "mémorisées" par le modèle dans ses poids. Un attaquant pourrait potentiellement extraire des fragments de ces données en interrogeant finement le modèle. 

> [!TIP]
🔑 **Mon message** : Le fine-tuning est l'acte final de l'artisan. Il donne au modèle son caractère unique. Mais un bon artisan sait aussi quand s'arrêter pour ne pas fragiliser la structure globale. Spécialisez avec précision, mais évaluez avec rigueur.


---
Nous avons terminé notre exploration de la recherche sémantique ! Vous savez désormais transformer du texte en vecteurs, mesurer leur similarité, construire une architecture à grande échelle et même entraîner le modèle pour qu'il devienne un expert. La semaine prochaine, nous irons encore plus loin en apprenant à regrouper ces documents automatiquement par thématiques : place au **Clustering** et à **BERTopic**.